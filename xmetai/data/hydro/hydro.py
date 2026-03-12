"""
描述:
    基于场次的山洪数据库

训练:
    使用 history_steps 帧历史状态 + 当前时刻 forcing，
    预测从当前时刻开始的多步 [h, u, v]

测试:
    以整个事件为单位返回:
        - 初始输入: history_steps 帧历史 + t=0 forcing
        - targets : 全部 24 帧 [h, u, v]
        - future_rain : t=1..23 的未来降雨，用于 rollout
"""

from __future__ import annotations

import os
import logging
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from xmetai.builder import DATASETS


__all__ = ["FloodDataset"]


def _to_python_key(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return x


def _safe_int_last_digit(x: Any) -> int:
    try:
        return int(x) % 10
    except Exception:
        s = str(x)
        digits = [ch for ch in s if ch.isdigit()]
        return int(digits[-1])


def _nan_to_zero(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


@DATASETS.register()
class FloodDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        training: bool = True,
        history_steps: int = 2,
        train_digits: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
        test_digits: Tuple[int, ...] = (7, 8, 9),
        dem_source: str = "static",
        use_sqlite_meta: bool = True,
        allow_leading_padding: bool = True,
        consolidated: bool = False,
        test_event_limit: Optional[int] = 2,
        max_train_future_frames: int = 24,
        train_target_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.data_dir = Path(data_dir)
        self.training = bool(training)
        self.history_steps = int(history_steps)
        self.train_digits = tuple(train_digits)
        self.test_digits = tuple(test_digits)
        self.dem_source = str(dem_source).lower().strip()
        self.use_sqlite_meta = bool(use_sqlite_meta)
        self.allow_leading_padding = bool(allow_leading_padding)
        self.consolidated = bool(consolidated)
        self.test_event_limit = None if test_event_limit is None else int(test_event_limit)

        # 优先兼容 hydro_base.py 里传入的 train_target_steps
        if train_target_steps is not None:
            self.max_train_future_frames = int(train_target_steps)
        else:
            self.max_train_future_frames = int(max_train_future_frames)

        static_path = self.data_dir / "static_fp16.zarr"
        dynamic_path = self.data_dir / "dynamic_fp16.zarr"
        dem_path = self.data_dir / "dem_fp32.zarr"

        self.ds_static = xr.open_zarr(static_path, consolidated=self.consolidated)
        self.ds_dynamic = xr.open_zarr(dynamic_path, consolidated=self.consolidated)
        self.ds_dem = xr.open_zarr(dem_path, consolidated=self.consolidated) if dem_path.exists() else None

        self.history_vars: List[str] = ["rainfall_depth", "h", "u", "v"]
        self.forcing_var: str = "rainfall_depth"
        self.target_vars: List[str] = ["h", "u", "v"]

        self.static_non_dem_vars: List[str] = ["lai", "sand", "clay", "som"]

        self.input_channel_names: List[str] = ["rain", "h", "u", "v"]
        self.target_channel_names: List[str] = ["h", "u", "v"]

        all_sample_ids = (
            self.ds_static["sample"].values
            if "sample" in self.ds_static.coords
            else np.arange(self.ds_static.sizes["sample"], dtype=np.int64)
        )

        self.sample_ids: List[Any] = [_to_python_key(x) for x in all_sample_ids]
        self.valid_indices: List[int] = []
        self.valid_ids: List[Any] = []

        selected_digits = self.train_digits if self.training else self.test_digits
        for zarr_idx, sid in enumerate(self.sample_ids):
            if _safe_int_last_digit(sid) in selected_digits:
                self.valid_indices.append(zarr_idx)
                self.valid_ids.append(sid)

        if (not self.training) and (self.test_event_limit is not None):
            original_num = len(self.valid_indices)
            self.valid_indices = self.valid_indices[:self.test_event_limit]
            self.valid_ids = self.valid_ids[:self.test_event_limit]
            self.logger.info(
                f"Apply test_event_limit={self.test_event_limit}: "
                f"keep first {len(self.valid_indices)} / {original_num} test events."
            )

        self.num_events = len(self.valid_indices)
        self.max_time_steps = int(self.ds_dynamic.sizes["time"])
        self.max_train_future_frames = min(self.max_train_future_frames, self.max_time_steps)

        self.sample_meta: Dict[Any, Dict[str, Any]] = {}
        if self.use_sqlite_meta:
            self.sample_meta = self._load_sqlite_meta(self.data_dir / "run_meta.sqlite")

        mode_str = "Train" if self.training else "Test"
        static_desc = (
            ["dem"] + self.static_non_dem_vars
            if self.dem_source == "static"
            else ["dem_fp32"] + self.static_non_dem_vars
        )

        extra_desc = ""
        if (not self.training) and (self.test_event_limit is not None):
            extra_desc = f", test_event_limit={self.test_event_limit}"

        self.logger.info(
            f"FloodDataset loaded: mode={mode_str}, "
            f"events={self.num_events}, "
            f"time_steps={self.max_time_steps}, "
            f"history_steps={self.history_steps}, "
            f"dem_source={self.dem_source}, "
            f"dynamic_history={self.history_vars}, "
            f"forcing={self.forcing_var}, "
            f"target={self.target_vars}, "
            f"const={static_desc}, "
            f"max_train_future_frames={self.max_train_future_frames}"
            f"{extra_desc}"
        )

    # ------------------------------------------------------------------
    # sqlite meta
    # ------------------------------------------------------------------
    def _load_sqlite_meta(self, sqlite_path: Path) -> Dict[Any, Dict[str, Any]]:
        meta_map: Dict[Any, Dict[str, Any]] = {}
        if not sqlite_path.exists():
            return meta_map

        rows = None

        try:
            conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("SELECT sample_index, sid, num, tag, proj_x, proj_y FROM samples")
            rows = cur.fetchall()
            conn.close()
            self.logger.info(f"Loaded SQLite metadata directly from {sqlite_path}")
        except Exception as e:
            self.logger.warning(f"Direct read of SQLite metadata failed from {sqlite_path}: {e}")

        if rows is None:
            try:
                tmp_dir = Path(tempfile.gettempdir())
                tmp_sqlite = tmp_dir / f"run_meta_{os.getpid()}.sqlite"
                shutil.copy2(sqlite_path, tmp_sqlite)

                conn = sqlite3.connect(f"file:{tmp_sqlite}?mode=ro", uri=True)
                cur = conn.cursor()
                cur.execute("SELECT sample_index, sid, num, tag, proj_x, proj_y FROM samples")
                rows = cur.fetchall()
                conn.close()
                self.logger.info(f"Loaded SQLite metadata from copied temp file: {tmp_sqlite}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to read SQLite metadata from {sqlite_path} even after copying to temp: {e}"
                )
                return meta_map

        for sample_index, sid, num, tag, proj_x, proj_y in rows:
            key1 = _to_python_key(sample_index)
            meta_map[key1] = {
                "sample_index": _to_python_key(sample_index),
                "sid": sid,
                "num": _to_python_key(num),
                "tag": _to_python_key(tag),
                "x": _to_python_key(proj_x),
                "y": _to_python_key(proj_y),
            }
            if sid is not None:
                meta_map[_to_python_key(sid)] = meta_map[key1]

        return meta_map

    # ------------------------------------------------------------------
    # dataset basics
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_events

    def _resolve_event(self, idx: int) -> Tuple[int, Any]:
        logical_event_idx = int(idx)
        sample_id = self.valid_ids[logical_event_idx]
        return logical_event_idx, sample_id

    # ------------------------------------------------------------------
    # readers
    # ------------------------------------------------------------------
    def _get_const_np(self, real_zarr_idx: int) -> np.ndarray:
        static_non_dem = (
            self.ds_static["data"]
            .isel(sample=real_zarr_idx)
            .sel(level=self.static_non_dem_vars)
            .values
        )
        static_non_dem = _nan_to_zero(static_non_dem.astype(np.float32, copy=False))

        if self.dem_source == "static":
            dem = (
                self.ds_static["data"]
                .isel(sample=real_zarr_idx)
                .sel(level="dem")
                .values
            )
            dem = _nan_to_zero(dem.astype(np.float32, copy=False))
        else:
            dem = (
                self.ds_dem["data"]
                .isel(sample=real_zarr_idx)
                .sel(level="dem")
                .values
            )
            if dem.ndim == 3:
                dem = dem[0]
            dem = _nan_to_zero(dem.astype(np.float32, copy=False))

        const_np = np.concatenate([dem[np.newaxis, ...], static_non_dem], axis=0)
        return const_np

    def _get_dynamic_slice_with_left_padding(
        self,
        real_zarr_idx: int,
        levels: List[str],
        t_start: int,
        t_end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        window = int(t_end - t_start + 1)
        H = int(self.ds_dynamic.sizes["y"])
        W = int(self.ds_dynamic.sizes["x"])
        C = len(levels)

        out = np.zeros((window, C, H, W), dtype=np.float32)
        valid_mask = np.zeros((window,), dtype=np.float32)

        valid_t = [t for t in range(t_start, t_end + 1) if 0 <= t < self.max_time_steps]
        if len(valid_t) == 0:
            return out, valid_mask

        arr = (
            self.ds_dynamic["data"]
            .isel(sample=real_zarr_idx, time=valid_t)
            .sel(level=levels)
            .values
        )
        arr = _nan_to_zero(arr.astype(np.float32, copy=False))

        if len(valid_t) == 1 and arr.ndim == 3:
            arr = arr[np.newaxis, ...]

        fill_len = len(valid_t)
        out[-fill_len:] = arr
        valid_mask[-fill_len:] = 1.0
        return out, valid_mask

    # ------------------------------------------------------------------
    # item builder
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        logical_event_idx, sample_id = self._resolve_event(idx)
        real_zarr_idx = int(self.valid_indices[logical_event_idx])

        const_np = self._get_const_np(real_zarr_idx)
        meta = self.sample_meta.get(sample_id, self.sample_meta.get(real_zarr_idx, {}))
        meta_num = meta.get("num", None)
        meta_tag = meta.get("tag", None)
        meta_x = meta.get("x", None)
        meta_y = meta.get("y", None)

        # ---------------------------------------------------------
        # training mode: random start t + multi-step targets
        # ---------------------------------------------------------
        if self.training:
            train_future_frames = self.max_train_future_frames

            # 保证 target_t + train_future_frames - 1 不越界
            max_start_t = max(self.max_time_steps - train_future_frames, 0)
            target_t = int(np.random.randint(0, max_start_t + 1))

            hist_start = target_t - self.history_steps
            hist_end = target_t - 1
            history_np, history_mask_np = self._get_dynamic_slice_with_left_padding(
                real_zarr_idx=real_zarr_idx,
                levels=self.history_vars,
                t_start=hist_start,
                t_end=hist_end,
            )

            forcing_np, _ = self._get_dynamic_slice_with_left_padding(
                real_zarr_idx=real_zarr_idx,
                levels=[self.forcing_var],
                t_start=target_t,
                t_end=target_t,
            )
            forcing_np = forcing_np[0]  # [1, H, W]

            targets_np, _ = self._get_dynamic_slice_with_left_padding(
                real_zarr_idx=real_zarr_idx,
                levels=self.target_vars,
                t_start=target_t,
                t_end=target_t + train_future_frames - 1,
            )  # [Tf, 3, H, W]

            H, W = history_np.shape[-2], history_np.shape[-1]
            current_frame_np = np.zeros((1, 4, H, W), dtype=np.float32)
            current_frame_np[0, 0:1] = forcing_np
            inputs_np = np.concatenate([history_np, current_frame_np], axis=0)

            result = dict(
                inputs=torch.from_numpy(inputs_np).float(),
                targets=torch.from_numpy(targets_np).float(),
                const=torch.from_numpy(const_np).float(),
                history=torch.from_numpy(history_np).float(),
                forcing=torch.from_numpy(forcing_np).float(),
                step=torch.tensor(target_t, dtype=torch.long),
                history_mask=torch.from_numpy(history_mask_np).float(),
                input_channel_names=self.input_channel_names,
                target_channel_names=self.target_channel_names,
                const_channel_names=[
                    "dem" if self.dem_source == "static" else "dem_fp32",
                    "lai",
                    "sand",
                    "clay",
                    "som",
                ],
                meta_sample_id=sample_id,
                meta_zarr_idx=real_zarr_idx,
                meta_num=meta_num,
                meta_tag=meta_tag,
                meta_x=meta_x,
                meta_y=meta_y,
            )

            # 第1个 target 对应的雨已经放进 inputs 最后一帧里了
            # 所以后续 rollout 只需要 target_t+1 到 target_t+Tf-1 的未来雨
            if train_future_frames > 1:
                future_rain_np, _ = self._get_dynamic_slice_with_left_padding(
                    real_zarr_idx=real_zarr_idx,
                    levels=[self.forcing_var],
                    t_start=target_t + 1,
                    t_end=target_t + train_future_frames - 1,
                )  # [Tf-1, 1, H, W]
                result["future_rain"] = torch.from_numpy(future_rain_np).float()

            return result

        # ---------------------------------------------------------
        # test / inference mode: whole event
        # ---------------------------------------------------------
        target_t0 = 0

        hist_start = target_t0 - self.history_steps
        hist_end = target_t0 - 1
        history_np, history_mask_np = self._get_dynamic_slice_with_left_padding(
            real_zarr_idx=real_zarr_idx,
            levels=self.history_vars,
            t_start=hist_start,
            t_end=hist_end,
        )

        forcing0_np, _ = self._get_dynamic_slice_with_left_padding(
            real_zarr_idx=real_zarr_idx,
            levels=[self.forcing_var],
            t_start=0,
            t_end=0,
        )
        forcing0_np = forcing0_np[0]

        H, W = history_np.shape[-2], history_np.shape[-1]
        current_frame_np = np.zeros((1, 4, H, W), dtype=np.float32)
        current_frame_np[0, 0:1] = forcing0_np
        inputs_np = np.concatenate([history_np, current_frame_np], axis=0)

        targets_np, _ = self._get_dynamic_slice_with_left_padding(
            real_zarr_idx=real_zarr_idx,
            levels=self.target_vars,
            t_start=0,
            t_end=self.max_time_steps - 1,
        )  # [T, 3, H, W]

        result = dict(
            inputs=torch.from_numpy(inputs_np).float(),
            targets=torch.from_numpy(targets_np).float(),
            const=torch.from_numpy(const_np).float(),
            history=torch.from_numpy(history_np).float(),
            forcing=torch.from_numpy(forcing0_np).float(),
            step=torch.tensor(0, dtype=torch.long),
            history_mask=torch.from_numpy(history_mask_np).float(),
            input_channel_names=self.input_channel_names,
            target_channel_names=self.target_channel_names,
            const_channel_names=[
                "dem" if self.dem_source == "static" else "dem_fp32",
                "lai",
                "sand",
                "clay",
                "som",
            ],
            meta_sample_id=sample_id,
            meta_zarr_idx=real_zarr_idx,
            meta_num=meta_num,
            meta_tag=meta_tag,
            meta_x=meta_x,
            meta_y=meta_y,
        )

        # 同理，t=0 的雨已经放在 inputs 最后一帧
        # rollout 只需要 t=1..T-1 的未来雨
        if self.max_time_steps > 1:
            future_rain_np, _ = self._get_dynamic_slice_with_left_padding(
                real_zarr_idx=real_zarr_idx,
                levels=[self.forcing_var],
                t_start=1,
                t_end=self.max_time_steps - 1,
            )  # [T-1, 1, H, W]
            result["future_rain"] = torch.from_numpy(future_rain_np).float()

        return result