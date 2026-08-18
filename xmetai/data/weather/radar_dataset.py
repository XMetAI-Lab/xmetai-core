"""PyTorch Dataset for CINRAD composite reflectivity grid products."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from xmetai.registries import DATASETS
from xmetai.utils.radar import (
    RadarQCConfig,
    _build_qc_config,
    _crop_to_slices,
    _make_sequence_indices,
    _slice_spatial_array,
    _validate_common_grid,
    bbox_to_slices,
    discover_radar_files,
    read_radar_reflectivity,
)

__all__ = ["RadarDataset"]


def _normalize_crop_size(random_crop_size: Sequence[int] | int | None) -> tuple[int, int] | None:
    if random_crop_size is None:
        return None
    if isinstance(random_crop_size, int):
        crop_h = crop_w = int(random_crop_size)
    else:
        if len(random_crop_size) != 2:
            raise ValueError(f"random_crop_size must be an int or (height, width), got {random_crop_size}")
        crop_h, crop_w = [int(v) for v in random_crop_size]
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"random_crop_size values must be positive, got {(crop_h, crop_w)}")
    return crop_h, crop_w


@DATASETS.register()
class RadarDataset(Dataset):
    """PyTorch Dataset for time-ordered CINRAD composite reflectivity files."""

    def __init__(
        self,
        data_paths: str | Path | Iterable[str | Path],
        hist_frames: int,
        fcst_frames: int | Sequence[int],
        start_frames: int = 0,
        freq: int = 6,
        freq_unit: str = "minutes",
        interval: int = 1,
        training: bool = True,
        inference_only: bool = False,
        years: Sequence[object] | None = None,
        pattern: str = "*.bin",
        lat_flip: bool = False,
        bbox: Sequence[float] | None = None,
        crop: Sequence[int] | None = None,
        stride: int = 1,
        scale_factor: float = 10.0,
        fill_value: float | None = 0.0,
        qc: bool | RadarQCConfig | dict = True,
        qc_config: RadarQCConfig | dict | None = None,
        clutter_mask_path: str | Path | None = None,
        random_crop_size: Sequence[int] | int | None = None,
        echo_crop_prob: float = 0.7,
        echo_threshold_dbz: float = 20.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading radar data from: {data_paths}")

        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.atleast_1d(np.asarray(fcst_frames, dtype=np.int32)) + self.start_frames
        self.training = training
        self.inference_only = inference_only
        self.lat_flip = lat_flip
        self.bbox = bbox
        self.crop = crop
        self.stride = stride
        self.scale_factor = scale_factor
        self.fill_value = fill_value
        self.qc_config = _build_qc_config(qc, qc_config)
        self.random_crop_size = _normalize_crop_size(random_crop_size)
        self.echo_crop_prob = float(echo_crop_prob)
        self.echo_threshold_dbz = float(echo_threshold_dbz)

        if self.hist_frames <= 0:
            raise ValueError(f"hist_frames must be positive, got {self.hist_frames}")
        if np.any(self.fcst_frames <= 0):
            raise ValueError(f"fcst_frames + start_frames must be positive, got {self.fcst_frames.tolist()}")
        if self.training and len(self.fcst_frames) != 1:
            raise ValueError("RadarDataset training mode expects a single fcst_frames value")
        if bbox is not None and crop is not None:
            raise ValueError("bbox and crop are mutually exclusive")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not 0.0 <= self.echo_crop_prob <= 1.0:
            raise ValueError(f"echo_crop_prob must be in [0, 1], got {echo_crop_prob}")

        self.records = discover_radar_files(data_paths, pattern=pattern, years=years)
        self.header = _validate_common_grid(self.records)
        self.times = [record.time for record in self.records]
        self.static_clutter_mask = self._load_static_clutter_mask(clutter_mask_path)

        if self.inference_only:
            self.total_frames = self.hist_frames
        elif self.training:
            self.total_frames = self.hist_frames + int(self.fcst_frames[0])
        else:
            self.total_frames = self.hist_frames + int(self.fcst_frames.max())

        step = pd.Timedelta(**{freq_unit: freq})
        self.inds = _make_sequence_indices(self.times, self.total_frames, step=step, interval=interval)
        if len(self.inds) == 0:
            raise ValueError(f"No regular radar sequences found with total_frames={self.total_frames}, freq={freq} {freq_unit}")

        self.id_to_time = {i: self.times[i + self.hist_frames - 1] for i in self.inds}
        self.start_time = self.id_to_time[int(self.inds[0])]
        self.end_time = self.id_to_time[int(self.inds[-1])]

        lat = self.header.lat
        lon = self.header.lon
        if self.bbox is not None:
            row_slice, col_slice = bbox_to_slices(self.header, self.bbox)
            lat = lat[row_slice]
            lon = lon[col_slice]
        elif self.crop is not None:
            row_slice, col_slice = _crop_to_slices(self.crop)
            lat = lat[row_slice]
            lon = lon[col_slice]
        if self.stride > 1:
            lat = lat[:: self.stride]
            lon = lon[:: self.stride]
        if self.lat_flip:
            lat = lat[::-1].copy()
        if self.random_crop_size is not None:
            crop_h, crop_w = self.random_crop_size
            if crop_h > len(lat) or crop_w > len(lon):
                raise ValueError(f"random_crop_size {(crop_h, crop_w)} exceeds sliced radar grid {(len(lat), len(lon))}")
            lat = lat[:crop_h]
            lon = lon[:crop_w]
        self.lat = lat.astype(np.float32, copy=False)
        self.lon = lon.astype(np.float32, copy=False)
        self.grid_shape = torch.tensor([len(self.lat), len(self.lon)], dtype=torch.long)

        stage = "train" if self.training else "test"
        self.logger.info(f"{stage}_times: {self.start_time} ~ {self.end_time} = {len(self.inds)} samples, grid={tuple(self.grid_shape.tolist())}")

    def __len__(self):
        return len(self.inds)

    def _rand_another(self):
        return np.random.randint(len(self))

    def _load_static_clutter_mask(self, clutter_mask_path: str | Path | None) -> np.ndarray | None:
        if clutter_mask_path is None:
            return None
        mask_path = Path(clutter_mask_path).expanduser()
        mask = np.load(mask_path).astype(bool, copy=False)
        expected_shape = (self.header.height, self.header.width)
        if mask.shape != expected_shape:
            raise ValueError(f"Static clutter mask {mask_path} shape {mask.shape} does not match radar grid {expected_shape}")
        return _slice_spatial_array(mask, self.header, self.bbox, self.crop, self.stride)

    def __getitem__(self, idx):
        while True:
            data = self.prepare_data(idx)
            if self.training and data is None:
                idx = self._rand_another()
                continue
            return data

    def _read_frame(self, record_index: int) -> np.ndarray:
        record = self.records[record_index]
        return read_radar_reflectivity(
            record.path,
            header=record.header,
            scale_factor=self.scale_factor,
            fill_value=self.fill_value,
            bbox=self.bbox,
            crop=self.crop,
            stride=self.stride,
            qc=self.qc_config,
            static_clutter_mask=self.static_clutter_mask,
        )

    def prepare_data(self, i):
        idx = int(self.inds[i])
        inds = np.arange(idx, idx + self.total_frames)
        init_time = pd.to_datetime(self.id_to_time[idx])
        time_tensor = torch.tensor(init_time.value)

        lft_inds = inds[: self.hist_frames]
        rht_inds = inds[self.hist_frames :]

        if self.training:
            rht_inds = rht_inds[self.start_frames :]
        elif self.inference_only:
            rht_inds = np.array([], dtype=int)
        else:
            rht_inds = rht_inds[self.fcst_frames - 1]

        frame_inds = np.concatenate([lft_inds, rht_inds])
        imgs = np.stack([self._read_frame(int(frame_idx)) for frame_idx in frame_inds], axis=0)
        imgs = torch.from_numpy(imgs[:, None, ...])
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.lat_flip:
            imgs = torch.flip(imgs, dims=[2])
        crop_origin = None
        if self.random_crop_size is not None:
            imgs, crop_origin = self._crop_frames(imgs)

        inputs = imgs[: self.hist_frames]
        if self.inference_only:
            data = dict(times=time_tensor, inputs=inputs, grid_shape=self.grid_shape)
            if crop_origin is not None:
                data["crop_origin"] = crop_origin
            return data

        targets = imgs[self.hist_frames :]
        data = dict(times=time_tensor, inputs=inputs, targets=targets, grid_shape=self.grid_shape)
        if crop_origin is not None:
            data["crop_origin"] = crop_origin
        return data

    def _crop_frames(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        crop_h, crop_w = self.random_crop_size or (0, 0)
        height, width = imgs.shape[-2:]
        if crop_h > height or crop_w > width:
            raise ValueError(f"random_crop_size {(crop_h, crop_w)} exceeds frame shape {(height, width)}")

        max_row = height - crop_h
        max_col = width - crop_w
        if not self.training:
            row_start = max_row // 2
            col_start = max_col // 2
        elif torch.rand(()) < self.echo_crop_prob and torch.any(imgs[:, 0] >= self.echo_threshold_dbz):
            echo_pixels = torch.nonzero(torch.amax(imgs[:, 0], dim=0) >= self.echo_threshold_dbz, as_tuple=False)
            selected = echo_pixels[torch.randint(echo_pixels.shape[0], (1,)).item()]
            row_offset = torch.randint(crop_h, (1,)).item()
            col_offset = torch.randint(crop_w, (1,)).item()
            row_start = min(max(int(selected[0].item()) - row_offset, 0), max_row)
            col_start = min(max(int(selected[1].item()) - col_offset, 0), max_col)
        else:
            row_start = torch.randint(max_row + 1, (1,)).item() if max_row > 0 else 0
            col_start = torch.randint(max_col + 1, (1,)).item() if max_col > 0 else 0

        cropped = imgs[..., row_start : row_start + crop_h, col_start : col_start + crop_w]
        origin = torch.tensor([row_start, col_start], dtype=torch.long)
        return cropped.contiguous(), origin
