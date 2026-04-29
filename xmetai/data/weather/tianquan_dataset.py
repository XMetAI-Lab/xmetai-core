"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-08-18 20:06:08
Description: Read From Zarr format ERA5 dataset. Makesure dataset is already normalized.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import numpy as np
import pandas as pd
import torch
import xarray as xr
import torch.nn.functional as F
from torch.utils.data import Dataset

from xmetai.builder import DATASETS
from .data_util import make_seq, make_seq_inference

__all__ = ["TianQuanDataset"]


@DATASETS.register()
class TianQuanDataset(Dataset):
    def __init__(
            self,
            data_name: str,
            hist_frames: int,
            fcst_frames: int | list[int],
            start_frames: int = 0,
            freq: int = 6,
            interval: int = 1,
            training: bool = True,
            inference_only: bool = False,
            years: tuple = ("19790101", "20161231"),
            in_names: list[str] = [],
            # ================= 新增参数 =================
            img_size: tuple = (128, 256),       # 目标分辨率
            climate_zarr: str = None,           # 气候态数据的路径
            # ============================================
    ):
        self.logger = logging.getLogger(__name__)
        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.array(fcst_frames)
        self.training = training
        self.inference_only = inference_only
        self.img_size = tuple(int(i) for i in img_size)

        # 加载观测数据
        ds = xr.open_zarr(data_name, consolidated=True)
        ds = ds.sel(time=slice(*years))

        # ================= 加载气候态 =================
        assert climate_zarr is not None, "气候态数据集 climate_ds 不能为空！请传入有效的 climate_zarr 路径"
        self.climate_ds = xr.open_zarr(climate_zarr, consolidated=True)
        self.logger.info(f"Climate dataset loaded from {climate_zarr}")
        # ============================================

        self.fcst_frames += self.start_frames
        self.total_frames = self.hist_frames + (
            0 if inference_only else self.fcst_frames if self.training else self.fcst_frames.max())

        hit_time = [t for t in pd.to_datetime(ds.time) if t.hour % freq == 0]
        self.ds = ds.sel(time=hit_time)
        self.logger.info(f"Dataset loaded with {len(self.ds.time)} time steps.")

        times = self.ds.time.values
        if interval < 0:
            self.inds = make_seq_inference(times, self.total_frames, freq=freq)
        else:
            self.inds = make_seq(times, self.total_frames, freq=freq, interval=interval)

        self.id_to_time = {i: times[i + self.hist_frames - 1] for i in self.inds}

        self.start_time = self.id_to_time[self.inds[0]]
        self.end_time = self.id_to_time[self.inds[-1]]

        stage = "train" if self.training else "test"
        self.logger.info(f"{stage}_times: {self.start_time} ~ {self.end_time} = {len(self.inds)} with dataset length {len(self.inds)}")

        if len(in_names) > 0:
            level_values = ds.level.values.tolist()
            self.select_vars: list[int] | None = [level_values.index(v) for v in in_names if v in level_values]
            missing = [v for v in in_names if v not in level_values]
            if len(missing) > 0:
                self.logger.warning(f"Variables not found in 'level': {missing}")
            self.logger.info(f"Selected variables: {self.select_vars}, total {len(self.select_vars)} variables.")
        else:
            self.select_vars = None

    def __len__(self):
        return len(self.inds)

    def _rand_another(self):
        return np.random.randint(len(self))

    # 辅助函数：把当前时间映射到 1979 年以提取气候态
    def _replace_year_with_reference(self, idx, year=1979):
        replaced = []
        for ts in idx:
            try:
                new_ts = ts.replace(year=year)
            except ValueError:
                if ts.month == 2 and ts.day == 29:
                    new_ts = ts.replace(year=year, day=28)
                else:
                    raise
            replaced.append(new_ts)

        return replaced

    def __getitem__(self, idx):
        while True:
            data = self.prepare_data(idx)
            if self.training and data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_data(self, i):
        idx = self.inds[i]
        inds = np.arange(idx, idx + self.total_frames)
        init_time = pd.to_datetime(self.id_to_time[idx])
        time_tensor = torch.tensor(init_time.value)

        lft_inds = inds[: self.hist_frames]
        rht_inds = inds[self.hist_frames:]

        if self.training:
            rht_inds = rht_inds[self.start_frames:]
        elif self.inference_only:
            rht_inds = np.array([], dtype=int)
        else:
            rht_inds = rht_inds[self.fcst_frames - 1]

        inds = np.concatenate([lft_inds, rht_inds])
        imgs = torch.from_numpy(self.ds.isel(time=inds).data.values)
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)

        if self.select_vars is not None:
            imgs = imgs[:, self.select_vars, ...]

        # input resize
        if imgs.shape[-2:] != self.img_size:
            imgs = F.interpolate(
                imgs.float(), size=self.img_size, mode="bilinear", align_corners=False
            )
        inputs = imgs[: self.hist_frames]
        ret_dict = dict(times=time_tensor, inputs=inputs)

        # 二、加入气候态并 resize
        time_seq = pd.to_datetime(self.ds.time.values[inds])
        ref_times = [self._replace_year_with_reference(t) for t in time_seq]

        climate_imgs = torch.from_numpy(self.climate_ds.sel(time=ref_times).data.values)
        climate_imgs = torch.nan_to_num(climate_imgs, nan=0.0, posinf=0.0, neginf=0.0)
        climate_imgs = climate_imgs[:, self.select_vars, ...]

        # Resize 到目标分辨率
        if climate_imgs.shape[-2:] != self.img_size:
            climate_imgs = F.interpolate(
                climate_imgs.float(), size=self.img_size, mode="bilinear", align_corners=False
            )
        climate_seq = climate_imgs[self.hist_frames - 1:]


        ret_dict["climate_seq"] = climate_seq


        if self.inference_only:
            return ret_dict
        # resize targets
        targets = imgs[self.hist_frames:]
        ret_dict["targets"] = targets
        ret_dict["future_frames"] = targets.shape[0]
        return ret_dict