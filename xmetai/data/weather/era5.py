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
from torch.utils.data import Dataset

from xmetai.builder import DATASETS

from .data_util import make_seq, make_seq_inference

__all__ = ["ERA5"]


@DATASETS.register()
class ERA5(Dataset):
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
        in_names: list[str] = [],  # list of variable names to use, if empty, use all variables in the dataset
    ):
        self.logger = logging.getLogger(__name__)
        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.array(fcst_frames)
        self.training = training
        self.inference_only = inference_only

        ds = xr.open_zarr(data_name, consolidated=True)
        ds = ds.sel(time=slice(*years))

        self.fcst_frames += self.start_frames
        self.total_frames = self.hist_frames + (0 if inference_only else self.fcst_frames if self.training else self.fcst_frames.max())

        hit_time = [t for t in pd.to_datetime(ds.time) if t.hour % freq == 0]
        self.ds = ds.sel(time=hit_time)
        self.logger.info(f"Dataset loaded with {len(self.ds.time)} time steps.")

        times = self.ds.time.values
        if interval < 0:
            self.inds = make_seq_inference(times, self.total_frames, freq=freq)
        else:
            self.inds = make_seq(times, self.total_frames, freq=freq, interval=interval)
        self.logger.debug(f"inds: {self.inds}")
        self.id_to_time = {i: times[i + self.hist_frames - 1] for i in self.inds}  # id maps to the time of the last frame in the sequence

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
        # convert time to tensor
        time_tensor = torch.tensor(init_time.value)

        lft_inds = inds[: self.hist_frames]
        rht_inds = inds[self.hist_frames :]

        if self.training:
            rht_inds = rht_inds[self.start_frames :]
        elif self.inference_only:
            rht_inds = np.array([], dtype=int)
        else:
            rht_inds = rht_inds[self.fcst_frames - 1]

        inds = np.concatenate([lft_inds, rht_inds])
        imgs = torch.from_numpy(self.ds.isel(time=inds).data.values)
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.select_vars is not None:
            imgs = imgs[:, self.select_vars, ...]
        inputs = imgs[: self.hist_frames]
        if self.inference_only:
            return dict(times=time_tensor, inputs=inputs)

        targets = imgs[self.hist_frames :]
        return dict(times=time_tensor, inputs=inputs, targets=targets)
