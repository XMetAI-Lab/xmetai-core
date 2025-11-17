import logging
import numpy as np
import pandas as pd
import torch
import xarray as xr

from torch.utils.data import Dataset
from xmetai.builder import DATASETS
from .data_util import make_seq, make_seq_inference

__all__ = ["MultiZarrDataset"]


@DATASETS.register()
class MultiZarrDataset(Dataset):
    def __init__(
        self,
        data_paths,  # List[str]
        hist_frames,
        fcst_frames,
        start_frames: int = 0,
        freq: int = 6,
        interval: int = 1,
        training: bool = True,
        inference_only: bool = False,
        years: tuple = ('19790101', '20161231'),
        in_names: list = [],  # list of variable names to use, if empty, use all variables in the dataset
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading data from: {data_paths}")
        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.array(fcst_frames)
        self.training = training
        self.inference_only = inference_only

        # 1. 加载并筛选时间范围
        datasets = []
        times_list = []

        for path in data_paths:
            ds = xr.open_zarr(path, consolidated=True)  # single dataset
            self.logger.info(f"Loaded dataset from {path}, shape: {ds.data.shape} (time, level, lat, lon)")
            ds = ds.sel(time=slice(*years))
            datasets.append(ds)
            times_list.append(pd.to_datetime(ds.time.values))

        # 2. 取时间交集
        common_times = times_list[0]
        for t in times_list[1:]:
            common_times = np.intersect1d(common_times, t)

        # 3. 只取 freq 对齐的时间点（如每6小时）
        common_times = pd.to_datetime(common_times)
        common_times = sorted([t for t in common_times if t.hour % freq == 0])

        if len(common_times) == 0:
            raise ValueError("No overlapping time steps among zarr files with desired freq.")

        # 4. 对所有 ds 按时间交集切片
        datasets = [ds.sel(time=common_times) for ds in datasets]

        # 5. 在 level 维度上拼接
        self.ds = xr.concat([ds.data for ds in datasets], dim="level")
        self.logger.info(f"Combined dataset shape: {self.ds.shape} with coords: {self.ds.coords}")

        # 6. 构造时间索引
        self.fcst_frames += self.start_frames
        self.total_frames = self.hist_frames + (self.fcst_frames if self.training else max(self.fcst_frames))

        # 用第一个 ds 来生成索引
        if interval < 0:
            self.inds = make_seq_inference(common_times, self.total_frames, freq=freq)
        else:
            self.inds = make_seq(common_times, self.total_frames, freq=freq, interval=interval)

        self.id_to_time = {
            i: common_times[i + self.hist_frames - 1] for i in self.inds
        }

        self.start_time = self.id_to_time[self.inds[0]]
        self.end_time = self.id_to_time[self.inds[-1]]

        stage = "train" if self.training else "test"
        self.logger.info(f"{stage}_times: {self.start_time} ~ {self.end_time} = {len(self.inds)} samples")
        if len(in_names) > 0:
            level_values = ds.level.values.tolist()
            self.select_vars = [level_values.index(v) for v in in_names if v in level_values]
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

        lft_inds = inds[:self.hist_frames]
        rht_inds = inds[self.hist_frames:]

        if self.training:
            rht_inds = rht_inds[self.start_frames:]
        else:
            rht_inds = rht_inds[self.fcst_frames - 1]

        inds = np.concatenate([lft_inds, rht_inds])
        imgs = torch.from_numpy(self.ds.isel(time=inds).values)
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.select_vars is not None:
            imgs = imgs[:, self.select_vars, ...]
        inputs = imgs[: self.hist_frames]
        if self.inference_only:
            return dict(times=time_tensor, inputs=inputs)

        targets = imgs[self.hist_frames:]
        return dict(times=time_tensor, inputs=inputs, targets=targets)
