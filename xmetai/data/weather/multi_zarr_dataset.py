import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from xmetai.registries import DATASETS

from .data_util import make_seq, make_seq_inference

__all__ = ["MultiZarrDataset"]


def _select_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float] | None) -> xr.Dataset:
    if bbox is None:
        return ds
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (lon_min, lat_min, lon_max, lat_max), got {bbox}")

    lon_min, lat_min, lon_max, lat_max = [float(v) for v in bbox]
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError(f"Invalid bbox bounds: {bbox}")

    lat_slice = slice(lat_min, lat_max) if ds.lat.values[0] <= ds.lat.values[-1] else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if ds.lon.values[0] <= ds.lon.values[-1] else slice(lon_max, lon_min)
    ds = ds.sel(lat=lat_slice, lon=lon_slice)
    if ds.sizes["lat"] == 0 or ds.sizes["lon"] == 0:
        raise ValueError(f"bbox {bbox} does not overlap zarr grid")
    return ds


@DATASETS.register()
class MultiZarrDataset(Dataset):
    def __init__(
        self,
        data_paths: list,  # List[str]
        hist_frames: int,
        fcst_frames: int | list[int],
        start_frames: int = 0,
        freq: int = 6,
        interval: int = 1,
        training: bool = True,
        inference_only: bool = False,
        years: tuple = ("19790101", "20161231"),
        in_names: list[str] = [],  # list of variable names to use, if empty, use all variables in the dataset
        lat_flip: bool = False,  # whether to flip latitude axis
        freq_unit: str = "hours",
        bbox: tuple[float, float, float, float] | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading data from: {data_paths}")
        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.array(fcst_frames)
        self.max_train_fcst_frames = int(np.max(np.atleast_1d(self.fcst_frames)))
        self.runtime_train_fcst_frames = mp.Value("i", self.max_train_fcst_frames) if training and not inference_only else None
        self.training = training
        self.inference_only = inference_only
        self.lat_flip = lat_flip
        self.freq_unit = freq_unit

        # 1. 加载并筛选时间范围
        datasets = []
        times_list: list[pd.DatetimeIndex] = []

        for path in data_paths:
            ds = xr.open_zarr(path, consolidated=True)  # single dataset
            self.logger.info(f"Loaded dataset from {path}, shape: {ds.data.shape} (time, channel, lat, lon)")
            ds = ds.sel(time=slice(*years))
            ds = _select_bbox(ds, bbox)
            datasets.append(ds)
            times_list.append(pd.DatetimeIndex(ds.time.values))

        # 2. 取时间交集
        common_times = times_list[0]
        for t in times_list[1:]:
            common_times = common_times.intersection(t)

        # 3. 只取 freq 对齐的时间点（如每6小时）
        if freq_unit == "hours":
            common_times = common_times[common_times.hour % freq == 0]
        else:
            freq_step = pd.Timedelta(**{freq_unit: freq})
            if freq_step <= pd.Timedelta(0):
                raise ValueError(f"freq must define a positive time step, got {freq_step}")
            time_offsets = common_times - common_times[0]
            common_times = common_times[time_offsets % freq_step == pd.Timedelta(0)]

        if len(common_times) == 0:
            raise ValueError("No overlapping time steps among zarr files with desired freq.")

        # 4. 对所有 ds 按时间交集切片。常见的单 zarr 连续时间轴无需回选，避免构造超大的高级索引。
        aligned_datasets = []
        for ds, times in zip(datasets, times_list):
            if common_times.equals(times):
                aligned_datasets.append(ds)
                continue
            positions = times.get_indexer(common_times)
            if np.any(positions < 0):
                raise ValueError("Could not align zarr time coordinates to common_times.")
            aligned_datasets.append(ds.isel(time=positions))
        datasets = aligned_datasets

        # 5. 在 channel 维度上拼接
        data_arrays = [ds.data.rename(level="channel") if "level" in ds.data.dims else ds.data for ds in datasets]
        self.ds = data_arrays[0] if len(data_arrays) == 1 else xr.concat(data_arrays, dim="channel")
        self.logger.info(f"Combined dataset shape: {self.ds.shape}, sizes: {dict(self.ds.sizes)}")

        # 6. 构造时间索引
        self.fcst_frames += self.start_frames
        self.total_frames = self.hist_frames + (0 if inference_only else self.fcst_frames if self.training else self.fcst_frames.max())

        # 用第一个 ds 来生成索引
        if interval < 0:
            self.inds = make_seq_inference(common_times, self.total_frames, freq=freq, freq_unit=freq_unit)
        else:
            self.inds = make_seq(common_times, self.total_frames, freq=freq, interval=interval, freq_unit=freq_unit)
        if len(self.inds) == 0:
            raise ValueError(f"No regular zarr sequences found with total_frames={self.total_frames}, freq={freq} {freq_unit}")

        self.id_to_time = {i: common_times[i + self.hist_frames - 1] for i in self.inds}

        self.start_time = self.id_to_time[self.inds[0]]
        self.end_time = self.id_to_time[self.inds[-1]]

        stage = "train" if self.training else "test"
        self.logger.info(f"{stage}_times: {self.start_time} ~ {self.end_time} = {len(self.inds)} samples")
        self.select_vars: list[int] | None = None
        if len(in_names) > 0:
            channel_values = self.ds.channel.values.tolist()
            self.select_vars = [channel_values.index(v) for v in in_names if v in channel_values]
            missing = [v for v in in_names if v not in channel_values]
            if len(missing) > 0:
                self.logger.warning(f"Variables not found in 'channel': {missing}")
            self.logger.info(f"Selected variables: {self.select_vars}, total {len(self.select_vars)} variables.")

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
        total_frames = self.total_frames
        if self.training:
            train_fcst_frames = self.max_train_fcst_frames if self.runtime_train_fcst_frames is None else int(self.runtime_train_fcst_frames.value)
            total_frames = self.hist_frames + self.start_frames + train_fcst_frames
        inds = np.arange(idx, idx + total_frames)
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
        imgs_np = self.ds.isel(time=inds).values.astype(np.float32, copy=False)
        imgs = torch.from_numpy(imgs_np)
        imgs.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if self.select_vars is not None:
            imgs = imgs[:, self.select_vars, ...]
        if self.lat_flip:  # flip latitude axis
            imgs = torch.flip(imgs, dims=[2])
        inputs = imgs[: self.hist_frames]
        if self.inference_only:
            return dict(times=time_tensor, inputs=inputs)

        targets = imgs[self.hist_frames :]
        return dict(times=time_tensor, inputs=inputs, targets=targets)
