"""GraphCast-style dataset wrapper for weather zarr data."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from xmetai.builder import DATASETS

from .data_util import make_seq, make_seq_inference
from .graphcast_grid import build_edge_attr, build_grid_edges, build_node_static

__all__ = ["GraphCastDataset"]


@DATASETS.register()
class GraphCastDataset(Dataset):
    def __init__(
        self,
        data_name: Optional[str] = None,
        data_paths: Optional[Iterable[str]] = None,
        hist_frames: int = 2,
        fcst_frames: int = 1,
        start_frames: int = 0,
        freq: int = 6,
        interval: int = 1,
        training: bool = True,
        inference_only: bool = False,
        years: tuple = ("19790101", "20161231"),
        in_names: Optional[list] = None,
        lat_flip: bool = False,
        neighbors: int = 8,
        periodic_lon: bool = True,
        edge_trig: bool = False,
        static_fields: Optional[list] = None,
        static_levels: Optional[dict] = None,
    ):
        self.logger = logging.getLogger(__name__)
        if data_paths is None:
            if data_name is None:
                raise ValueError("data_name or data_paths must be provided")
            data_paths = [data_name]

        self.logger.info(f"Loading GraphCast data from: {data_paths}")
        self.start_frames = start_frames
        self.hist_frames = hist_frames
        self.fcst_frames = np.array(fcst_frames)
        self.training = training
        self.inference_only = inference_only
        self.lat_flip = lat_flip
        self.neighbors = neighbors
        self.periodic_lon = periodic_lon
        self.edge_trig = edge_trig
        self.static_fields = static_fields or []
        self.static_levels = static_levels or {}

        def _select_data_array(ds: xr.Dataset) -> xr.DataArray:
            if "data" in ds:
                return ds["data"]
            data_vars = list(ds.data_vars)
            if len(data_vars) == 1:
                return ds[data_vars[0]]
            raise ValueError(f"GraphCastDataset expects a single data variable or 'data'. Found: {data_vars}")

        datasets = []
        times_list = []
        for path in data_paths:
            ds = xr.open_zarr(path, consolidated=True)
            data_da = _select_data_array(ds)
            self.logger.info(f"Loaded dataset from {path}, shape: {data_da.shape} (time, level, lat, lon)")
            ds = ds.sel(time=slice(*years))
            datasets.append(ds)
            times_list.append(pd.to_datetime(ds.time.values))

        common_times = times_list[0]
        for t in times_list[1:]:
            common_times = np.intersect1d(common_times, t)

        common_times = pd.to_datetime(common_times)
        common_times = sorted([t for t in common_times if t.hour % freq == 0])
        if len(common_times) == 0:
            raise ValueError("No overlapping time steps among zarr files with desired freq.")

        datasets = [ds.sel(time=common_times) for ds in datasets]
        self.datasets = datasets
        data_arrays = [_select_data_array(ds) for ds in datasets]
        self.ds = xr.concat(data_arrays, dim="level")
        self.logger.info(f"Combined dataset shape: {self.ds.shape} with coords: {self.ds.coords}")

        self.fcst_frames += self.start_frames
        self.total_frames = self.hist_frames + (0 if inference_only else self.fcst_frames if self.training else self.fcst_frames.max())

        if interval < 0:
            self.inds = make_seq_inference(common_times, self.total_frames, freq=freq)
        else:
            self.inds = make_seq(common_times, self.total_frames, freq=freq, interval=interval)

        self.id_to_time = {i: common_times[i + self.hist_frames - 1] for i in self.inds}
        self.start_time = self.id_to_time[self.inds[0]]
        self.end_time = self.id_to_time[self.inds[-1]]

        stage = "train" if self.training else "test"
        self.logger.info(f"{stage}_times: {self.start_time} ~ {self.end_time} = {len(self.inds)} samples")

        in_names = in_names or []
        if len(in_names) > 0 and "level" in self.ds.coords:
            level_values = self.ds.coords["level"].values.tolist()
            self.select_vars: list | None = [level_values.index(v) for v in in_names if v in level_values]
            missing = [v for v in in_names if v not in level_values]
            if len(missing) > 0:
                self.logger.warning(f"Variables not found in 'level': {missing}")
            self.logger.info(f"Selected variables: {self.select_vars}, total {len(self.select_vars)} variables.")
        else:
            self.select_vars = None

        lat = np.asarray(self.ds.lat.values, dtype=np.float32)
        lon = np.asarray(self.ds.lon.values, dtype=np.float32)
        if self.lat_flip:
            lat = lat[::-1].copy()
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        height, width = len(lat_rad), len(lon_rad)
        self.grid_shape = torch.tensor([height, width], dtype=torch.long)
        self.edge_index = build_grid_edges(height, width, neighbors=self.neighbors, periodic_lon=self.periodic_lon)
        self.edge_attr = build_edge_attr(lat_rad, lon_rad, self.edge_index, include_trig=self.edge_trig)
        node_static = build_node_static(lat_rad, lon_rad)
        if len(self.static_fields) > 0:
            extra_feats = []
            for name in self.static_fields:
                da = None
                for ds in self.datasets:
                    if name in ds:
                        da = ds[name]
                        break
                if da is None:
                    self.logger.warning(f"Static field '{name}' not found in dataset.")
                    continue
                if "time" in da.dims:
                    da = da.isel(time=0)
                if "level" in da.dims:
                    level_idx = self.static_levels.get(name, 0)
                    da = da.isel(level=level_idx)
                if "lat" not in da.dims or "lon" not in da.dims:
                    self.logger.warning(f"Static field '{name}' missing lat/lon dims.")
                    continue
                arr = np.asarray(da.values, dtype=np.float32)
                if self.lat_flip:
                    arr = arr[::-1, ...]
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                extra_feats.append(arr.reshape(-1, 1))
            if len(extra_feats) > 0:
                extra_feats_arr = np.concatenate(extra_feats, axis=-1)
                node_static = torch.cat([node_static, torch.from_numpy(extra_feats_arr)], dim=-1)
        self.node_static = node_static

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
        imgs = torch.from_numpy(self.ds.isel(time=inds).values)
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)
        if self.select_vars is not None:
            imgs = imgs[:, self.select_vars, ...]
        if self.lat_flip:
            imgs = torch.flip(imgs, dims=[2])

        inputs = imgs[: self.hist_frames]
        if self.inference_only:
            return dict(
                times=time_tensor,
                inputs=inputs,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                node_static=self.node_static,
                grid_shape=self.grid_shape,
            )

        targets = imgs[self.hist_frames :]
        return dict(
            times=time_tensor,
            inputs=inputs,
            targets=targets,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_static=self.node_static,
            grid_shape=self.grid_shape,
        )
