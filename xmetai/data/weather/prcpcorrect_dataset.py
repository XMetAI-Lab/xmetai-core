"""FENGQ15 and CMPAS dataset for six-hour precipitation correction."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset

from xmetai.data.loader import build_train_loader
from xmetai.data.sampler import RepeatFactorTrainingSampler, TrainingSampler
from xmetai.registries import DATASETS

__all__ = ["PrcpCorrectionUNet3DDataset", "build_prcp_train_loader"]

ScaleValue = Union[str, os.PathLike, np.ndarray, torch.Tensor]
LOGGER = logging.getLogger(__name__)

_FENGQ_SCHEMA = "xmetai.fengq15.cropped_v2"
_FENGQ_SHAPE = (12, 30, 64, 40)
_CMPAS_FIELD_SHAPE = (1, 64, 40)
_FENGQ_DIMENSIONS = ["time", "dtime", "pl", "lat", "lon"]
_CMPAS_DIMENSIONS = ["valid_time", "channel", "lat", "lon"]
_DTIMES = np.arange(6, 73, 6, dtype=np.int16)
_LATITUDES = (24.25 + 0.25 * np.arange(64)).astype(np.float32)
_LONGITUDES = (110.0 + 0.25 * np.arange(40)).astype(np.float32)
_FENGQ_CHANNELS = (
    "tp",
    "t2m",
    "10u",
    "10v",
    "msl",
    *(f"{name}{level}" for name in ("z", "t", "u", "v", "q") for level in (925, 850, 700, 500, 200)),
)


def _as_beijing_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp


def _datetime_keys(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype("datetime64[ns]").astype(np.int64)


def _load_scale(value: ScaleValue, name: str) -> np.ndarray:
    if isinstance(value, (str, os.PathLike)):
        value = np.load(os.fspath(value))
    elif torch.is_tensor(value):
        value = cast(torch.Tensor, value).detach().cpu().numpy()
    scale = np.asarray(value, dtype=np.float32)
    expected_shape = (_FENGQ_SHAPE[1], 1, 1)
    if scale.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {scale.shape}.")
    if not np.isfinite(scale).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return scale


@DATASETS.register()
class PrcpCorrectionUNet3DDataset(Dataset):
    """Read FENGQ15 forecasts, with optional six-hour CMPAS targets."""

    def __init__(
        self,
        fengq_zarr_path: str,
        cmpas_zarr_path: Optional[str],
        input_min: ScaleValue,
        input_max: ScaleValue,
        split_csv: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        super().__init__()
        self.input_min = _load_scale(input_min, "input_min")
        input_max = _load_scale(input_max, "input_max")
        if np.any(input_max <= self.input_min):
            invalid = np.flatnonzero((input_max <= self.input_min).reshape(-1)).tolist()
            raise ValueError(f"input_max must exceed input_min for every channel; invalid channels: {invalid}")
        self.input_range = input_max - self.input_min

        self.fengq_group = zarr.open_group(fengq_zarr_path, mode="r")
        self.cmpas_group = zarr.open_group(cmpas_zarr_path, mode="r") if cmpas_zarr_path is not None else None

        self.init_times, valid_times, latitude, longitude = self._validate_fengq()
        self.sample_indices = self._resolve_samples(self.init_times, split_csv, start_time, end_time)
        if not self.sample_indices:
            raise ValueError("PrcpCorrectionUNet3DDataset selected no samples.")
        self.target_indices = self._validate_cmpas(valid_times[self.sample_indices], latitude, longitude)

        if self.cmpas_group is None:
            LOGGER.info("Loaded %d FENGQ15 inference samples from %s without targets", len(self.sample_indices), fengq_zarr_path)
        else:
            LOGGER.info(
                "Loaded %d FENGQ15 samples from %s with CMPAS targets from %s",
                len(self.sample_indices),
                fengq_zarr_path,
                cmpas_zarr_path,
            )

    @staticmethod
    def _validate_array_dimensions(array: zarr.Array, expected: list[str], name: str) -> None:
        actual = list(cast(list[str], array.attrs.get("_ARRAY_DIMENSIONS", [])))
        if actual != expected:
            raise ValueError(f"{name} dimensions must be {expected}, got {actual}.")

    def _validate_fengq(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.fengq_group.attrs.get("schema") != _FENGQ_SCHEMA:
            raise ValueError(f"FENGQ15 Zarr schema must be {_FENGQ_SCHEMA}.")
        if self.fengq_group.attrs.get("normalization") != "none":
            raise ValueError("FENGQ15 Zarr must contain unnormalized physical values.")
        if self.fengq_group.attrs.get("time_zone") != "Beijing time":
            raise ValueError("FENGQ15 time coordinates must use Beijing time.")

        required = {"input", "time", "dtime", "valid_time", "pl", "lat", "lon"}
        if missing := required - set(self.fengq_group.array_keys()):
            raise ValueError(f"FENGQ15 Zarr is missing arrays: {sorted(missing)}")

        self.fengq_data = cast(zarr.Array, self.fengq_group["input"])
        self._validate_array_dimensions(self.fengq_data, _FENGQ_DIMENSIONS, "FENGQ15 input")
        if self.fengq_data.shape[1:] != _FENGQ_SHAPE:
            raise ValueError(f"FENGQ15 input shape must be (N,{_FENGQ_SHAPE}), got {self.fengq_data.shape}.")
        if tuple(np.asarray(cast(zarr.Array, self.fengq_group["pl"])[:]).tolist()) != _FENGQ_CHANNELS:
            raise ValueError("FENGQ15 channel names or order do not match the dataset schema.")
        if not np.array_equal(np.asarray(cast(zarr.Array, self.fengq_group["dtime"])[:]), _DTIMES):
            raise ValueError("FENGQ15 dtime must be 6-72 hours at six-hour intervals.")

        latitude = np.asarray(cast(zarr.Array, self.fengq_group["lat"])[:])
        longitude = np.asarray(cast(zarr.Array, self.fengq_group["lon"])[:])
        if not np.allclose(latitude, _LATITUDES) or not np.allclose(longitude, _LONGITUDES):
            raise ValueError("FENGQ15 grid must be 24.25-40.00 N and 110.00-119.75 E at 0.25-degree resolution.")

        init_times = np.asarray(cast(zarr.Array, self.fengq_group["time"])[:]).astype("datetime64[ns]")
        valid_times = np.asarray(cast(zarr.Array, self.fengq_group["valid_time"])[:]).astype("datetime64[ns]")
        if init_times.shape != (self.fengq_data.shape[0],):
            raise ValueError("FENGQ15 time coordinate does not match the input sample count.")
        if valid_times.shape != (self.fengq_data.shape[0], len(_DTIMES)):
            raise ValueError("FENGQ15 valid_time must have shape (N,12).")

        init_keys = _datetime_keys(init_times)
        if len(np.unique(init_keys)) != len(init_keys) or np.any(np.diff(init_keys) <= 0):
            raise ValueError("FENGQ15 time must be unique and strictly increasing.")
        if not np.array_equal(valid_times, init_times[:, None] + _DTIMES.astype("timedelta64[h]")):
            raise ValueError("FENGQ15 valid_time must equal time plus dtime.")
        return init_times, valid_times, latitude, longitude

    def _validate_cmpas(self, fengq_valid_times: np.ndarray, fengq_lat: np.ndarray, fengq_lon: np.ndarray) -> Optional[np.ndarray]:
        if self.cmpas_group is None:
            self.cmpas_data = None
            return None

        if self.cmpas_group.attrs.get("time_zone") != "Beijing time":
            raise ValueError("CMPAS time coordinates must use Beijing time.")
        required = {"data", "valid_mask", "valid_time", "channel", "lat", "lon"}
        if missing := required - set(self.cmpas_group.array_keys()):
            raise ValueError(f"CMPAS Zarr is missing arrays: {sorted(missing)}")

        self.cmpas_data = cast(zarr.Array, self.cmpas_group["data"])
        self.cmpas_mask = cast(zarr.Array, self.cmpas_group["valid_mask"])
        self._validate_array_dimensions(self.cmpas_data, _CMPAS_DIMENSIONS, "CMPAS data")
        self._validate_array_dimensions(self.cmpas_mask, _CMPAS_DIMENSIONS, "CMPAS valid_mask")
        if self.cmpas_mask.shape != self.cmpas_data.shape or self.cmpas_mask.dtype != np.bool_:
            raise ValueError("CMPAS valid_mask must be boolean with the same shape as data.")
        if self.cmpas_data.shape[1:] != _CMPAS_FIELD_SHAPE:
            raise ValueError(f"CMPAS data shape must be (M,{_CMPAS_FIELD_SHAPE}), got {self.cmpas_data.shape}.")
        if np.asarray(cast(zarr.Array, self.cmpas_group["channel"])[:]).tolist() != ["tp"]:
            raise ValueError("CMPAS must contain exactly one channel named tp.")
        if not np.array_equal(fengq_lat, np.asarray(cast(zarr.Array, self.cmpas_group["lat"])[:])) or not np.array_equal(
            fengq_lon, np.asarray(cast(zarr.Array, self.cmpas_group["lon"])[:])
        ):
            raise ValueError("FENGQ15 and CMPAS grids do not match.")

        cmpas_times = np.asarray(cast(zarr.Array, self.cmpas_group["valid_time"])[:]).astype("datetime64[ns]")
        if cmpas_times.shape != (self.cmpas_data.shape[0],):
            raise ValueError("CMPAS valid_time coordinate does not match the data length.")
        cmpas_keys = _datetime_keys(cmpas_times)
        if len(np.unique(cmpas_keys)) != len(cmpas_keys) or np.any(np.diff(cmpas_keys) <= 0):
            raise ValueError("CMPAS valid_time must be unique and strictly increasing.")
        if not set(pd.DatetimeIndex(cmpas_times).hour).issubset({2, 8, 14, 20}):
            raise ValueError("CMPAS valid_time must end at 02, 08, 14, or 20 Beijing time.")

        lookup = {int(key): index for index, key in enumerate(cmpas_keys)}
        try:
            indices = np.asarray(
                [[lookup[int(key)] for key in row] for row in _datetime_keys(fengq_valid_times)],
                dtype=np.int64,
            )
        except KeyError as error:
            missing_time = pd.Timestamp(int(error.args[0]), unit="ns")
            raise ValueError(f"CMPAS is missing target time {missing_time}.") from None
        return indices

    @staticmethod
    def _resolve_samples(
        init_times: np.ndarray,
        split_csv: Optional[str],
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> list[int]:
        init_index = {_as_beijing_timestamp(time): index for index, time in enumerate(init_times)}
        if split_csv is not None:
            if start_time is not None or end_time is not None:
                raise ValueError("split_csv cannot be combined with start_time or end_time.")
            split = pd.read_csv(split_csv, header=None)
            if split.shape[1] == 0:
                raise ValueError("Split CSV contains no initialization times.")
            split_times = split.iloc[:, 0].map(_as_beijing_timestamp)
            if split_times.isnull().any() or split_times.duplicated().any():
                raise ValueError("Split CSV times must be non-null and unique.")
            missing = [time for time in split_times if time not in init_index]
            if missing:
                preview = ", ".join(str(time) for time in missing[:3])
                raise ValueError(f"{len(missing)} split times are missing from FENGQ15 Zarr, e.g. {preview}")
            return [init_index[time] for time in split_times]

        start = _as_beijing_timestamp(start_time) if start_time is not None else None
        end = _as_beijing_timestamp(end_time) if end_time is not None else None
        if start is not None and end is not None and end < start:
            raise ValueError("end_time must not be earlier than start_time.")
        return [index for index, time in enumerate(pd.DatetimeIndex(init_times)) if (start is None or time >= start) and (end is None or time <= end)]

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_index = self.sample_indices[idx]
        init_time = self.init_times[sample_index]
        inputs = np.asarray(self.fengq_data[sample_index], dtype=np.float32)
        if not np.isfinite(inputs).all():
            raise ValueError(f"Sample {sample_index} ({init_time}) contains NaN or Inf in FENGQ15 inputs.")
        if np.any(inputs[:, 0] < 0.0):
            raise ValueError(f"Sample {sample_index} ({init_time}) contains negative FENGQ15 tp.")

        inputs = (inputs - self.input_min) / self.input_range
        sample = {
            "times": torch.tensor(init_time.astype(np.int64), dtype=torch.int64),
            "inputs": torch.from_numpy(np.ascontiguousarray(inputs)),
        }
        if self.target_indices is not None:
            assert self.cmpas_data is not None
            targets = np.stack([np.asarray(self.cmpas_data[index], dtype=np.float32) for index in self.target_indices[idx]])
            valid = np.stack([np.asarray(self.cmpas_mask[index]) for index in self.target_indices[idx]])
            if not np.isfinite(targets).all():
                raise ValueError(f"Sample {sample_index} ({init_time}) contains NaN or Inf in CMPAS targets.")
            if np.any(targets < 0.0):
                raise ValueError(f"Sample {sample_index} ({init_time}) contains negative CMPAS precipitation.")
            if np.any(targets[~valid] != 0) or not valid.any(axis=(-2, -1)).all():
                raise ValueError(f"Sample {sample_index} ({init_time}) has invalid mask placeholders or a lead without valid cells.")
            sample["targets"] = torch.from_numpy(np.ascontiguousarray(targets))
            sample["valid_mask"] = torch.from_numpy(np.ascontiguousarray(valid))
        return sample


def _load_repeat_factors(dataset: PrcpCorrectionUNet3DDataset, event_manifest: str | os.PathLike) -> torch.Tensor:
    """Match manifest factors to selected initialization times, not Zarr indices."""
    with open(event_manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "xmetai.prcp_correction.event_manifest_v2":
        raise ValueError("RFS requires an event_manifest_v2 manifest.")
    rows = manifest["samples"]
    if manifest["sample_count"] != len(rows):
        raise ValueError("RFS manifest sample_count does not match its sample rows.")
    times = [_as_beijing_timestamp(row["init_time"]) for row in rows]
    if any(pd.isna(time) for time in times) or len(set(times)) != len(times):
        raise ValueError("RFS manifest initialization times must be non-null and unique.")
    selected = [_as_beijing_timestamp(dataset.init_times[index]) for index in dataset.sample_indices]
    missing = set(selected) - set(times)
    extra = set(times) - set(selected)
    if missing or extra:
        raise ValueError(
            f"RFS manifest must exactly match the selected training samples: {len(missing)} missing, {len(extra)} extra. "
            "Rebuild the manifest with the current training selection."
        )
    by_time = dict(zip(times, rows))
    factors = np.asarray([by_time[time]["repeat_factor"] for time in selected], dtype=np.float32)
    if factors.shape != (len(dataset),) or not np.isfinite(factors).all() or np.any(factors < 1.0):
        raise ValueError("RFS repeat factors must be a finite vector with one value >= 1 per training sample.")
    LOGGER.info("RFS manifest: %s; score strata: %s", os.fspath(event_manifest), manifest.get("sampling", {}))
    return torch.from_numpy(factors)


def build_prcp_train_loader(
    dataset: PrcpCorrectionUNet3DDataset,
    *,
    total_batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int | None = None,
    event_manifest: str | os.PathLike | None = None,
) -> torch.utils.data.DataLoader:
    """Use native repeat-factor sampling when a training manifest is supplied."""
    if event_manifest is None:
        LOGGER.info("Training sampling: native TrainingSampler, samples=%d, seed=%s", len(dataset), seed)
        sampler = TrainingSampler(len(dataset), seed=seed)
    else:
        factors = _load_repeat_factors(dataset, event_manifest)
        LOGGER.info(
            "Training sampling: native RepeatFactorTrainingSampler, samples=%d, repeat_factor min/mean/max=%.4f/%.4f/%.4f, seed=%s",
            len(dataset),
            factors.min().item(),
            factors.mean().item(),
            factors.max().item(),
            seed,
        )
        sampler = RepeatFactorTrainingSampler(factors, seed=seed)
    return build_train_loader(
        dataset,
        sampler=sampler,
        total_batch_size=total_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        do_augment=False,
    )
