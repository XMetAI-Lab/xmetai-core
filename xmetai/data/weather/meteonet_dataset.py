"""PyTorch Dataset for MeteoNet radar reflectivity NPZ files."""

from __future__ import annotations

import logging
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib import format as npy_format
from torch.utils.data import Dataset

from xmetai.registries import DATASETS
from xmetai.utils.radar import _crop_to_slices, _make_sequence_indices, _time_range

__all__ = ["MeteoNetRadarDataset"]

logger = logging.getLogger(__name__)

_DEFAULT_SCALE_FACTORS = {
    "data": 10.0,
    "prob": 1.0,
    "height": 1.0,
}
_DEFAULT_MISSING_VALUES = {
    "data": (-200,),
    "prob": (-100,),
    "height": (-200,),
}


@dataclass(frozen=True)
class MeteoNetRadarRecord:
    """One timestamp inside a MeteoNet NPZ chunk."""

    time: pd.Timestamp
    path: Path
    index: int


def _as_paths(data_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]
    return [Path(path).expanduser() for path in data_paths]


def _as_channels(channels: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(channels, str):
        channels = (channels,)
    normalized = tuple(str(channel) for channel in channels)
    if len(normalized) == 0:
        raise ValueError("channels must contain at least one NPZ array name")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"channels contains duplicates: {normalized}")
    return normalized


def _read_npy_header(archive: zipfile.ZipFile, member: str) -> tuple[tuple[int, ...], bool, np.dtype]:
    with archive.open(member) as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            return npy_format.read_array_header_1_0(handle)
        if version == (2, 0):
            return npy_format.read_array_header_2_0(handle)
        return npy_format._read_array_header(handle, version)  # noqa: SLF001


def _read_npz_metadata(path: Path, keys: Sequence[str]) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    metadata: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        for key in keys:
            member = f"{key}.npy"
            if member not in members:
                raise KeyError(f"NPZ file {path} does not contain array {key!r}")
            shape, _, dtype = _read_npy_header(archive, member)
            metadata[key] = (tuple(int(dim) for dim in shape), np.dtype(dtype))
    return metadata


def _load_npz_dates(path: Path) -> tuple[pd.DatetimeIndex, set[pd.Timestamp]]:
    warning_class = getattr(getattr(np, "exceptions", object), "VisibleDeprecationWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warning_class)
        with np.load(path, allow_pickle=True) as npz_file:
            if "dates" not in npz_file.files:
                raise KeyError(f"NPZ file {path} does not contain required array 'dates'")
            dates = pd.to_datetime(npz_file["dates"])
            if "miss_dates" in npz_file.files:
                missing_dates = pd.to_datetime(npz_file["miss_dates"])
            else:
                missing_dates = pd.DatetimeIndex([])

    if dates.ndim != 1:
        raise ValueError(f"NPZ file {path} dates must be 1D, got shape {dates.shape}")
    if dates.hasnans:
        raise ValueError(f"NPZ file {path} contains invalid dates")
    return pd.DatetimeIndex(dates), {pd.Timestamp(value) for value in missing_dates if not pd.isna(value)}


def _discover_meteonet_records(
    data_paths: str | Path | Iterable[str | Path],
    *,
    pattern: str,
    channels: tuple[str, ...],
    years: Sequence[object] | None,
    exclude_missing_dates: bool,
) -> tuple[list[MeteoNetRadarRecord], tuple[int, int]]:
    files: list[Path] = []
    for path in _as_paths(data_paths):
        if path.is_dir():
            files.extend(sorted(item for item in path.rglob(pattern) if item.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"MeteoNet radar path does not exist: {path}")
    files = sorted(set(files))
    if len(files) == 0:
        raise ValueError(f"No MeteoNet radar NPZ files found under {data_paths} with pattern {pattern!r}")

    start_time, end_time = _time_range(years)
    records: list[MeteoNetRadarRecord] = []
    seen: dict[pd.Timestamp, Path] = {}
    grid_shape: tuple[int, int] | None = None

    for path in files:
        metadata = _read_npz_metadata(path, ("dates", *channels))
        dates, missing_dates = _load_npz_dates(path)
        if metadata["dates"][0] != (len(dates),):
            raise ValueError(f"NPZ file {path} dates metadata shape {metadata['dates'][0]} does not match loaded dates length {len(dates)}")

        for channel in channels:
            shape, _ = metadata[channel]
            if len(shape) != 3:
                raise ValueError(f"NPZ file {path} array {channel!r} must have shape (time, height, width), got {shape}")
            if shape[0] != len(dates):
                raise ValueError(f"NPZ file {path} array {channel!r} time dimension {shape[0]} does not match dates length {len(dates)}")
            channel_grid_shape = (shape[1], shape[2])
            if grid_shape is None:
                grid_shape = channel_grid_shape
            elif channel_grid_shape != grid_shape:
                raise ValueError(f"NPZ file {path} array {channel!r} grid {channel_grid_shape} does not match expected grid {grid_shape}")

        for local_index, timestamp in enumerate(dates):
            timestamp = pd.Timestamp(timestamp)
            if exclude_missing_dates and timestamp in missing_dates:
                continue
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            if timestamp in seen:
                raise ValueError(f"Duplicate MeteoNet radar timestamp {timestamp}: {seen[timestamp]} and {path}")
            seen[timestamp] = path
            records.append(MeteoNetRadarRecord(time=timestamp, path=path, index=local_index))

    records.sort(key=lambda record: record.time)
    if grid_shape is None:
        raise ValueError(f"Could not infer MeteoNet radar grid shape from {data_paths}")
    if len(records) == 0:
        raise ValueError(f"No MeteoNet radar frames remain after time filtering with years={years}")
    return records, grid_shape


def _normalize_crop_size(random_crop_size: Sequence[int] | int | None) -> tuple[int, int] | None:
    if random_crop_size is None:
        return None
    if isinstance(random_crop_size, int):
        crop_h = crop_w = int(random_crop_size)
    else:
        if len(random_crop_size) != 2:
            raise ValueError(f"random_crop_size must be an int or (height, width), got {random_crop_size}")
        crop_h, crop_w = [int(value) for value in random_crop_size]
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"random_crop_size values must be positive, got {(crop_h, crop_w)}")
    return crop_h, crop_w


def _normalize_scale_factors(channels: tuple[str, ...], value: float | Mapping[str, float] | None) -> dict[str, float]:
    if value is None:
        value = _DEFAULT_SCALE_FACTORS
    if isinstance(value, Mapping):
        scale_factors = {channel: float(value.get(channel, _DEFAULT_SCALE_FACTORS.get(channel, 1.0))) for channel in channels}
        zero_channels = [channel for channel, scale_factor in scale_factors.items() if scale_factor == 0.0]
        if len(zero_channels) > 0:
            raise ValueError(f"scale_factors must be non-zero for channels {zero_channels}")
        return scale_factors
    scale_factor = float(value)
    if scale_factor == 0.0:
        raise ValueError("scale_factor must be non-zero")
    return {channel: scale_factor for channel in channels}


def _as_missing_tuple(value: object) -> tuple[int | float, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (value.item() if hasattr(value, "item") else value,)
    return tuple(value)  # type: ignore[arg-type]


def _normalize_missing_values(
    channels: tuple[str, ...],
    value: Mapping[str, object] | Sequence[int | float] | int | float | None,
) -> dict[str, tuple[int | float, ...]]:
    if value is None:
        return {channel: _as_missing_tuple(_DEFAULT_MISSING_VALUES.get(channel, ())) for channel in channels}
    if isinstance(value, Mapping):
        return {channel: _as_missing_tuple(value.get(channel, _DEFAULT_MISSING_VALUES.get(channel, ()))) for channel in channels}
    shared = _as_missing_tuple(value)
    return {channel: shared for channel in channels}


def _normalize_fill_values(channels: tuple[str, ...], value: float | Mapping[str, float | None] | None) -> dict[str, float | None]:
    if isinstance(value, Mapping):
        return {channel: value.get(channel, 0.0) for channel in channels}
    return {channel: value for channel in channels}


def _sliced_shape(shape: tuple[int, int], crop: Sequence[int] | None, stride: int) -> tuple[int, int]:
    height, width = shape
    if crop is None:
        row_slice = slice(0, height)
        col_slice = slice(0, width)
    else:
        row_slice, col_slice = _crop_to_slices(crop)
        if row_slice.stop is None or col_slice.stop is None or row_slice.stop > height or col_slice.stop > width:
            raise ValueError(f"crop {crop} exceeds MeteoNet radar grid {shape}")

    row_start, row_stop, _ = row_slice.indices(height)
    col_start, col_stop, _ = col_slice.indices(width)
    return len(range(row_start, row_stop, stride)), len(range(col_start, col_stop, stride))


@DATASETS.register()
class MeteoNetRadarDataset(Dataset):
    """Dataset for QCed MeteoNet radar NPZ chunks.

    Each NPZ file is expected to contain ``dates`` and one or more 3D arrays such
    as ``data``, ``prob`` or ``height`` with shape ``(time, height, width)``.
    The dataset discovers every NPZ under ``data_paths`` and uses the embedded
    ``dates`` arrays as the authoritative timeline.
    """

    def __init__(
        self,
        data_paths: str | Path | Iterable[str | Path],
        hist_frames: int,
        fcst_frames: int | Sequence[int],
        start_frames: int = 0,
        freq: int = 5,
        freq_unit: str = "minutes",
        interval: int = 1,
        training: bool = True,
        inference_only: bool = False,
        years: Sequence[object] | None = None,
        pattern: str = "*.npz",
        channels: str | Sequence[str] = ("data",),
        crop: Sequence[int] | None = None,
        stride: int = 1,
        lat_flip: bool = False,
        scale_factors: float | Mapping[str, float] | None = None,
        missing_values: Mapping[str, object] | Sequence[int | float] | int | float | None = None,
        fill_value: float | Mapping[str, float | None] | None = 0.0,
        exclude_missing_dates: bool = True,
        cache_size: int = 1,
        random_crop_size: Sequence[int] | int | None = None,
        echo_crop_prob: float = 0.7,
        echo_threshold_dbz: float = 20.0,
        echo_channel: str = "data",
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading MeteoNet radar data from: %s", data_paths)

        self.start_frames = int(start_frames)
        self.hist_frames = int(hist_frames)
        self.fcst_frames = np.atleast_1d(np.asarray(fcst_frames, dtype=np.int32)) + self.start_frames
        self.training = bool(training)
        self.inference_only = bool(inference_only)
        self.channels = _as_channels(channels)
        self.crop = crop
        self.stride = int(stride)
        self.lat_flip = bool(lat_flip)
        self.scale_factors = _normalize_scale_factors(self.channels, scale_factors)
        self.missing_values = _normalize_missing_values(self.channels, missing_values)
        self.fill_values = _normalize_fill_values(self.channels, fill_value)
        self.cache_size = int(cache_size)
        self.random_crop_size = _normalize_crop_size(random_crop_size)
        self.echo_crop_prob = float(echo_crop_prob)
        self.echo_threshold_dbz = float(echo_threshold_dbz)

        if self.hist_frames <= 0:
            raise ValueError(f"hist_frames must be positive, got {self.hist_frames}")
        if np.any(self.fcst_frames <= 0):
            raise ValueError(f"fcst_frames + start_frames must be positive, got {self.fcst_frames.tolist()}")
        if self.training and len(self.fcst_frames) != 1:
            raise ValueError("MeteoNetRadarDataset training mode expects a single fcst_frames value")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {self.cache_size}")
        if not 0.0 <= self.echo_crop_prob <= 1.0:
            raise ValueError(f"echo_crop_prob must be in [0, 1], got {echo_crop_prob}")
        if self.random_crop_size is not None and echo_channel not in self.channels:
            raise ValueError(f"echo_channel {echo_channel!r} is not present in channels {self.channels}")
        self.echo_channel_index = self.channels.index(echo_channel) if echo_channel in self.channels else 0

        self.records, source_grid_shape = _discover_meteonet_records(
            data_paths,
            pattern=pattern,
            channels=self.channels,
            years=years,
            exclude_missing_dates=exclude_missing_dates,
        )
        self.times = [record.time for record in self.records]
        self.source_grid_shape = source_grid_shape

        sliced_grid_shape = _sliced_shape(source_grid_shape, self.crop, self.stride)
        if self.random_crop_size is not None:
            crop_h, crop_w = self.random_crop_size
            if crop_h > sliced_grid_shape[0] or crop_w > sliced_grid_shape[1]:
                raise ValueError(f"random_crop_size {(crop_h, crop_w)} exceeds sliced MeteoNet radar grid {sliced_grid_shape}")
            self.output_grid_shape = (crop_h, crop_w)
        else:
            self.output_grid_shape = sliced_grid_shape
        self.grid_shape = torch.tensor(self.output_grid_shape, dtype=torch.long)

        if self.inference_only:
            self.total_frames = self.hist_frames
        elif self.training:
            self.total_frames = self.hist_frames + int(self.fcst_frames[0])
        else:
            self.total_frames = self.hist_frames + int(self.fcst_frames.max())

        step = pd.Timedelta(**{freq_unit: freq})
        self.inds = _make_sequence_indices(self.times, self.total_frames, step=step, interval=interval)
        if len(self.inds) == 0:
            raise ValueError(f"No regular MeteoNet radar sequences found with total_frames={self.total_frames}, freq={freq} {freq_unit}")

        self.id_to_time = {int(i): self.times[int(i) + self.hist_frames - 1] for i in self.inds}
        self.start_time = self.id_to_time[int(self.inds[0])]
        self.end_time = self.id_to_time[int(self.inds[-1])]
        self._array_cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

        stage = "train" if self.training else "test"
        self.logger.info(
            "%s_times: %s ~ %s = %d samples, channels=%s, grid=%s",
            stage,
            self.start_time,
            self.end_time,
            len(self.inds),
            self.channels,
            self.output_grid_shape,
        )

    def __len__(self):
        return len(self.inds)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array_cache"] = OrderedDict()
        return state

    def _rand_another(self):
        return np.random.randint(len(self))

    def __getitem__(self, idx):
        while True:
            data = self.prepare_data(idx)
            if self.training and data is None:
                idx = self._rand_another()
                continue
            return data

    def _load_channel_arrays(self, path: Path) -> dict[str, np.ndarray]:
        if self.cache_size > 0 and path in self._array_cache:
            arrays = self._array_cache.pop(path)
            self._array_cache[path] = arrays
            return arrays

        with np.load(path, allow_pickle=True) as npz_file:
            arrays = {channel: np.asarray(npz_file[channel]) for channel in self.channels}

        if self.cache_size > 0:
            self._array_cache[path] = arrays
            while len(self._array_cache) > self.cache_size:
                self._array_cache.popitem(last=False)
        return arrays

    def _apply_spatial_selection(self, array: np.ndarray) -> np.ndarray:
        if self.crop is not None:
            row_slice, col_slice = _crop_to_slices(self.crop)
            array = array[row_slice, col_slice]
        if self.stride > 1:
            array = array[:: self.stride, :: self.stride]
        return array

    def _decode_channel(self, channel: str, raw: np.ndarray) -> np.ndarray:
        missing_values = self.missing_values[channel]
        missing_mask = np.isin(raw, np.asarray(missing_values, dtype=raw.dtype)) if len(missing_values) > 0 else np.zeros(raw.shape, dtype=bool)
        data = raw.astype(np.float32, copy=True)
        scale_factor = self.scale_factors[channel]
        if scale_factor != 1.0:
            data /= scale_factor
        fill_value = self.fill_values[channel]
        if fill_value is not None and np.any(missing_mask):
            data[missing_mask] = float(fill_value)
        return data

    def _read_frame(self, record_index: int, sample_cache: dict[Path, dict[str, np.ndarray]]) -> np.ndarray:
        record = self.records[record_index]
        arrays = sample_cache.get(record.path)
        if arrays is None:
            arrays = self._load_channel_arrays(record.path)
            sample_cache[record.path] = arrays

        channels = []
        for channel in self.channels:
            raw = self._apply_spatial_selection(arrays[channel][record.index])
            channels.append(self._decode_channel(channel, raw))
        return np.stack(channels, axis=0)

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
        sample_cache: dict[Path, dict[str, np.ndarray]] = {}
        imgs = np.stack([self._read_frame(int(frame_idx), sample_cache) for frame_idx in frame_inds], axis=0)
        imgs = torch.from_numpy(imgs)
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
        elif torch.rand(()) < self.echo_crop_prob and torch.any(imgs[:, self.echo_channel_index] >= self.echo_threshold_dbz):
            echo_pixels = torch.nonzero(torch.amax(imgs[:, self.echo_channel_index], dim=0) >= self.echo_threshold_dbz, as_tuple=False)
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
