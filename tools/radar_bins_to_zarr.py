#!/usr/bin/env python
"""Convert CINRAD composite reflectivity binaries into an int16 Zarr dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_radar_utils():
    """Load the lightweight radar reader without importing the training package."""

    module_path = REPO_ROOT / "xmetai" / "utils" / "radar.py"
    spec = importlib.util.spec_from_file_location("_xmetai_radar_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load radar utilities from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_radar_utils = _load_radar_utils()
RadarQCConfig = _radar_utils.RadarQCConfig
_validate_common_grid = _radar_utils._validate_common_grid
bbox_to_slices = _radar_utils.bbox_to_slices
discover_radar_files = _radar_utils.discover_radar_files
read_radar_reflectivity = _radar_utils.read_radar_reflectivity

logger = logging.getLogger(__name__)


def _parse_bbox(values: Sequence[str] | None) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--bbox expects four values: lon_min lat_min lon_max lat_max")
    return tuple(float(value) for value in values)


def _slice_lat_lon(header, bbox: Sequence[float] | None, stride: int) -> tuple[np.ndarray, np.ndarray]:
    lat = header.lat
    lon = header.lon
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        lat = lat[row_slice]
        lon = lon[col_slice]
    if stride > 1:
        lat = lat[::stride]
        lon = lon[::stride]
    return lat.astype(np.float32, copy=False), lon.astype(np.float32, copy=False)


def _encode_dbz10(data: np.ndarray, fill_value: int, valid_min: int, valid_max: int) -> np.ndarray:
    encoded = np.full(data.shape, fill_value, dtype=np.int16)
    valid = np.isfinite(data)
    if np.any(valid):
        dbz10 = np.rint(data[valid] * 10.0).astype(np.int32)
        encoded[valid] = np.clip(dbz10, valid_min, valid_max).astype(np.int16)
    return encoded


def _make_qc_config(args) -> RadarQCConfig:
    return RadarQCConfig(
        enabled=not args.no_qc,
        min_dbz=args.qc_min_dbz,
        max_dbz=args.qc_max_dbz,
        speckle_dbz=args.qc_speckle_dbz,
        speckle_min_neighbors=args.qc_speckle_min_neighbors,
        spike_delta_dbz=args.qc_spike_delta_dbz,
        fill_value=np.nan,
    )


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _open_zarr_v2_group(output: Path):
    try:
        return zarr.open_group(str(output), mode="w", zarr_format=2)
    except TypeError as exc:
        if "zarr_format" not in str(exc):
            raise
        return zarr.open_group(str(output), mode="w", zarr_version=2)


def _create_zarr_array(group, name: str, **kwargs):
    create_array = getattr(group, "create_array", None)
    if create_array is not None:
        return create_array(name, **kwargs)

    if "data" in kwargs:
        array = getattr(group, "array", None)
        if array is None:
            raise AttributeError("Zarr group does not provide create_array or array for data-backed array creation.")
        return array(name, **kwargs)

    create_dataset = getattr(group, "create_dataset", None)
    if create_dataset is None:
        raise AttributeError("Zarr group does not provide create_array or create_dataset for shaped array creation.")
    return create_dataset(name, **kwargs)


def _write_coord(group, name: str, data: np.ndarray, dim: str, chunks: tuple[int, ...] | None = None) -> None:
    array = _create_zarr_array(group, name, data=data, chunks=chunks or data.shape, overwrite=True)
    array.attrs["_ARRAY_DIMENSIONS"] = [dim]


def _timedelta_to_ns(delta) -> int:
    if hasattr(delta, "value"):
        return int(delta.value)
    return int(delta / np.timedelta64(1, "ns"))


def _estimate_missing_count(diff, expected_step) -> int | None:
    diff_ns = _timedelta_to_ns(diff)
    expected_ns = _timedelta_to_ns(expected_step)
    if expected_ns <= 0 or diff_ns <= expected_ns:
        return 0
    if diff_ns % expected_ns != 0:
        return None
    return diff_ns // expected_ns - 1


def _segment_summary(records) -> dict[str, object]:
    return {
        "time_start": str(records[0].time),
        "time_end": str(records[-1].time),
        "time_count": len(records),
    }


def _analyze_continuity(records):
    if len(records) <= 1:
        return None, [], _segment_summary(records)

    diffs = [records[i].time - records[i - 1].time for i in range(1, len(records))]
    counts = {}
    for diff in diffs:
        if diff <= np.timedelta64(0, "ns"):
            continue
        counts[diff] = counts.get(diff, 0) + 1
    if len(counts) == 0:
        raise ValueError("Could not infer a positive radar time interval from input files.")

    expected_step = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    segments = []
    gaps = []
    current = [records[0]]
    for prev, record, diff in zip(records[:-1], records[1:], diffs):
        if diff == expected_step:
            current.append(record)
            continue
        segments.append(current)
        gaps.append(
            {
                "prev_time": str(prev.time),
                "next_time": str(record.time),
                "delta": str(diff),
                "missing_count": _estimate_missing_count(diff, expected_step),
            }
        )
        current = [record]
    segments.append(current)

    longest = sorted(segments, key=lambda segment: (-len(segment), segment[0].time))[0]
    return expected_step, gaps, _segment_summary(longest)


def _create_store(args, records, lat: np.ndarray, lon: np.ndarray, qc_config: RadarQCConfig, continuity: dict[str, object]):
    output = Path(args.output).expanduser()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output Zarr already exists: {output}. Pass --overwrite to replace it.")
        if not output.is_dir():
            raise ValueError(f"Output exists and is not a directory: {output}")

    compressor = Blosc(cname=args.compressor, clevel=args.compression_level, shuffle=Blosc.BITSHUFFLE)
    group = _open_zarr_v2_group(output)

    times = np.asarray([record.time.to_datetime64() for record in records], dtype="datetime64[ns]")
    _write_coord(group, "time", times, "time", chunks=(min(args.time_chunk, len(times)),))
    level_dtype = f"U{max(1, len(args.level_name))}"
    _write_coord(group, "level", np.asarray([args.level_name], dtype=level_dtype), "level", chunks=(1,))
    _write_coord(group, "lat", lat, "lat", chunks=(min(args.y_chunk, len(lat)),))
    _write_coord(group, "lon", lon, "lon", chunks=(min(args.x_chunk, len(lon)),))

    chunks = (min(args.time_chunk, len(records)), 1, min(args.y_chunk, len(lat)), min(args.x_chunk, len(lon)))
    data = _create_zarr_array(
        group,
        "data",
        shape=(len(records), 1, len(lat), len(lon)),
        chunks=chunks,
        dtype=np.int16,
        compressor=compressor,
        fill_value=None,
        overwrite=True,
    )
    data.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time", "level", "lat", "lon"],
            "units": "dBZ",
            "_FillValue": args.fill_value,
            "scale_factor": 0.1,
            "add_offset": 0.0,
            "fill_value": args.fill_value,
            "dbz_scale_factor": 0.1,
            "dbz_add_offset": 0.0,
            "valid_min": args.valid_min,
            "valid_max": args.valid_max,
            "encoding": "int16 dbz10 with CF scale_factor/add_offset/_FillValue; physical_dBZ = value * 0.1",
        }
    )
    group.attrs.update(
        {
            "product": "ACHN_CR",
            "source_format": "CINRAD composite reflectivity binary",
            "source_paths": [str(Path(args.input).expanduser())],
            "bbox": list(args.bbox) if args.bbox is not None else None,
            "stride": args.stride,
            "time_start": str(records[0].time),
            "time_end": str(records[-1].time),
            "time_count": len(records),
            "expected_time_step": continuity["expected_time_step"],
            "time_gaps": continuity["time_gaps"],
            "longest_continuous_segment": continuity["longest_continuous_segment"],
            "writer": "dask-threaded" if args.workers > 1 else "serial",
            "writer_workers": args.workers,
            "qc": _json_safe(json.loads(json.dumps(qc_config.__dict__))),
            "notes": "Stored as physical reflectivity code, not model-normalized values.",
        }
    )
    return group, data


def _iter_record_batches(record_count: int, batch_size: int):
    for start in range(0, record_count, batch_size):
        yield start, min(start + batch_size, record_count)


def _align_task_size_to_time_chunk(time_chunk: int, task_size: int) -> int:
    return max(time_chunk, math.ceil(task_size / time_chunk) * time_chunk)


def _write_record_batch(data, records, start: int, args, qc_config: RadarQCConfig, spatial_shape: tuple[int, int]) -> int:
    block = np.empty((len(records), *spatial_shape), dtype=np.int16)
    for offset, record in enumerate(records):
        frame = read_radar_reflectivity(record.path, header=record.header, bbox=args.bbox, stride=args.stride, fill_value=np.nan, qc=qc_config)
        encoded = _encode_dbz10(frame, fill_value=args.fill_value, valid_min=args.valid_min, valid_max=args.valid_max)
        if encoded.shape != spatial_shape:
            raise ValueError(f"Encoded radar frame shape {encoded.shape} does not match output spatial shape {spatial_shape}: {record.path}")
        block[offset] = encoded
    data[start : start + len(records), 0, :, :] = block
    return len(records)


def _write_records_serial(data, records, args, qc_config: RadarQCConfig, spatial_shape: tuple[int, int]) -> int:
    written = 0
    for start, end in tqdm(_iter_record_batches(len(records), 1), total=len(records), desc="Writing radar zarr", unit="frame"):
        written += _write_record_batch(data, records[start:end], start, args, qc_config, spatial_shape)
    return written


def _load_dask():
    try:
        from dask import compute, delayed
        from dask.callbacks import Callback
    except ImportError as exc:
        raise ImportError("Parallel radar Zarr writing requires dask. Install dask or run with --workers 1.") from exc
    return compute, delayed, Callback


def _make_tqdm_dask_callback(callback_cls, total: int):
    progress = None
    completed = 0

    def start(_dsk):
        nonlocal progress
        progress = tqdm(total=total, desc="Writing radar zarr", unit="task", dynamic_ncols=True)

    def posttask(_key, _result, _dsk, _state, _worker_id):
        nonlocal completed
        completed += 1
        if progress is not None:
            progress.update(1)

    def finish(_dsk, _state, _failed):
        if progress is not None:
            progress.close()

    return callback_cls(start=start, posttask=posttask, finish=finish)


def _write_records_dask(data, records, args, qc_config: RadarQCConfig, spatial_shape: tuple[int, int], batch_size: int) -> int:
    compute, delayed, Callback = _load_dask()
    tasks = [
        delayed(_write_record_batch)(data, records[start:end], start, args, qc_config, spatial_shape)
        for start, end in _iter_record_batches(len(records), batch_size)
    ]
    logger.info("Launching %d Dask threaded write task(s) with %d worker(s).", len(tasks), args.workers)
    with _make_tqdm_dask_callback(Callback, len(tasks)):
        counts = compute(*tasks, scheduler="threads", num_workers=args.workers)
    return int(sum(counts))


def _write_records(data, records, args, qc_config: RadarQCConfig, spatial_shape: tuple[int, int], batch_size: int) -> int:
    if args.workers == 1:
        return _write_records_serial(data, records, args, qc_config, spatial_shape)
    dask_batch_size = _align_task_size_to_time_chunk(batch_size, args.task_size)
    return _write_records_dask(data, records, args, qc_config, spatial_shape, dask_batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Radar .bin file or directory containing radar files.")
    parser.add_argument("output", help="Output Zarr directory.")
    parser.add_argument("--pattern", default="*.bin", help="Directory search pattern. Default: %(default)s")
    parser.add_argument("--years", nargs=2, metavar=("START", "END"), help="Optional time prefilter before continuity scanning.")
    parser.add_argument("--bbox", nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"), help="Optional lon/lat crop before writing.")
    parser.add_argument("--stride", type=int, default=1, help="Optional spatial downsampling stride. Default keeps native resolution.")
    parser.add_argument("--level-name", default="cr", help="Name for the single level/channel coordinate.")
    parser.add_argument("--fill-value", type=int, default=-32768, help="int16 sentinel for no valid echo.")
    parser.add_argument("--valid-min", type=int, default=0, help="Minimum valid encoded dbz10 value.")
    parser.add_argument("--valid-max", type=int, default=750, help="Maximum valid encoded dbz10 value.")
    parser.add_argument("--time-chunk", type=int, default=1, help="Zarr chunk size along time.")
    parser.add_argument("--y-chunk", type=int, default=0, help="Zarr chunk size along latitude. Default 0 uses the full latitude axis.")
    parser.add_argument("--x-chunk", type=int, default=0, help="Zarr chunk size along longitude. Default 0 uses the full longitude axis.")
    parser.add_argument("--compressor", default="zstd", choices=["zstd", "lz4", "zlib", "blosclz"], help="Blosc compressor.")
    parser.add_argument("--compression-level", type=int, default=5, help="Blosc compression level.")
    parser.add_argument("--no-qc", action="store_true", help="Disable conservative reflectivity QC before writing.")
    parser.add_argument("--qc-min-dbz", type=float, default=0.0, help="Minimum valid reflectivity after QC.")
    parser.add_argument("--qc-max-dbz", type=float, default=75.0, help="Maximum valid reflectivity after QC.")
    parser.add_argument("--qc-speckle-dbz", type=float, default=5.0, help="Reflectivity threshold used for speckle support checks.")
    parser.add_argument("--qc-speckle-min-neighbors", type=int, default=2, help="Minimum echo pixels in the local QC window, including the center pixel.")
    parser.add_argument("--qc-spike-delta-dbz", type=float, default=25.0, help="Remove isolated spikes this many dBZ above all neighboring pixels.")
    parser.add_argument("--workers", type=int, default=1, help="Number of Dask threaded workers for data conversion and chunk writes. Default: serial.")
    parser.add_argument("--task-size", type=int, default=32, help="Number of time records per Dask task. Larger values reduce scheduler overhead.")
    parser.add_argument("--write", action="store_true", help="Actually write the Zarr dataset. Without this flag the script only prints a dry run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output Zarr directory.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    args.bbox = _parse_bbox(args.bbox)
    if args.stride <= 0:
        raise ValueError(f"--stride must be positive, got {args.stride}")
    if args.time_chunk <= 0:
        raise ValueError("time_chunk must be positive.")
    if args.y_chunk < 0 or args.x_chunk < 0:
        raise ValueError("y_chunk and x_chunk must be non-negative; use 0 for the full axis.")
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.task_size <= 0:
        raise ValueError(f"--task-size must be positive, got {args.task_size}")
    if args.fill_value < np.iinfo(np.int16).min or args.fill_value > np.iinfo(np.int16).max:
        raise ValueError(f"--fill-value must fit int16, got {args.fill_value}")

    records = discover_radar_files(args.input, pattern=args.pattern, years=args.years)
    expected_step, gaps, longest_segment = _analyze_continuity(records)
    continuity = {
        "expected_time_step": str(expected_step) if expected_step is not None else None,
        "time_gaps": gaps,
        "longest_continuous_segment": longest_segment,
    }
    first_header = _validate_common_grid(records)
    lat, lon = _slice_lat_lon(first_header, args.bbox, args.stride)
    if args.y_chunk == 0:
        args.y_chunk = len(lat)
    if args.x_chunk == 0:
        args.x_chunk = len(lon)
    shape = (len(records), 1, len(lat), len(lon))
    chunks = (min(args.time_chunk, len(records)), 1, min(args.y_chunk, len(lat)), min(args.x_chunk, len(lon)))
    raw_bytes = int(np.prod(shape) * np.dtype(np.int16).itemsize)

    logger.info("Scanned %d radar files: %s ~ %s", len(records), records[0].time, records[-1].time)
    if expected_step is not None:
        logger.info("Inferred regular interval: %s", expected_step)
    if len(gaps) == 0:
        logger.info("All scanned files are time-continuous.")
    else:
        logger.warning("Found %d time break(s); writing all scanned records and recording gaps.", len(gaps))
        for gap in gaps:
            logger.warning(
                "Time break: %s -> %s, delta=%s, missing_count=%s",
                gap["prev_time"],
                gap["next_time"],
                gap["delta"],
                gap["missing_count"],
            )
    logger.info(
        "Longest continuous segment: %d radar files, %s ~ %s",
        longest_segment["time_count"],
        longest_segment["time_start"],
        longest_segment["time_end"],
    )
    logger.info("Output shape: time=%d level=1 lat=%d lon=%d", shape[0], shape[2], shape[3])
    logger.info("Chunks: %s, raw int16 size before compression: %.2f MiB", chunks, raw_bytes / 1024**2)
    logger.info("Writer: %s, workers=%d", "Dask threaded" if args.workers > 1 else "serial", args.workers)
    if args.workers > 1:
        dask_task_size = _align_task_size_to_time_chunk(chunks[0], args.task_size)
        logger.info("Dask task size: %d time record(s), estimated tasks: %d", dask_task_size, math.ceil(len(records) / dask_task_size))
    logger.info("Encoding: int16 dbz10, dbz_scale_factor=0.1, encoded fill_value=%d", args.fill_value)

    qc_config = _make_qc_config(args)
    if not args.write:
        logger.info("Dry run only. Re-run with --write to create the Zarr store.")
        return

    _, data = _create_store(args, records, lat, lon, qc_config, continuity)
    written = _write_records(data, records, args, qc_config, (len(lat), len(lon)), chunks[0])
    if written != len(records):
        raise RuntimeError(f"Wrote {written} radar frame(s), expected {len(records)}")

    zarr.consolidate_metadata(str(Path(args.output).expanduser()))
    logger.info("Wrote Zarr dataset: %s", Path(args.output).expanduser())


if __name__ == "__main__":
    main()
