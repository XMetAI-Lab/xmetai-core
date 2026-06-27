#!/usr/bin/env python3
"""
Compute quantiles for Zarr samples grouped by month-day (MMDD).

Two input layouts are supported:

1. ``files`` (default): a directory of many Zarr stores. Files are grouped by an
   MMDD token embedded in each filename, merged into samples, and quantiles are
   computed along the sample axis.

2. ``timevar``: a single Zarr store shaped as (time, level, lat, lon). Time labels
   such as ``19790101-00`` or ``19790101-202401231`` are parsed from the init-time
   prefix before ``-``, grouped by MMDD, and quantiles are computed across all
   matching time steps for each calendar month-day.

Usage:
    python compute_quantile.py --layout files --input ./test_data --output ./quantile_results \\
        --quantiles 0.33,0.66 --gpus 8

    python compute_quantile.py --layout timevar --input ./dataset.zarr --output ./quantile_results \\
        --var data --quantiles 0.33,0.66 --time-range 19790101,20240131
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import xarray as xr

LEVEL_DIM_NAMES = ("level", "channel", "var", "variable")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute quantiles for Zarr samples grouped by month-day.")
    parser.add_argument(
        "--layout",
        choices=("files", "timevar"),
        default="files",
        help="Input layout: 'files' for many Zarr files, 'timevar' for one (time, level, lat, lon) store",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input path: directory of .zarr files (files) or a single .zarr store (timevar)",
    )
    parser.add_argument("--output", "-o", required=True, help="Directory to save quantile .npy files")
    parser.add_argument(
        "--dates",
        "-d",
        default="",
        help="Comma-separated month-day tokens to process (e.g. 0101,0104). Default: auto-discover",
    )
    parser.add_argument("--dates-file", help="Optional text file with one month-day token per line")
    parser.add_argument(
        "--quantiles",
        "-q",
        default="0.33,0.66",
        help="Comma-separated quantile levels in [0, 1]",
    )
    parser.add_argument(
        "--var",
        "-v",
        default=None,
        help="Variable name inside the Zarr store. Default: auto-detect ('data' for timevar layout)",
    )
    parser.add_argument(
        "--merge-dims",
        type=int,
        default=2,
        help="[files] Number of leading dimensions to merge into the sample axis (default: 2)",
    )
    parser.add_argument(
        "--date-start",
        type=int,
        default=4,
        help="[files] Start index of the month-day token in each filename (default: 4)",
    )
    parser.add_argument(
        "--date-len",
        type=int,
        default=4,
        help="[files] Length of the month-day token in each filename (default: 4)",
    )
    parser.add_argument(
        "--time-range",
        default="",
        help="[timevar] Optional init-time range as START,END (e.g. 19790101,20240131 or 19790101-00,20240131-18)",
    )
    parser.add_argument(
        "--split-dim",
        type=int,
        default=None,
        help="Dimension index to split across GPUs (default: 1, i.e. level for timevar layout)",
    )
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument(
        "--no-consolidated",
        action="store_true",
        help="Open Zarr stores with consolidated=False",
    )
    return parser.parse_args()


def parse_quantiles(raw: str) -> list[float]:
    """Parse comma-separated quantile levels."""
    quantiles = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not quantiles:
        raise ValueError("At least one quantile level must be provided")
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {quantile}")
    return quantiles


def parse_time_range(raw: str) -> tuple[str, str] | None:
    """Parse optional START,END time range."""
    if not raw.strip():
        return None
    parts = [value.strip() for value in raw.split(",") if value.strip()]
    if len(parts) != 2:
        raise ValueError("--time-range expects exactly two comma-separated values: START,END")
    return parts[0], parts[1]


def resolve_var_name(args: argparse.Namespace) -> str:
    """Resolve the data variable name for the selected layout."""
    if args.var:
        return args.var
    return "data"


def parse_dates(args: argparse.Namespace, month_days_from_time: list[str] | None = None) -> list[str]:
    """Resolve the list of month-day tokens to process."""
    if args.dates_file:
        path = Path(args.dates_file)
        if not path.is_file():
            print(f"Error: dates file '{path}' does not exist")
            sys.exit(1)
        dates = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if dates:
            return dates

    if args.dates.strip():
        return [value.strip() for value in args.dates.split(",") if value.strip()]

    if args.layout == "timevar":
        if not month_days_from_time:
            print("Error: no month-day groups discovered from the time coordinate")
            sys.exit(1)
        return month_days_from_time

    return discover_month_days(Path(args.input), args.date_start, args.date_len)


def discover_month_days(zarr_dir: Path, date_start: int, date_len: int) -> list[str]:
    """Discover unique month-day tokens from Zarr filenames."""
    if not zarr_dir.is_dir():
        print(f"Error: input directory '{zarr_dir}' does not exist")
        sys.exit(1)

    month_days: set[str] = set()
    for path in sorted(zarr_dir.iterdir()):
        if not path.name.endswith(".zarr"):
            continue
        token_end = date_start + date_len
        if len(path.name) < token_end:
            continue
        month_days.add(path.name[date_start:token_end])

    if not month_days:
        print(f"Error: no .zarr files found under '{zarr_dir}'")
        sys.exit(1)

    return sorted(month_days)


def time_to_month_day(value: object) -> str:
    """Extract an MMDD token from a time coordinate value."""
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%m%d")

    text = str(value).strip()
    if not text:
        raise ValueError("empty time coordinate value")

    if "-" in text:
        text = text.split("-", 1)[0]

    if len(text) >= 8 and text[:8].isdigit():
        return text[4:8]

    parsed = pd.to_datetime(text)
    return parsed.strftime("%m%d")


def discover_month_days_from_time(times: np.ndarray) -> list[str]:
    """Discover unique month-day tokens from a time coordinate."""
    month_days = {time_to_month_day(value) for value in times}
    return sorted(month_days)


def resolve_zarr_store(path: Path) -> Path:
    """Resolve a single Zarr store path."""
    if not path.exists():
        raise FileNotFoundError(f"input path '{path}' does not exist")

    if path.name.endswith(".zarr") or (path / ".zgroup").exists() or (path / "zarr.json").exists():
        return path

    if path.is_dir():
        zarr_paths = sorted(child for child in path.iterdir() if child.name.endswith(".zarr"))
        if len(zarr_paths) == 1:
            return zarr_paths[0]
        if len(zarr_paths) == 0:
            raise ValueError(f"no .zarr store found under '{path}'")
        raise ValueError(f"expected one .zarr store under '{path}', found {len(zarr_paths)}")

    raise ValueError(f"'{path}' is not a Zarr store")


def open_zarr_dataset(path: Path, consolidated: bool) -> xr.Dataset:
    """Open one Zarr dataset."""
    return xr.open_zarr(
        path,
        consolidated=consolidated,
        mask_and_scale=False,
        use_zarr_fill_value_as_mask=False,
    )


def select_data_array(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Select and normalize a (time, level, lat, lon) data array."""
    if var_name in ds:
        data = ds[var_name]
    elif "data" in ds:
        data = ds["data"]
    elif len(ds.data_vars) == 1:
        data = ds[list(ds.data_vars)[0]]
    else:
        available = list(ds.data_vars)
        raise KeyError(f"variable '{var_name}' not found, available: {available}")

    if "time" not in data.dims:
        raise ValueError(f"expected a 'time' dimension, got dims={data.dims}")

    level_dim = next((name for name in LEVEL_DIM_NAMES if name in data.dims), None)
    if level_dim is None:
        raise ValueError(f"expected one of {LEVEL_DIM_NAMES} in dims, got dims={data.dims}")

    if level_dim != "level":
        data = data.rename({level_dim: "level"})

    expected_tail = ("level", "lat", "lon")
    if data.dims != ("time", *expected_tail):
        raise ValueError(f"expected dims (time, level, lat, lon), got {data.dims}")

    return data


def load_timevar_dataset(
    path: Path,
    var_name: str,
    consolidated: bool,
    time_range: tuple[str, str] | None,
) -> xr.DataArray:
    """Load a normalized (time, level, lat, lon) array from one Zarr store."""
    ds = open_zarr_dataset(path, consolidated)
    try:
        data = select_data_array(ds, var_name)
        if time_range is not None:
            data = data.sel(time=slice(*time_range))
        if data.sizes["time"] == 0:
            raise ValueError(f"no time steps left after applying time range {time_range}")
        return data
    finally:
        ds.close()


def list_matched_zarr_files(zarr_dir: Path, target_month_day: str, date_start: int, date_len: int) -> list[Path]:
    """Return Zarr paths whose filename contains the requested month-day token."""
    token_end = date_start + date_len
    matched = []
    for path in sorted(zarr_dir.iterdir()):
        if not path.name.endswith(".zarr"):
            continue
        if len(path.name) >= token_end and path.name[date_start:token_end] == target_month_day:
            matched.append(path)
    return matched


def load_zarr_array(path: Path, var_name: str, consolidated: bool) -> np.ndarray:
    """Load one variable from a Zarr store as a NumPy array."""
    ds = open_zarr_dataset(path, consolidated)
    try:
        if var_name not in ds:
            available = list(ds.data_vars)
            raise KeyError(f"variable '{var_name}' not found, available: {available}")
        return np.asarray(ds[var_name].values)
    finally:
        ds.close()


def merge_samples(data: np.ndarray, merge_dims: int) -> np.ndarray:
    """Merge leading dimensions into a single sample axis."""
    if data.ndim <= merge_dims:
        raise ValueError(f"Expected more than {merge_dims} dimensions, got shape {data.shape}")
    sample_size = int(np.prod(data.shape[:merge_dims]))
    return data.reshape(sample_size, *data.shape[merge_dims:])


def compute_quantile_on_gpu(data_chunk: torch.Tensor, quantiles: list[float], device_id: int) -> torch.Tensor:
    """Compute quantiles for one chunk on a single GPU."""
    torch.cuda.set_device(device_id)
    data_chunk = data_chunk.cuda(device_id, non_blocking=True)
    results = [torch.quantile(data_chunk, quantile, dim=0) for quantile in quantiles]
    return torch.stack(results, dim=0).cpu()


def parallel_quantile_computation(
    data: torch.Tensor,
    quantiles: list[float],
    num_gpus: int,
    split_dim: int = 1,
) -> torch.Tensor:
    """Split one dimension across GPUs and compute quantiles along the sample axis."""
    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions (sample, ..., ...), got shape {tuple(data.shape)}")

    if not (0 <= split_dim < data.ndim):
        raise ValueError(f"split_dim must be in [0, {data.ndim - 1}], got {split_dim}")
    if split_dim == 0:
        raise ValueError("split_dim must not be 0 (sample axis); quantiles are computed along dim=0")

    split_size = int(data.shape[split_dim])
    effective_gpus = min(max(int(num_gpus), 1), split_size)
    if effective_gpus <= 1 or split_size <= 1:
        results = [torch.quantile(data, quantile, dim=0) for quantile in quantiles]
        return torch.stack(results, dim=0)

    chunk_size = (split_size + effective_gpus - 1) // effective_gpus
    chunks: list[tuple[int, torch.Tensor]] = []
    for device_id in range(effective_gpus):
        start_idx = device_id * chunk_size
        end_idx = min(start_idx + chunk_size, split_size)
        if start_idx >= end_idx:
            continue
        slices = [slice(None)] * data.ndim
        slices[split_dim] = slice(start_idx, end_idx)
        chunks.append((device_id, data[tuple(slices)]))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=effective_gpus) as pool:
        results = pool.starmap(
            compute_quantile_on_gpu,
            [(chunk, quantiles, device_id) for device_id, chunk in chunks],
        )

    return torch.cat(results, dim=split_dim)

def collect_arrays(
    zarr_paths: list[Path],
    var_name: str,
    consolidated: bool,
) -> np.ndarray | None:
    """Load and concatenate arrays from multiple Zarr stores."""
    arrays: list[np.ndarray] = []
    for path in zarr_paths:
        try:
            arrays.append(load_zarr_array(path, var_name, consolidated))
        except Exception as exc:
            print(f"Warning: skipping '{path.name}': {exc}")

    if not arrays:
        return None

    return np.concatenate(arrays, axis=0)


def save_quantile_result(output_dir: Path, target_month_day: str, quantile_result: np.ndarray) -> None:
    """Save one quantile result array."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_month_day}.npy"
    np.save(output_path, quantile_result)
    print(f"Saved: {output_path}")


def process_quantile_for_date_files(
    zarr_dir: Path,
    target_month_day: str,
    output_dir: Path,
    quantiles: list[float],
    num_gpus: int,
    var_name: str,
    merge_dims: int,
    date_start: int,
    date_len: int,
    consolidated: bool,
    split_dim: int,
) -> None:
    """Compute and save quantiles for one month-day group in files layout."""
    print(f"Processing month-day: {target_month_day}")

    matched_files = list_matched_zarr_files(zarr_dir, target_month_day, date_start, date_len)
    print(f"Found {len(matched_files)} files")

    if not matched_files:
        print(f"Warning: no files matched month-day '{target_month_day}'")
        return

    combined = collect_arrays(matched_files, var_name, consolidated)
    if combined is None:
        print(f"Error: no valid data loaded for month-day '{target_month_day}'")
        return

    reshaped = merge_samples(combined, merge_dims)
    print(f"Merged sample shape: {reshaped.shape}")

    torch_data = torch.from_numpy(reshaped).float()
    print(f"Computing quantiles on {num_gpus} GPU(s)...")
    quantile_result = parallel_quantile_computation(torch_data, quantiles, num_gpus, split_dim=split_dim)
    save_quantile_result(output_dir, target_month_day, quantile_result.numpy())


def process_quantile_for_date_timevar(
    data: xr.DataArray,
    target_month_day: str,
    output_dir: Path,
    quantiles: list[float],
    num_gpus: int,
    split_dim: int,
) -> None:
    """Compute and save quantiles for one month-day group in timevar layout."""
    print(f"Processing month-day: {target_month_day}")

    time_values = data.time.values
    matched_indices = [idx for idx, value in enumerate(time_values) if time_to_month_day(value) == target_month_day]
    print(f"Found {len(matched_indices)} time steps")

    if not matched_indices:
        print(f"Warning: no time steps matched month-day '{target_month_day}'")
        return

    subset = data.isel(time=matched_indices)
    sample_array = np.asarray(subset.values)
    print(f"Sample array shape: {sample_array.shape}")

    torch_data = torch.from_numpy(sample_array).float()
    print(f"Computing quantiles on {num_gpus} GPU(s)...")
    quantile_result = parallel_quantile_computation(torch_data, quantiles, num_gpus, split_dim=split_dim)
    save_quantile_result(output_dir, target_month_day, quantile_result.numpy())


def resolve_num_gpus(requested: int) -> int:
    """Clamp requested GPU count to the number of available CUDA devices."""
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)

    available = torch.cuda.device_count()
    if available == 0:
        print("Error: no CUDA devices found")
        sys.exit(1)

    num_gpus = min(max(requested, 1), available)
    if num_gpus != requested:
        print(f"Warning: requested {requested} GPU(s), using {num_gpus}")
    return num_gpus


def main() -> None:
    """Main function."""
    args = parse_args()
    quantiles = parse_quantiles(args.quantiles)
    num_gpus = resolve_num_gpus(args.gpus)
    var_name = resolve_var_name(args)
    split_dim = 1 if args.split_dim is None else args.split_dim
    consolidated = not args.no_consolidated

    output_dir = Path(args.output)
    input_path = Path(args.input)

    print(f"Layout: {args.layout}")
    print(f"Input path: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Variable: {var_name}")
    print(f"Quantiles: {quantiles}")
    print(f"Using {num_gpus} GPU(s)")

    if args.layout == "files":
        month_days = parse_dates(args)
        for target_month_day in month_days:
            process_quantile_for_date_files(
                zarr_dir=input_path,
                target_month_day=target_month_day,
                output_dir=output_dir,
                quantiles=quantiles,
                num_gpus=num_gpus,
                var_name=var_name,
                merge_dims=args.merge_dims,
                date_start=args.date_start,
                date_len=args.date_len,
                consolidated=consolidated,
                split_dim=split_dim,
            )
    else:
        time_range = parse_time_range(args.time_range)
        zarr_store = resolve_zarr_store(input_path)
        data = load_timevar_dataset(zarr_store, var_name, consolidated, time_range)
        print(f"Loaded Zarr store: {zarr_store}")
        print(f"Data shape: {tuple(data.shape)} dims={data.dims}")
        if time_range is not None:
            print(f"Applied time range: {time_range[0]} ~ {time_range[1]}")

        month_days = parse_dates(args, discover_month_days_from_time(data.time.values))
        for target_month_day in month_days:
            process_quantile_for_date_timevar(
                data=data,
                target_month_day=target_month_day,
                output_dir=output_dir,
                quantiles=quantiles,
                num_gpus=num_gpus,
                split_dim=split_dim,
            )

    print("Done.")


if __name__ == "__main__":
    main()
