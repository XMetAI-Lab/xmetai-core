#!/usr/bin/env python3
"""
Compute multi-year mean (climatology) for a time dimension of a Zarr dataset.

Given an input Zarr store with a ``time`` dimension, this script:

1. Selects data between the provided start and end timestamps.
2. Buckets timestamps by intra-year slots (day or day+hour depending on freq).
3. Averages over all years for each slot via Dask-powered ``groupby`` reductions.
4. Writes the resulting climatology to a new Zarr store.

Typical usage::

    python calculate_zarr_mean.py \\
        --input /data/era5.zarr \\
        --output /data/era5_climatology.zarr \\
        --freq 24h \\
        --start 19900101 --end 20201231 \\
        --local-workers 8 --memory-limit 16GB \\
        --overwrite

The output Zarr has one year of timestamps (e.g., 365 daily snapshots), each
holding the multi-year mean for that calendar slot.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

#: Map frequency key → (pandas frequency alias, strftime format string).
FREQ_MAP: Dict[str, tuple[str, str]] = {
    "6h": ("6H", "%m-%d-%H"),
    "24h": ("1D", "%m-%d"),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute intra-year multi-year mean for a Zarr dataset.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input Zarr store.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Destination path for the climatology Zarr store.",
    )
    parser.add_argument(
        "--freq",
        "-f",
        required=True,
        choices=list(FREQ_MAP),
        help="Temporal resolution: '6h' (6-hourly) or '24h' (daily).",
    )
    parser.add_argument(
        "--start",
        "-s",
        default="19500101",
        help="Start date (YYYYMMDD).  Default: 19500101.",
    )
    parser.add_argument(
        "--end",
        "-e",
        default="20251231",
        help="End date (YYYYMMDD).  Default: 20251231.",
    )
    parser.add_argument(
        "--consolidated",
        action="store_true",
        help="Open the input Zarr with consolidated metadata (faster for large stores).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output path if it already exists.",
    )
    parser.add_argument(
        "--chunks",
        default="auto",
        help="Chunk spec for xarray, e.g. 'time:365,lat:90'.  Default: auto.",
    )
    parser.add_argument(
        "--scheduler",
        help="Existing Dask scheduler address (e.g., tcp://127.0.0.1:8786).",
    )
    parser.add_argument(
        "--local-workers",
        type=int,
        default=None,
        help="Number of workers for an auto-provisioned local Dask cluster.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Threads per worker when launching a local cluster.",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Memory limit per worker, e.g. '8GB'.",
    )
    parser.add_argument(
        "--dashboard-address",
        default=None,
        help="Dashboard bind address for the local Dask cluster (e.g., :8787).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Open the dataset and print summary, but do not compute or write.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------


def parse_chunk_spec(spec: str) -> str | dict[str, int]:
    """Parse a ``dim:size,dim:size`` chunk string into a dict for xarray.

    Returns ``"auto"`` when *spec* is empty or the literal ``"auto"``.
    """
    normalized = spec.strip()
    if not normalized or normalized.lower() == "auto":
        return "auto"

    chunk_dict: dict[str, int] = {}
    for token in normalized.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid chunk segment '{token}'. Expected dim:size.")
        dim, size = token.split(":", 1)
        chunk_dict[dim.strip()] = int(size)

    if not chunk_dict:
        raise ValueError("Chunk specification produced no valid entries.")
    return chunk_dict


# ---------------------------------------------------------------------------
# Dask cluster
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def dask_client_context(
    *,
    scheduler: str | None = None,
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    memory_limit: str | None = None,
    dashboard_address: str | None = None,
) -> Iterator[Any]:
    """Yield a Dask ``Client``, optionally launching a local cluster."""
    from dask.distributed import Client, LocalCluster

    cluster = None
    if scheduler:
        logger.info("Connecting to existing Dask scheduler at %s", scheduler)
        client = Client(address=scheduler)
    else:
        cluster_kwargs: dict[str, Any] = {}
        if n_workers is not None:
            cluster_kwargs["n_workers"] = n_workers
        if threads_per_worker is not None:
            cluster_kwargs["threads_per_worker"] = threads_per_worker
        if memory_limit is not None:
            cluster_kwargs["memory_limit"] = memory_limit
        if dashboard_address is not None:
            cluster_kwargs["dashboard_address"] = dashboard_address
        logger.info("Starting local Dask cluster%s", f" ({n_workers} workers)" if n_workers else "")
        cluster = LocalCluster(**cluster_kwargs)
        client = Client(cluster)

    try:
        scheduler_info = client.scheduler_info()
        worker_ct = len(scheduler_info.get("workers", {}))
        print(f"[dask] Connected to cluster with {worker_ct} worker(s).")
        dashboard = getattr(client, "dashboard_link", None)
        if dashboard:
            print(f"[dask] Dashboard: {dashboard}")
        yield client
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def open_zarr_dataset(
    store: Path,
    consolidated: bool,
    chunks: str | dict[str, int],
) -> xr.Dataset:
    """Open a Zarr store, falling back to non-consolidated on metadata errors."""
    try:
        return xr.open_zarr(store, consolidated=consolidated, chunks=chunks)  # type: ignore[arg-type]
    except (ValueError, KeyError):
        if consolidated:
            logger.warning("Consolidated metadata read failed, retrying without consolidation.")
            return xr.open_zarr(store, consolidated=False, chunks=chunks)  # type: ignore[arg-type]
        raise


def compute_slot_mean(
    ds: xr.Dataset,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    freq_key: str,
) -> xr.Dataset:
    """Compute multi-year mean bucketed by intra-year time slots.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a ``time`` dimension.
    start_ts / end_ts : pd.Timestamp
        Time range to include.
    freq_key : str
        Key into :data:`FREQ_MAP` (``"6h"`` or ``"24h"``).

    Returns
    -------
    xr.Dataset
        Climatology with one representative year of timestamps.
    """
    freq_alias, slot_fmt = FREQ_MAP[freq_key]

    sliced = ds.sel(time=slice(start_ts, end_ts))
    if "time" not in sliced.dims:
        raise ValueError("Dataset must contain a 'time' dimension.")
    if sliced.time.size == 0:
        raise ValueError(f"No data found in time range [{start_ts}, {end_ts}].")

    time_index = pd.DatetimeIndex(sliced.indexes["time"])
    n_steps = time_index.size
    logger.info("Selected %d time steps between %s and %s.", n_steps, start_ts, end_ts)

    slot_strings = time_index.strftime(slot_fmt)
    unique_slots = pd.unique(slot_strings)
    logger.info("Found %d unique intra-year slots for freq='%s'.", len(unique_slots), freq_key)

    sliced = sliced.assign_coords(time_slot=("time", slot_strings))

    # Dask-powered groupby + mean
    print(f"[compute] Grouping {n_steps} steps into {len(unique_slots)} slots …")
    grouped = sliced.groupby("time_slot").mean("time")

    # Enforce natural calendar order
    grouped = grouped.sel(time_slot=pd.Index(unique_slots))

    # Build a representative year of reference timestamps
    reference_start = start_ts.replace(year=start_ts.year)
    reference_times = pd.date_range(start=reference_start, periods=len(unique_slots), freq=freq_alias)

    climatology = grouped.rename({"time_slot": "time"}).assign_coords(time=("time", reference_times))
    climatology.attrs.update(
        {
            "source_range": f"{start_ts.isoformat()} to {end_ts.isoformat()}",
            "freq": freq_key,
            "description": "Multi-year mean computed per intra-year time slot.",
        }
    )
    return climatology


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- Logging setup ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(message)s",
    )

    input_store = args.input.expanduser().resolve()
    output_store = args.output.expanduser().resolve()

    # --- Validate paths ---
    if not input_store.exists():
        raise FileNotFoundError(f"Input Zarr store not found: {input_store}")
    if output_store.exists():
        if args.overwrite:
            logger.warning("Removing existing output: %s", output_store)
            shutil.rmtree(output_store)
        else:
            raise FileExistsError(f"Output path already exists: {output_store}\n" "Use --overwrite to replace it.")

    start_ts = pd.to_datetime(args.start)
    end_ts = pd.to_datetime(args.end)
    if end_ts < start_ts:
        raise ValueError(f"End date ({args.end}) must be >= start date ({args.start}).")

    try:
        chunk_spec = parse_chunk_spec(args.chunks)
    except ValueError as exc:
        raise SystemExit(f"Invalid --chunks: {exc}") from exc

    # --- Open dataset ---
    with dask_client_context(
        scheduler=args.scheduler,
        n_workers=args.local_workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
        dashboard_address=args.dashboard_address,
    ):
        print(f"[load] Opening {input_store}")
        ds = open_zarr_dataset(input_store, args.consolidated, chunk_spec)

        if args.dry_run:
            print(f"[dry-run] Dataset summary:\n{ds}")
            return

        # --- Compute ---
        climatology = compute_slot_mean(ds, start_ts, end_ts, args.freq)

        # --- Write ---
        print(f"[write] Saving climatology to {output_store}")
        climatology.to_zarr(output_store, mode="w", consolidated=True)

    print(f"[done] Climatology written to {output_store}")


if __name__ == "__main__":
    main()
