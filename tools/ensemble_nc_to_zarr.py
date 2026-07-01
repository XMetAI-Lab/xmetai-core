#!/usr/bin/env python3
"""
Convert ensemble NetCDF forecasts into daily Zarr stores.

Operational context
-------------------
This module is used in the operational workflow of the FengShun v1.5 model at
the National Climate Center (NCC). It ingests streaming ensemble NetCDF outputs,
consolidates them into daily Zarr stores, and supports long-running production
runs with resume and optional source cleanup.

Workflow diagram
----------------

    Ensemble NetCDF tree                 This daemon                      Output
    under DATA_ROOT/YYYYMMDD/*.nc
              |
              v
    +----------------------+
    | Producer: scan tree|
    | group by init date   |
    +----------+-----------+
               |
               v
    +----------------------+
    | Stability check:     |
    | expected member count|
    | held for N seconds   |
    +----------+-----------+
               |
               v
    +----------------------+
    | Consumer: process    |
    | pool merges members  |
    | and lead times       |
    +----------+-----------+
               |
               v
    +----------------------+       +----------------------+
    | Atomic write         | ----> | OUTPUT_DIR/          |
    | YYYYMMDD.zarr.tmp    |       | YYYYMMDD.zarr        |
    +----------------------+       +----------+-----------+
                                              |
                                              v
                                   +----------------------+
                                   | Downstream FengShun  |
                                   | v1.5 training / infer|
                                   +----------------------+
               ^
               |
    +----------+-----------+
    | Monitor: status log  |
    | resume existing zarr |
    | optional source purge|
    +----------------------+

This tool watches a directory tree of NetCDF files, groups them by initialization
date, and merges ensemble members and lead times into one Zarr dataset per date.
It is designed for long-running ingestion pipelines with resume support and
optional cleanup of source files after successful conversion.

Input layout
------------
The scanner expects files under ``DATA_ROOT/YYYYMMDD/.../*.nc``:

- The first directory level below ``DATA_ROOT`` is the initialization date token.
- The NetCDF basename encodes lead time in hours, for example ``03.nc`` -> 3.
- Multiple files with the same lead time are treated as ensemble members and
  concatenated along a ``member`` dimension.
- Different lead times are concatenated along a ``lead_time`` dimension.

Output layout
-------------
For each stable date group the tool writes:

``OUTPUT_DIR/YYYYMMDD.zarr``

Each store contains the merged ensemble cube for that initialization date.
Temporary writes use ``*.zarr.tmp`` and are atomically renamed on success.

Processing model
----------------
The daemon uses three background threads:

1. **Producer** scans ``DATA_ROOT`` periodically and enqueues dates whose member
   count reaches ``--expected-members`` and stays stable for
   ``--stable-seconds``.
2. **Consumer** pulls queued dates and processes them in a
   ``ProcessPoolExecutor``.
3. **Monitor** prints periodic status reports and executes delayed source-file
   deletions.

Resume behavior
---------------
On startup the tool scans ``OUTPUT_DIR`` for existing ``*.zarr`` stores. Valid
stores are treated as already processed, so interrupted runs can continue
without redoing finished dates.

Usage
-----
Run once in the foreground:

    python tools/ensemble_nc_to_zarr.py \\
        --data-root /workspace/data/hjhdatasets/hresinfer \\
        --output-dir /workspace/data/hjhdatasets/tmp \\
        --expected-members 50 \\
        --max-workers 2

Use a custom Dask chunk map when opening NetCDF files:

    python tools/ensemble_nc_to_zarr.py \\
        --data-root /path/to/nc \\
        --output-dir /path/to/zarr \\
        --chunks '{"lat": 721, "lon": 1440}'

Delete source NetCDF files after a successful conversion:

    python tools/ensemble_nc_to_zarr.py \\
        --data-root /path/to/nc \\
        --output-dir /path/to/zarr \\
        --delete-source \\
        --delete-delay-seconds 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert ensemble NetCDF forecasts into daily Zarr stores.")
    parser.add_argument("--data-root", "-i", required=True, help="Root directory containing YYYYMMDD subdirectories of NetCDF files")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to write daily YYYYMMDD.zarr stores")
    parser.add_argument("--log-file", help="Optional log file path (default: OUTPUT_DIR/ensemble_nc_to_zarr.log)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Directory scan interval in seconds (default: 30)")
    parser.add_argument("--stable-seconds", type=int, default=10, help="Seconds a date must remain stable before enqueue (default: 10)")
    parser.add_argument("--min-members", type=int, default=2, help="Minimum member files required to consider a date (default: 2)")
    parser.add_argument("--expected-members", type=int, default=11, help="Member count required before a date is enqueued (default: 11)")
    parser.add_argument("--max-workers", type=int, default=None, help="Process pool size (default: CPU count - 1)")
    parser.add_argument(
        "--chunks",
        default='{"lat": 60, "lon": 120}',
        help='JSON object passed to xr.open_dataset(..., chunks=...) (default: {"lat": 60, "lon": 120})',
    )
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing Zarr stores and process all dates from scratch")
    parser.add_argument("--delete-source", action="store_true", help="Delete source NetCDF files after successful conversion")
    parser.add_argument("--delete-delay-seconds", type=int, default=60, help="Delay before deleting source files (default: 60)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def parse_chunks(raw: str) -> dict[str, int]:
    """Parse a JSON chunk mapping for xarray/dask."""
    try:
        chunks = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --chunks JSON: {raw}") from exc
    if not isinstance(chunks, dict):
        raise ValueError("--chunks must be a JSON object")
    return {str(key): int(value) for key, value in chunks.items()}


def setup_logging(log_file: Path | None, log_level: str) -> None:
    """Configure root logging for the daemon and worker processes."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def scan_date_groups(data_root: Path, min_members: int) -> dict[str, list[str]]:
    """Scan NetCDF files and group them by initialization date."""
    groups: dict[str, list[str]] = defaultdict(list)
    if not data_root.is_dir():
        logger.warning("Data root does not exist: %s", data_root)
        return {}

    for date_dir in sorted(data_root.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        if len(date_str) == 8 and date_str.isdigit():
            groups[date_str].extend(str(file_path) for file_path in sorted(date_dir.rglob("*.nc")))
    filtered = {date_str: files for date_str, files in groups.items() if len(files) >= min_members}
    logger.debug("Found %d date groups (%d before min-members filter)", len(filtered), len(groups))
    return filtered


def validate_zarr_file(zarr_path: Path) -> bool:
    """Return True when a Zarr store looks readable and non-empty."""
    try:
        with xr.open_zarr(zarr_path, consolidated=True) as dataset:
            required_coords = ("lat", "lon", "time")
            if not all(coord in dataset.coords for coord in required_coords):
                return False
            if len(dataset.data_vars) == 0:
                return False
            return all(
                not dataset[var_name].isnull().all().compute().item() for var_name in dataset.data_vars
            )
    except Exception:
        return False


def atomic_write_zarr(dataset: xr.Dataset, zarr_path: Path) -> None:
    """Write a Zarr store through a temporary path and rename atomically."""
    temp_path = zarr_path.with_suffix(zarr_path.suffix + ".tmp")
    if temp_path.exists():
        shutil.rmtree(temp_path)

    try:
        dataset.to_zarr(temp_path, consolidated=True, mode="w")
        if zarr_path.exists():
            shutil.rmtree(zarr_path)
        temp_path.rename(zarr_path)
        logger.info("Wrote Zarr store: %s", zarr_path)
    except Exception:
        if temp_path.exists():
            shutil.rmtree(temp_path)
        raise


def check_existing_zarrs(output_dir: Path, validate: bool = True) -> set[str]:
    """Collect already processed date tokens from existing Zarr stores."""
    existing: set[str] = set()
    if not output_dir.is_dir():
        return existing

    for zarr_path in sorted(output_dir.glob("*.zarr")):
        date_str = zarr_path.stem
        if len(date_str) != 8 or not date_str.isdigit():
            logger.warning("Skipping unexpected Zarr name: %s", zarr_path.name)
            continue
        if validate and not validate_zarr_file(zarr_path):
            logger.warning("Removing invalid Zarr store: %s", zarr_path)
            shutil.rmtree(zarr_path, ignore_errors=True)
            continue
        existing.add(date_str)
    logger.info("Found %d existing Zarr stores", len(existing))
    return existing


def process_single_group(task: tuple[list[str], str, str, dict[str, int]]) -> tuple[str, str | None]:
    """Process one initialization date inside a worker process."""
    files, date_str, output_dir, chunks = task
    worker_logger = logging.getLogger(__name__)
    pid = os.getpid()

    try:
        worker_logger.info("[pid=%s] Processing date %s with %d files", pid, date_str, len(files))

        leadtime_groups: dict[int, list[str]] = defaultdict(list)
        for file_path in files:
            stem = Path(file_path).stem
            try:
                leadtime_groups[int(stem)].append(file_path)
            except ValueError:
                worker_logger.warning("[pid=%s] Could not parse lead time from %s", pid, file_path)

        leadtime_datasets: list[xr.Dataset] = []
        for lead_time, leadtime_files in sorted(leadtime_groups.items()):
            member_datasets: list[xr.Dataset] = []
            for file_path in sorted(leadtime_files):
                try:
                    member_datasets.append(xr.open_dataset(file_path, chunks=chunks))
                except Exception as exc:
                    worker_logger.error("[pid=%s] Failed to open %s: %s", pid, file_path, exc)

            if not member_datasets:
                worker_logger.warning("[pid=%s] No valid files for lead_time=%s", pid, lead_time)
                continue

            leadtime_datasets.append(xr.concat(member_datasets, dim="member"))

        if not leadtime_datasets:
            worker_logger.error("[pid=%s] No valid lead-time groups for %s", pid, date_str)
            return date_str, None

        combined = xr.concat(leadtime_datasets, dim="lead_time")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        zarr_path = output_path / f"{date_str}.zarr"
        atomic_write_zarr(combined, zarr_path)

        if validate_zarr_file(zarr_path):
            worker_logger.info("[pid=%s] Completed %s", pid, date_str)
            return date_str, str(zarr_path)

        worker_logger.error("[pid=%s] Validation failed for %s", pid, date_str)
        return date_str, None
    except Exception as exc:
        worker_logger.error("[pid=%s] Failed to process %s: %s", pid, date_str, exc, exc_info=True)
        return date_str, None


@dataclass
class ProcessorConfig:
    """Runtime configuration for the ensemble ingestion daemon."""

    data_root: Path
    output_dir: Path
    poll_interval_sec: int = 30
    stable_seconds: int = 10
    min_members: int = 2
    expected_members: int = 11
    max_workers: int | None = None
    chunks: dict[str, int] = field(default_factory=lambda: {"lat": 60, "lon": 120})
    delete_source_after_process: bool = False
    delete_delay_seconds: int = 60
    resume: bool = True


class SharedState:
    """Thread-safe bookkeeping for processed and pending dates."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processed: set[str] = set()
        self._enqueued: set[str] = set()
        self._pending_deletes: dict[str, datetime] = {}

    def is_known(self, date_str: str) -> bool:
        with self._lock:
            return date_str in self._processed or date_str in self._enqueued

    def mark_processed(self, date_str: str) -> None:
        with self._lock:
            self._processed.add(date_str)
            self._enqueued.discard(date_str)

    def mark_enqueued(self, date_str: str) -> None:
        with self._lock:
            self._enqueued.add(date_str)

    def schedule_delete(self, date_str: str, delay_seconds: int) -> None:
        with self._lock:
            self._pending_deletes[date_str] = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

    def pop_due_deletes(self) -> list[str]:
        with self._lock:
            now = datetime.now(timezone.utc)
            due = [date_str for date_str, delete_at in self._pending_deletes.items() if now >= delete_at]
            for date_str in due:
                del self._pending_deletes[date_str]
            return due

    def add_existing(self, existing_dates: set[str]) -> int:
        with self._lock:
            added = {date_str for date_str in existing_dates if date_str not in self._processed}
            self._processed.update(added)
            return len(added)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "processed_count": len(self._processed),
                "enqueued_count": len(self._enqueued),
                "pending_deletes_count": len(self._pending_deletes),
            }


class StatsManager:
    """Simple counters for daemon progress reporting."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._counters = Counter(enqueued=0, processed=0, failed=0, deleted=0)
            self._start_time = time.time()

    def inc(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[key] += amount

    def summary(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(time.time() - self._start_time, 1.0)
            return {
                **dict(self._counters),
                "elapsed_seconds": elapsed,
                "processed_per_min": self._counters["processed"] * 60.0 / elapsed,
            }


class EnsembleProcessor:
    """Daemon that converts stable daily NetCDF groups into Zarr stores."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.state = SharedState()
        self.stats = StatsManager()
        self.task_queue: queue.Queue[tuple[list[str], str, str, dict[str, int]]] = queue.Queue()
        self.stop_event = threading.Event()
        self._pending_stable: dict[str, float] = {}
        self.threads: list[threading.Thread] = []

    def scan_groups(self) -> dict[str, list[str]]:
        return scan_date_groups(self.config.data_root, self.config.min_members)

    def producer(self) -> None:
        logger.info("Producer started, watching %s", self.config.data_root)
        while not self.stop_event.is_set():
            try:
                groups = self.scan_groups()
                new_count = 0
                now_ts = time.time()
                for date_str, files in groups.items():
                    if self.state.is_known(date_str):
                        continue

                    lead_counts = Counter(
                        Path(path).stem for path in files if Path(path).stem.isdigit()
                    )
                    member_count = min(lead_counts.values(), default=0)
                    if member_count >= self.config.expected_members:
                        first_seen = self._pending_stable.get(date_str)
                        if first_seen is None:
                            self._pending_stable[date_str] = now_ts
                        elif now_ts - first_seen >= self.config.stable_seconds and member_count >= self.config.expected_members:
                            task = (files, date_str, str(self.config.output_dir), self.config.chunks)
                            self.task_queue.put(task)
                            self.state.mark_enqueued(date_str)
                            self.stats.inc("enqueued")
                            self._pending_stable.pop(date_str, None)
                            new_count += 1
                    else:
                        self._pending_stable.pop(date_str, None)

                if new_count:
                    logger.info("Enqueued %d new date(s)", new_count)
                time.sleep(self.config.poll_interval_sec)
            except Exception as exc:
                logger.error("Producer scan failed: %s", exc, exc_info=True)
                time.sleep(self.config.poll_interval_sec)

    def consumer(self) -> None:
        max_workers = self.config.max_workers or max(1, (os.cpu_count() or 2) - 1)
        logger.info("Consumer started with %d worker(s)", max_workers)
        futures: dict[Future[tuple[str, str | None]], str] = {}
        max_pending = max_workers * 2

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            while not self.stop_event.is_set():
                while len(futures) < max_pending:
                    try:
                        task = self.task_queue.get(timeout=1)
                    except queue.Empty:
                        break
                    future = executor.submit(process_single_group, task)
                    futures[future] = task[1]

                for future in [item for item in futures if item.done()]:
                    date_str = futures.pop(future)
                    try:
                        result_date, zarr_path = future.result()
                    except Exception as exc:
                        logger.error("Task failed for %s: %s", date_str, exc, exc_info=True)
                        self.stats.inc("failed")
                        continue

                    if zarr_path:
                        self.state.mark_processed(result_date)
                        self.stats.inc("processed")
                        if self.config.delete_source_after_process:
                            self.state.schedule_delete(result_date, self.config.delete_delay_seconds)
                    else:
                        self.stats.inc("failed")

                time.sleep(0.1)

    def monitor(self) -> None:
        logger.info("Monitor started")
        last_report = time.time()
        last_cleanup = time.time()
        while not self.stop_event.is_set():
            try:
                now = time.time()
                if now - last_report >= 300:
                    stats = self.stats.summary()
                    state = self.state.summary()
                    logger.info(
                        "Status processed=%s failed=%s deleted=%s pending=%s speed=%.2f/min",
                        stats["processed"],
                        stats["failed"],
                        stats["deleted"],
                        state["enqueued_count"],
                        stats["processed_per_min"],
                    )
                    last_report = now

                if now - last_cleanup >= 60:
                    for date_str in self.state.pop_due_deletes():
                        self._delete_date_files(date_str)
                        self.stats.inc("deleted")
                    last_cleanup = now

                time.sleep(10)
            except Exception as exc:
                logger.error("Monitor failed: %s", exc, exc_info=True)
                time.sleep(30)

    def _delete_date_files(self, date_str: str) -> None:
        groups = self.scan_groups()
        files = groups.get(date_str, [])
        deleted = 0
        for file_path in files:
            try:
                Path(file_path).unlink(missing_ok=True)
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", file_path, exc)
        logger.info("Deleted %d/%d source files for %s", deleted, len(files), date_str)

    def start(self) -> None:
        if self.config.max_workers is None:
            self.config.max_workers = max(1, (os.cpu_count() or 2) - 1)

        self.threads = [
            threading.Thread(target=self.producer, name="Producer", daemon=True),
            threading.Thread(target=self.consumer, name="Consumer", daemon=True),
            threading.Thread(target=self.monitor, name="Monitor", daemon=True),
        ]
        for thread in self.threads:
            thread.start()
            logger.info("Started thread: %s", thread.name)

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("Thread did not stop cleanly: %s", thread.name)

    def run(self) -> None:
        if self.config.resume:
            existing = check_existing_zarrs(self.config.output_dir, validate=True)
            added = self.state.add_existing(existing)
            if added:
                self.stats.inc("processed", added)
                logger.info("Resumed with %d existing date(s)", added)

        self.start()
        try:
            while any(thread.is_alive() for thread in self.threads):
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file) if args.log_file else output_dir / "ensemble_nc_to_zarr.log"
    setup_logging(log_file, args.log_level)

    try:
        chunks = parse_chunks(args.chunks)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    config = ProcessorConfig(
        data_root=Path(args.data_root),
        output_dir=output_dir,
        poll_interval_sec=args.poll_interval,
        stable_seconds=args.stable_seconds,
        min_members=args.min_members,
        expected_members=args.expected_members,
        max_workers=args.max_workers,
        chunks=chunks,
        delete_source_after_process=args.delete_source,
        delete_delay_seconds=args.delete_delay_seconds,
        resume=not args.no_resume,
    )

    logger.info("Starting ensemble NetCDF -> Zarr daemon")
    logger.info(
        "data_root=%s output_dir=%s expected_members=%s max_workers=%s", config.data_root, config.output_dir, config.expected_members, config.max_workers
    )
    EnsembleProcessor(config).run()


if __name__ == "__main__":
    main()
