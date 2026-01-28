#!/usr/bin/env python3
"""
Check the min/max/mean values of a Zarr dataset along a specified axis.

This script loads a Zarr dataset and analyzes it by iterating over a specified
axis, printing statistics (min, max, mean) for each slice.
It uses Dask for parallel processing to speed up the analysis.


Usage:
    python check_zarr_range.py --input /path/to/zarr/dataset --var variable_name --axis time
"""

import argparse
import numpy as np
import xarray as xr
import sys
from tqdm import tqdm
import os
import dask
import concurrent.futures
import multiprocessing


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check min/max/mean values of a Zarr dataset along a specified axis.")
    parser.add_argument("--input", "-i", required=True, help="Path to Zarr dataset")
    parser.add_argument("--var", "-v", default="data", help="Variable name to analyze")
    parser.add_argument(
        "--axis",
        "-a",
        default="level",
        help="Axis to iterate over (e.g., time, level, lat, lon)",
    )
    parser.add_argument("--output", "-o", help="Output file to save results (default: print to stdout)")
    parser.add_argument("--slice", "-s", help="Optional slice in format 'start:end:step' for the axis")
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of Dask workers (default: number of CPU cores)",
    )
    parser.add_argument(
        "--scheduler",
        default="processes",
        choices=["processes", "threads", "distributed", "sequential"],
        help="Dask scheduler to use (default: processes)",
    )
    return parser.parse_args()


def load_dataset(input_path, var_name):
    """
    Load a Zarr dataset and validate the requested variable exists.

    Args:
        input_path: Path to Zarr dataset
        var_name: Name of the variable to analyze
        chunks: Chunk size for Dask array

    Returns:
        xarray.Dataset or xarray.DataArray: The loaded dataset or variable
    """
    try:
        if os.path.exists(input_path):
            # Open with Dask chunking for parallel computation
            ds = xr.open_zarr(input_path, consolidated=True)
        else:
            print(f"Error: Input path '{input_path}' does not exist")
            sys.exit(1)
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        sys.exit(1)

    # Validate variable exists
    if var_name not in ds:
        print(f"Error: Variable '{var_name}' not found in dataset")
        print(f"Available variables: {list(ds.data_vars)}")
        sys.exit(1)

    return ds[var_name]


def analyze_single_slice(data_slice, axis_values):
    """
    Analyze a single slice of data and return statistics.

    Args:
        data_slice: The data slice to analyze
        axis_val: The value of the axis for this slice

    Returns:
        dict: Dictionary with statistics for this slice
    """
    # Calculate statistics (using Dask's compute to calculate all at once)
    # data_slice shape is (76, lat, lon), calculate min and max along the first dimension with dask
    min_val, max_val = dask.compute(
        data_slice.min(dim=("lat", "lon"), skipna=True),
        data_slice.max(dim=("lat", "lon"), skipna=True),
    )

    result = {
        "axis_val": axis_values,
        "min": np.array(min_val.values),
        "max": np.array(max_val.values),
    }

    # Calculate nan statistics
    nan_count = np.isnan(data_slice.values).sum()
    inf_count = np.isinf(data_slice.values).sum()
    result["nan_count"] = int(nan_count)
    result["inf_count"] = int(inf_count)

    if nan_count > 0 or inf_count > 0:
        print(f"data {data_slice.time.values} has NaN/Inf values")

    if np.min(min_val) < -100.0 or np.max(max_val) > 100.0:
        print(f"data {data_slice.time.values} has out-of-range values: min={np.min(min_val.values)}, max={np.max(max_val.values)}")

    return result


def analyze_data(data, axis, slice_arg=None, scheduler="processes", num_workers=None):
    """
    Analyze data along specified axis, printing statistics for each slice using Dask.

    Args:
        data: xarray.DataArray to analyze
        axis: Axis name to iterate over
        slice_arg: Optional slice to apply to the axis
        scheduler: Dask scheduler to use ("processes", "threads" or "distributed")
        num_workers: Number of workers to use
    """
    if axis not in data.dims:
        print(f"Error: Axis '{axis}' not found in data dimensions")
        print(f"Available dimensions: {data.dims}")
        sys.exit(1)

    # Get all values along the specified axis
    axis_values = data[axis].values

    # Apply slice if provided
    if slice_arg:
        # time split by :
        slice_time = slice_arg.split(":")
        data = data.sel(time=slice(*slice_time))

    # Output header
    print(f"\nAnalyzing variable '{data.name}' along axis '{axis}'")
    print(f"Data shape: {data.shape}")
    print(f"Data dimensions: {data.dims}")
    print(f"Number of {axis} values: {len(axis_values)}")
    print(f"Using Dask with {scheduler} scheduler")
    if num_workers:
        print(f"Number of workers: {num_workers}")
    else:
        print("Number of workers: auto (using all available cores)")

    print("\n{:<20} | {:<15} | {:<15}".format(axis, "Min", "Max"))
    print("-" * 57)

    # Set up the parallel computation method based on scheduler
    # because data is chunked by "time", we use scheduler to parallelize the analysis
    chunk_axis = "time"
    results = {}

    def merge_result(results, result):
        """Merge results from multiple slices."""
        if "min" in results:
            results["min"] = np.minimum(results["min"], result["min"])
            results["max"] = np.maximum(results["max"], result["max"])
            results["nan_count"] += result.get("nan_count", 0)
            results["inf_count"] += result.get("inf_count", 0)
        else:
            results.update(result)
        return

    if scheduler == "distributed":
        from dask.distributed import Client

        # Use a distributed client
        n_workers = num_workers if num_workers else multiprocessing.cpu_count()
        client = Client(n_workers=n_workers)
        print(f"Dask dashboard at: {client.dashboard_link}")

        # Create task list
        futures = []
        for i, axis_val in enumerate(data[chunk_axis].values):
            selection = {chunk_axis: i}
            slice_data = data.isel(**selection)
            futures.append(client.submit(analyze_single_slice, slice_data, axis_val))

        # Compute and get results
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc=f"Processing {chunk_axis}",
            total=len(futures),
        ):
            merge_result(results, future.result())
    elif scheduler == "sequential":
        # Process slices sequentially
        for i, axis_val in enumerate(tqdm(data[chunk_axis].values, desc=f"Processing {chunk_axis}")):
            selection = {chunk_axis: i}
            slice_data = data.isel(**selection)
            result = analyze_single_slice(slice_data, axis_val)
            merge_result(results, result)
    else:
        # Use processes or threads through concurrent.futures
        executor_class = concurrent.futures.ProcessPoolExecutor if scheduler == "processes" else concurrent.futures.ThreadPoolExecutor
        max_workers = num_workers if num_workers else multiprocessing.cpu_count()

        # Create and execute tasks
        with executor_class(max_workers=max_workers) as executor:
            futures = []
            for i, _ in enumerate(data[chunk_axis].values):
                selection = {chunk_axis: i}
                slice_data = data.isel(**selection)
                futures.append(executor.submit(analyze_single_slice, slice_data, axis_values))

            # Get results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                desc=f"Processing {chunk_axis}",
                total=len(futures),
            ):
                merge_result(results, future.result())

    # Print results
    for idx, val in enumerate(results["axis_val"]):
        print("{:<20} | {:<15.6g} | {:<15.6g}".format(str(val), results["min"][idx], results["max"][idx]))

    if "nan_count" in results:
        print(f"  NaN count: {results['nan_count']}")
    if "inf_count" in results:
        print(f"  Inf count: {results['inf_count']}")

    # Close client if using distributed
    if scheduler == "distributed" and "client" in locals():
        client.close()


def main():
    """Main function."""
    args = parse_args()

    # Set number of workers
    num_workers = args.workers if args.workers else multiprocessing.cpu_count() // 4

    print(f"Loading Zarr dataset from: {args.input}")
    print(f"Using Dask parallel processing with {args.scheduler} scheduler")
    print(f"Number of workers: {num_workers}")

    # Load data with Dask chunking
    data = load_dataset(args.input, args.var)
    # Redirect output to file if specified
    if args.output:
        orig_stdout = sys.stdout
        f = open(args.output, "w")
        sys.stdout = f

    # Analyze data with parallel processing
    analyze_data(data, args.axis, args.slice, scheduler=args.scheduler, num_workers=num_workers)

    # Restore stdout if needed
    if args.output:
        sys.stdout = orig_stdout
        f.close()
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
