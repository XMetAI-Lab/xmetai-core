#!/usr/bin/env python3
"""
Extract data from a Zarr dataset, process it, and save as NetCDF.

This script extracts data based on a specified date range and levels,
applies inverse normalization, handles special processing for total precipitation (tp),
and saves the result as a NetCDF file.

Usage:
    python extract_nc_from_zarr.py --zarr_path /path/to/zarr --save_path output.nc
                                    --start_date 2022-01-01 --time_interval 7
                                    --levels t2m,tp,msl
"""

import argparse
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract data from Zarr dataset, process it, and save as NetCDF."
    )
    parser.add_argument(
        "--zarr_path", "-z", default="/home/douzsh/workspace/data/s2s.1950-2024.c76/",
        help="Path to input Zarr dataset"
    )
    parser.add_argument(
        "--save_path", "-s", default="~/workspace/tmp/",
        help="Path to save the output NetCDF file"
    )
    parser.add_argument(
        "--start_date", "-t", required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--time_interval", '-i', type=int, default=42,
        help="Number of days to extract after start date"
    )
    parser.add_argument(
        "--levels", "-l", default="z500,t2m,tp,msl",
        help="Comma-separated list of level names to extract (e.g., t2m,tp,msl)"
    )
    return parser.parse_args()


def load_zarr_data(zarr_path, levels_list, stats):
    """
    Load Zarr dataset and extract specified levels.

    Args:
        zarr_path: Path to Zarr dataset
        levels_list: List of level names to extract
        stats: Statistics dataset containing mean and std

    Returns:
        xarray.Dataset: Dataset with requested levels
    """
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        sys.exit(1)

    # Check if dataset has the expected structure
    if 'data' not in ds.data_vars:
        print("Warning: Expected 'data' variable not found in the dataset.")
        print(f"Available variables: {list(ds.data_vars)}")
        sys.exit(1)
    # fix coord name, if level not exist use channel
    if "level" in ds.coords:
        coord_name = "level"
    elif "channel" in ds.coords:
        ds = ds.rename({"channel": "level"})
        coord_name = "level"
    else:
        print("Error: The dataset does not contain a 'level' or 'channel' coordinate.")
        sys.exit(1)

    # Check if all requested levels exist in the dataset
    level_coords = ds.coords.get(coord_name, None)
    if level_coords is None:
        print(f"Error: The dataset does not contain a '{coord_name}' coordinate.")
        sys.exit(1)

    stats["mean"] = stats["mean"].assign_coords(**{coord_name: level_coords.values})
    stats["std"] = stats["std"].assign_coords(**{coord_name: level_coords.values})
    # Find indices of the requested levels
    level_values = level_coords.values.tolist()
    level_indices = []
    existing_levels = []

    for lvl in levels_list:
        try:
            idx = level_values.index(lvl)
            level_indices.append(idx)
            existing_levels.append(lvl)
        except ValueError:
            print(f"Warning: Level '{lvl}' not found in the dataset")

    if not existing_levels:
        print("Error: None of the requested levels were found in the dataset.")
        sys.exit(1)

    # Extract the requested levels from the data variable
    print(f"Extracting levels: {existing_levels}")
    extracted_data = ds.sel(level=existing_levels)

    return extracted_data


def load_stats(zarr_path):
    """
    Load normalization statistics.

    Args:
        zarr_path: Path to directory containing statistics files

    Returns:
        dict: Dictionary containing mean and std arrays for normalization
    """
    try:
        mean_path = os.path.join(zarr_path, "mean.nc")
        std_path = os.path.join(zarr_path, "std.nc")

        if os.path.exists(mean_path) and os.path.exists(std_path):
            print(f"Loaded statistics from {mean_path} and {std_path}")
            mean_ds = xr.open_dataarray(mean_path)
            std_ds = xr.open_dataarray(std_path)
            # update channel coords to level
            mean_ds = mean_ds.rename({'channel': 'level'})
            std_ds = std_ds.rename({'channel': 'level'})
            stats = {
                'mean': mean_ds,
                'std': std_ds
            }
        elif os.path.exists(os.path.join(zarr_path, "mean.npy")) and os.path.exists(os.path.join(zarr_path, "std.npy")):
            print(f"Loaded statistics from {zarr_path}")
            mean = np.load(os.path.join(zarr_path, "mean.npy")).reshape((-1, 1, 1))
            std = np.load(os.path.join(zarr_path, "std.npy")).reshape((-1, 1, 1))
            stats = {
                'mean': xr.DataArray(mean, dims=['level', "lat", "lon"]),
                'std': xr.DataArray(std, dims=['level', "lat", "lon"])
            }
        else:
            print(f"Error: Statistics files not found at {mean_path} or {std_path}")
            sys.exit(1)

        print(f"Loaded statistics with shapes - Mean: {stats['mean'].shape}, Std: {stats['std'].shape}")
        return stats
    except Exception as e:
        print(f"Error loading statistics files: {e}")
        sys.exit(1)


def extract_time_slice(ds, start_date, time_interval):
    """
    Extract time slice from dataset.

    Args:
        ds: Input dataset
        start_date: Start date as string in YYYY-MM-DD format
        time_interval: Number of days to extract

    Returns:
        xarray.Dataset: Dataset with filtered time range
    """
    try:
        # Convert start_date to pandas timestamp
        if len(start_date) == 8:
            # yyyymmdd
            start_time = pd.to_datetime(start_date, format="%Y%m%d")
        elif len(start_date) == 10:
            # yyyymmddhh
            start_time = pd.to_datetime(start_date, format="%Y%m%d%H")
        elif len(start_date) == 11:
            # yyyymmdd-hh
            start_time = pd.to_datetime(start_date, format="%Y%m%d-%H")
        else:
            print("Error: start_date format is incorrect. Use YYYYMMDD or YYYYMMDDHH or YYYYMMDD-HH")
            sys.exit(1)
        end_time = start_time + pd.Timedelta(days=time_interval)

        # Filter dataset by time
        time_slice = (ds.time >= start_time) & (ds.time < end_time)
        if not time_slice.any():
            print(f"Error: No data found for the specified time range: {start_date} to {end_time}")
            print(f"Available time range: {ds.time.min().values} to {ds.time.max().values}")
            sys.exit(1)

        return ds.sel(time=time_slice)
    except Exception as e:
        print(f"Error extracting time slice: {e}")
        sys.exit(1)


def inverse_normalize(ds, stats, levels_list):
    """
    Apply inverse normalization to dataset.

    Args:
        ds: Dataset to denormalize
        stats: Dict containing mean and std arrays
        levels_list: List of level names that were extracted

    Returns:
        xarray.Dataset: Denormalized dataset
    """
    result = ds.copy()

    # Get level indices for the selected levels
    level_coords = ds.coords.get('level', None)
    if level_coords is None:
        print("Error: Cannot apply inverse normalization, no level coordinate found")
        return result

    level_values = level_coords.values.tolist()

    # Apply inverse normalization
    if 'data' in ds.data_vars:
        print("Applying inverse normalization to data")

        # Create a new array for normalized data
        denorm_data = result.data.copy(deep=True)

        # Loop through each level in the extracted dataset
        for i, level in enumerate(level_values):
            # Find the corresponding index in the original level list
            try:
                level_idx = ds.level.values.tolist().index(level)
                mean_val = stats['mean'].sel(level=level).values
                std_val = stats['std'].sel(level=level).values

                print(f"Applying inverse normalization to level '{level}:{level_idx}' with mean={mean_val}, std={std_val}")

                # Apply inverse normalization: (normalized * std) + mean
                denorm_data.loc[{'level': level}] = (result.data.sel(level=level) * std_val) + mean_val
            except (ValueError, KeyError, IndexError) as e:
                print(f"Warning: Error normalizing level '{level}': {e}")

        # Assign denormalized data back to result
        result['data'] = denorm_data
    else:
        # Handle case where each level might be its own variable
        for var in result.data_vars:
            if var in levels_list:
                try:
                    level_idx = levels_list.index(var)
                    if level_idx < len(stats['mean']) and level_idx < len(stats['std']):
                        mean_val = stats['mean'][level_idx].item()
                        std_val = stats['std'][level_idx].item()
                        print(f"Applying inverse normalization to {var} with mean={mean_val:.6f}, std={std_val:.6f}")
                        result[var] = (result[var] * std_val) + mean_val
                    else:
                        print(f"Warning: Index {level_idx} out of range for stats arrays")
                except (IndexError, ValueError) as e:
                    print(f"Warning: Error normalizing {var}: {e}")
            else:
                print(f"Warning: No normalization stats found for {var}, skipping denormalization")

    return result


def process_tp_data(ds, levels_list):
    """
    Process total precipitation (tp) data by applying exp() function.

    Args:
        ds: Dataset containing tp variable
        levels_list: List of level names that were extracted

    Returns:
        xarray.Dataset: Dataset with processed tp
    """
    result = ds.copy()

    # Check if 'tp' is in the levels list
    if 'tp' in levels_list:
        # If data is structured as a single data variable with level dimension
        if 'data' in ds.data_vars and 'level' in ds.dims:
            level_values = ds.level.values.tolist()
            if 'tp' in level_values:
                print("Applying exp() to tp level in data variable (reverting log normalization)")
                # Get the tp index in the level dimension
                tp_index = level_values.index('tp')

                # Create a new array for processed data
                processed_data = result.data.copy(deep=True)

                # Apply exp() to the tp level
                processed_data.loc[{'level': 'tp'}] = np.exp(result.data.sel(level='tp'))

                # Assign processed data back to result
                result['data'] = processed_data
        # If tp is its own variable
        elif 'tp' in ds.data_vars:
            print("Applying exp() to tp variable (reverting log normalization)")
            result['tp'] = np.exp(result['tp'])

    return result


def convert_to_float32(ds):
    """
    Convert all data variables to float32.

    Args:
        ds: Input dataset

    Returns:
        xarray.Dataset: Dataset with float32 data type
    """
    result = ds.copy()
    for var in result.data_vars:
        result[var] = result[var].astype(np.float32)
    return result


def prepare_netcdf_compatible_dataset(ds):
    """
    Prepare dataset to be NetCDF compatible by converting string coordinates to character arrays.

    Args:
        ds: Input dataset with possibly string coordinates

    Returns:
        xarray.Dataset: Dataset with NetCDF compatible data types
    """
    ds_compat = ds.copy()

    # Check if level coordinate exists and contains strings
    if 'level' in ds_compat.coords and np.issubdtype(ds_compat.level.dtype, np.dtype('O')):
        level_values = ds_compat.level.values

        # Convert string levels to fixed-length character arrays
        try:
            # Try to convert to numeric first if possible
            numeric_levels = pd.to_numeric(level_values)
            ds_compat.coords['level'] = ('level', numeric_levels)
        except (ValueError, TypeError):
            # If conversion to numeric fails, use character arrays
            max_len = max(len(str(x)) for x in level_values)
            char_array = np.array([str(x).ljust(max_len) for x in level_values], dtype=f'S{max_len}')
            ds_compat.coords['level'] = ('level', char_array)

    return ds_compat


def main():
    """Main function."""
    args = parse_args()

    # Parse levels
    levels_list = args.levels.split(',')

    print(f"Extracting data from: {args.zarr_path}")
    print(f"Start date: {args.start_date}, Time interval: {args.time_interval} days")
    print(f"Levels: {levels_list}")

    # Load normalization statistics
    print("Loading normalization statistics...")
    stats = load_stats(args.zarr_path)

    # Load Zarr dataset and extract specified levels
    print("Loading Zarr dataset...")
    ds = load_zarr_data(args.zarr_path, levels_list, stats)

    # Extract time slice
    print("Extracting time slice...")
    ds_time_slice = extract_time_slice(ds, args.start_date, args.time_interval)

    # Convert to float32
    print("Converting to float32...")
    ds_float32 = convert_to_float32(ds_time_slice)

    # Apply inverse normalization
    print("Applying inverse normalization...")
    ds_denormalized = inverse_normalize(ds_float32, stats, levels_list)

    # Process tp data if present
    if 'tp' in levels_list:
        print("Processing total precipitation (tp) data...")
        ds_processed = process_tp_data(ds_denormalized, levels_list)
    else:
        ds_processed = ds_denormalized

    # Prepare dataset for NetCDF compatibility
    print("Preparing NetCDF compatible dataset...")
    ds_netcdf_compatible = prepare_netcdf_compatible_dataset(ds_processed)

    # Save as NetCDF
    args.save_path = os.path.join(args.save_path, f"data_{args.start_date.replace('-', '')}.nc")
    print(f"Saving to NetCDF: {args.save_path}")
    try:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save with compression
        encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_netcdf_compatible.data_vars}
        ds_netcdf_compatible.to_netcdf(
            args.save_path,
            encoding=encoding,
            format='NETCDF4'
        )
        print("Export completed successfully")
    except Exception as e:
        print(f"Error saving NetCDF file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
