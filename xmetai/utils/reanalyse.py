import calendar
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

__all__ = [
    "LEVELS_13",
    "LEVELS_37",
    "update_dims",
    "update_coords",
    "compute_level_weight",
    "normalize",
    "denormalize",
    "inv_normalize",
    "validate_range",
    "remove_invalid",
    "compute_statistics",
    "land_to_nan",
]

KELVIN = -273.15
GRAVITY = 9.80665
EARTH_RADIUS_M = 1000 * (6357 + 6378) / 2

# LEVELS_0 = np.array([0], dtype=np.int32)
LEVELS_6 = [100, 200, 500, 700, 850, 925]
LEVELS_13 = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
LEVELS_18 = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
LEVELS_20 = [1, 2, 3, 5, 10, 20, 30] + LEVELS_13
LEVELS_37 = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]  # fmt: skip
LEVELS_30 = [1, 5, 10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]  # fmt: skip


SHORT_NAME_MAPPING = dict(
    prmsl="msl",
    tp24hr="tp",
    u10="u10m",
    v10="v10m",
    ws10="ws10m",
    u100="u100m",
    v100="v100m",
    ws100="ws100m",
)


UNIT_SCALE = dict(
    ciwc=1000,
    clwc=1000,
    crwc=1000,
    cswc=1000,
    q=1000,
    q2m=1000,
    tp=1000,
    gh=9.80665,
    ttr=1 / 3600,
    ssr=1 / 3600,
    ssrd=1 / 3600,
    fdir=1 / 3600,
)

STATIONS_NORM = dict(tp=(0, 100), ws=(0, 35), gs=(0, 40), cr=(0, 70), ld=(0, 100))

HUADONG_AREA = dict(
    lat_range=np.arange(29.01, 36.69, 0.01, dtype=np.float32)[::-1],
    lon_range=np.arange(114.67, 123.62, 0.01, dtype=np.float32),
)

ZHEJIANG_AREA = dict(
    lat_range=np.linspace(26.5, 32.0, 551, dtype=np.float32)[::-1],
    lon_range=np.linspace(117, 123.50, 651, dtype=np.float32),
)


def select_data(names, vars):
    """Filter name pairs when specific variables are requested."""
    return [(ln, sn) for ln, sn in names if not vars or sn in vars]


def compute_level_weight():
    levels = np.array(LEVELS_37)
    weights = levels / levels.sum()
    inds = [LEVELS_37.index(lvl) for lvl in LEVELS_13]
    return weights[inds]


def chunk(ds, time=1, **kwargs):
    if isinstance(ds, xr.DataArray):
        dims = {k: v for k, v in zip(ds.dims, ds.shape)}
    else:
        dims = {k: v for k, v in ds.sizes.items()}

    if "time" in dims:
        dims["time"] = time

    for k, v in kwargs.items():
        if k in dims:
            dims[k] = v

    ds = ds.chunk(dims)
    return ds


def update_dims(ds):
    dim_mapping = {
        "valid_time": "time",
        "dayofyear": "doy",
        "lead_time": "step",
        "pressure_level": "level",
        "isobaricInhPa": "level",
        "heightabovefround": "level",
        "prediction_timedelta": "step",
        "latitude": "lat",
        "longitude": "lon",
        "numrer": "member",
        "plev": "level",
        "lev": "level",
    }
    dim_mapping.update(SHORT_NAME_MAPPING)

    if isinstance(ds, xr.DataArray):
        dims = {k: dim_mapping.get(k, k) for k in ds.dims}
    else:
        dims = {k: dim_mapping.get(k, k) for k in ds.sizes}
        dims.update({k: dim_mapping.get(k, k) for k in ds})

    ds = ds.rename(dims)
    return ds


def update_coords(ds):
    ds = ds.reset_coords(drop=True)

    if "lat" in ds.dims:
        if ds.lat.data[0] == -90 and ds.lat.data[-1] == 90:
            ds = ds.reindex(lat=ds.lat[::-1])

    if "lon" in ds.dims:
        if ds.lon.data[0] == -180 and ds.lon.data[-1] == 180:
            ds = ds.assign_coords(lon=((360 + (ds.lon % 360)) % 360))
            ds = ds.roll(lon=int(len(ds["lon"]) / 2), roll_coords=True)

    if ("level" not in ds.dims) and ("channel" not in ds.dims):
        ds = ds.expand_dims({"level": [1]}, axis=1)

    return ds


def update_units(v):
    if isinstance(v, xr.Dataset):
        return v.map(update_units)

    if v.name in UNIT_SCALE:
        scale = UNIT_SCALE[v.name]
        v *= scale
        print(f"scaling {v.name} by {scale:.6f}")

    if is_water(v.name):
        v = np.log1p(v)
        print(f"apply log1p to {v.name}")

    return v


def update_coords_zarr(data_path: str, coord_name: str, coord_values: np.ndarray) -> None:
    # Open the Zarr dataset in read/write mode
    ds = zarr.open_group(data_path, mode="r+")

    # Check if coordinate exists
    if coord_name not in ds:
        raise KeyError(f"Coordinate '{coord_name}' not found in dataset")

    # Verify shape compatibility
    coord_arr = ds[coord_name]  # type: ignore[union-attr]
    if coord_arr.shape != coord_values.shape:  # type: ignore[union-attr]
        raise ValueError(f"Shape mismatch: existing {coord_arr.shape} vs new {coord_values.shape}")  # type: ignore[union-attr]

    # Update the coordinate values in-place
    coord_arr[:] = coord_values  # type: ignore[index]


def load_grib(file_path, decode_timedelta=False):
    datasets = []

    # Try to open GRIB2 messages
    try:
        ds_grib2 = xr.open_dataset(
            file_path,
            engine="cfgrib",
            decode_timedelta=decode_timedelta,
            backend_kwargs={"filter_by_keys": {"edition": 2}},
        )
        datasets.append(ds_grib2)
    except Exception as e:
        print(f"GRIB2 messages not found or could not be opened: {e}")

    # Try to open GRIB1 messages
    try:
        ds_grib1 = xr.open_dataset(
            file_path,
            engine="cfgrib",
            decode_timedelta=decode_timedelta,
            backend_kwargs={"filter_by_keys": {"edition": 1}},
        )
        datasets.append(ds_grib1)
    except Exception as e:
        print(f"GRIB1 messages not found or could not be opened: {e}")

    if len(datasets) == 0:
        raise ValueError("No GRIB messages could be opened.")

    # Combine datasets if both editions are present
    if len(datasets) > 1:
        ds = xr.merge(datasets)
    else:
        ds = datasets[0]

    return ds


def level_to_channel(ds, short_name):
    if "channel" in ds.dims:
        ds = ds.rename({short_name: "data"})
        return ds

    # if "level" not in ds.dims:
    #     ds = ds.expand_dims({'level': [1]}, axis=min(1, len(ds.sizes)))

    if len(ds.level) == 1:
        channel = [SHORT_NAME_MAPPING.get(short_name, short_name)]
    else:
        channel = [f"{short_name}{lvl}" for lvl in ds.level.data]
    ds.attrs = {}
    ds = ds.rename({short_name: "data", "level": "channel"})
    ds = ds.assign_coords(channel=channel)
    return ds


def channel_to_level(ds, short_name):
    if "channel" in ds.dims:
        levels = [int(c.replace(short_name, "")) for c in ds.channel.data]
        ds = ds.rename({"channel": "level"}).assign_coords(level=levels)
    else:
        ds = ds.expand_dims({"level": [1]}, axis=1)
    return ds


def nearest_index(ds, coords, dim="depth"):
    _coords = ds[dim].values.tolist()
    inds = [_coords.index(min(_coords, key=lambda x: abs(x - coord))) for coord in coords]
    return inds


def is_pressure(name):
    return name in [
        "omega",
        "h2o",
        "o3",
        "gh",
        "z",
        "s",
        "t",
        "u",
        "v",
        "q",
        "cc",
        "ciwc",
        "clwc",
        "crwc",
        "cswc",
    ]


def is_accum(name):
    return name in ["ssr", "ssrd", "fdir", "ttr", "tp"]


def is_ocean(name):
    return name in ["sst", "mdts", "mdww", "mpts", "mpww", "shts", "shww"]


def is_water(name):
    return name in ["ciwc", "clwc", "crwc", "cswc", "tp"]


def check_grib(ds):
    for v in ds:
        print(ds.name)
    names = {v.shortName: v.name for v in ds}
    print(f"names: \n{names}")
    for v in ds.select(shortName=list(names.keys())):
        msg = f"name: {v.name}, dataDate: { v.dataDate}, date: {v.date}, time: {v.time}, stepType: {v.stepType}"
        msg += f"\nstartStep: {v.startStep}, endStep: {v.endStep}, level: {v.level}"
        print(msg)


def is_complete_year(ds, freq="hourly"):
    assert freq in ["hourly", "six_hourly", "daily"]
    hour_interval = dict(hourly=24, six_hourly=4, daily=1)[freq]
    acutal_num = ds.sizes["time"]
    year = pd.to_datetime(ds["time"].values[0]).year
    if calendar.isleap(year):
        expected_num = hour_interval * 366
    else:
        expected_num = hour_interval * 365
    return acutal_num == expected_num


def hasnan(ds, num=10):
    if isinstance(ds, xr.Dataset):
        res = 0
        for k in ds:
            res += hasnan(ds[k])
        return res

    res = 0
    for _, x in ds.groupby("time.dayofyear"):
        res += np.isnan(x.values).sum()
        break

    tid = np.random.choice(np.arange(0, ds.shape[0]), num)
    tid = np.append(tid, [0, ds.shape[0] - 1])
    x = ds.isel(time=tid)
    res += np.isnan(x.values).sum()
    return res > 0


def _safe_std(std):
    if torch.is_tensor(std):
        return torch.where(std == 0, torch.ones_like(std), std)
    return np.where(std == 0, 1, std)


def normalize(ds, mean, std, log_names=["tp"]):
    if not hasattr(ds, "sel"):
        return (ds - mean) / _safe_std(std)

    for channel in log_names:
        print(f"applying log transformation to {channel}")
        v = ds.sel(channel=channel)
        v = np.log1p(v.clip(0, 1000))
        ds.loc[dict(channel=channel)] = v
    ds = (ds - mean) / std
    return ds


def denormalize(ds, mean, std):
    return ds * _safe_std(std) + mean


def validate_range(data):
    if torch.is_tensor(data):
        if not torch.isfinite(data).all():
            raise ValueError("Data contains NaN or Inf values.")
        return
    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or Inf values.")


def remove_invalid(data, fill_value=0.0):
    if torch.is_tensor(data):
        return torch.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return np.nan_to_num(data, nan=fill_value, posinf=fill_value, neginf=fill_value)


def compute_statistics(data):
    if torch.is_tensor(data):
        return data.mean(dim=(0, 1, 3, 4), keepdim=True), data.std(dim=(0, 1, 3, 4), keepdim=True)
    return np.mean(data, axis=(0, 1, 3, 4), keepdims=True), np.std(data, axis=(0, 1, 3, 4), keepdims=True)


def inv_normalize(ds, mean, std, dim="level", log_names=["tp"]):
    ds = ds.astype(np.float32)
    ds = ds * std + mean

    for channel in log_names:
        print(f"applying log transformation to {channel}")

    if dim == "level":
        v = ds.sel(level="tp")
        v = np.exp(v.clip(0, 7)) - 1
        new_ds = ds.isel(level=np.arange(len(ds[dim]) - 1))
    elif dim == "channel":
        v = ds.sel(channel="tp")
        v = np.exp(v.clip(0, 7)) - 1
        new_ds = ds.isel(channel=np.arange(len(ds[dim]) - 1))

        # v = np.exp(v.clip(0, 7)) - 1
        # ds.loc[dict(channel="tp")] = v

    new_ds = xr.concat([new_ds, v], dim=dim)
    new_ds = chunk(new_ds)
    return new_ds


def land_to_nan(input, mask, dim="channel", ocean_names=["sst"]):
    chans = input[dim].values.tolist()
    for k in ocean_names:
        v = input.sel({dim: k})
        v = v.where(mask)
        idx = chans.index(k)
        input.data[:, idx] = v.data
    return input


def get_trend(ds):
    ds.coords["doy"] = ("time", get_doy(ds, no_leap_year=True))
    dg = ds.groupby("doy")
    trends = []
    groups = []
    doys = []
    for i, v in dg:
        doys.append(i)
        groups.append(len(v.time))
        p = v.polyfit(dim="time", deg=1)
        trends.append(p.polyfit_coefficients)
    groups = np.array(groups[:365], dtype=np.int32)
    print(f"groups: \n {groups}")
    trends = xr.concat(trends, "doy")
    trends = trends.assign_coords(doy=doys)
    return trends


def detrend_single(ds, trend, dim="time"):
    ds = chunk(ds)
    coeffs = trend[ds.name]
    coeffs = sel_like(coeffs, ds, no_leap_year=True)
    coeffs = chunk(coeffs, time=-1)
    fit = xr.polyval(ds[dim], coeffs)
    return ds - fit


def align_coords(ds, hflip=True):
    if hflip:
        ds = ds.reindex(lat=ds.lat[::-1])
    ds = ds.assign_coords(lon=((360 + (ds.lon % 360)) % 360))
    ds = ds.roll(lon=int(len(ds["lon"]) / 2), roll_coords=True)
    return ds


def print_grib(ds):
    for v in ds:
        num_member = v.numberOfForecastsInEnsemble
        step_type = v.stepType
        time_str = f"{v.dataDate}"
        init_time = pd.to_datetime(time_str)
        curr_step = v.endStep
        mid = v.perturbationNumber
        if mid == 1:
            print(f"init_time: {init_time}, mid: {mid:02d}/{num_member}, step: {step_type} {v.startStep} ~ {curr_step:03d}")


def handle_dataset(func):
    """Decorator that handles xarray Datasets by applying the function to each DataArray."""

    def wrapper(ds, *args, **kwargs):
        if isinstance(ds, xr.Dataset):
            results = []
            for v in ds:
                result = func(ds[v], *args, **kwargs)
                results.append(result)
            return results
        return func(ds, *args, **kwargs)

    return wrapper


@handle_dataset
def print_dataarray(ds, names=[], num_sample=10, skipna=False):
    if "time" in ds.dims:
        times = np.random.choice(ds.time, min(num_sample, len(ds.time)))
        v = ds.sel(time=times)
    else:
        v = ds

    msg = f"{v.name.upper()}: \nshape: \033[94m{v.shape}\033[0m"

    if "time" in ds.dims:
        start_date = ds.time[0].dt.strftime("%Y%m%d%H%M").item()
        end_date = ds.time[-1].dt.strftime("%Y%m%d%H%M").item()
        msg += f", time: \033[94m{len(ds.time)} = ({start_date} ~ {end_date})\033[0m"

    if "lat" in ds.dims and "lon" in ds.dims:
        lat = ds.lat.values
        lon = ds.lon.values
        msg += f", latlon: \033[94m({lat[0]:.4f} ~ {lat[-1]:.4f}) x ({lon[0]:.4f} ~ {lon[-1]:.4f})\033[0m"

    if "level" in v.dims:
        if len(names) > 0:
            v = v.sel(level=np.intersect1d(names, v.level))
        for lvl in v.level.data:
            x = v.sel(level=lvl).values
            if skipna:
                x = x[~np.isnan(x)]
            msg += f"\nlevel: {lvl}, value: \033[91m{x.min():.4f} ~ {x.max():.4f}\033[0m"
    elif "channel" in v.dims:
        if len(names) > 0:
            v = v.sel(channel=np.intersect1d(names, v.channel))
        for ch in v.channel.data:
            x = v.sel(channel=ch).values
            if skipna:
                x = x[~np.isnan(x)]
            msg += f"\nchannel: {ch}, value: \033[91m{x.min():.4f} ~ {x.max():.4f}\033[0m"
    else:
        x = v.values
        if skipna:
            x = x[~np.isnan(x)]
        msg += f", value: \033[91m{x.min():.4f} ~ {x.max():.4f}\033[0m"

    print(msg)


def resize_dataarray(ds, lat=None, lon=None, resolution=0.25, fill_value=False):
    src_lat = ds.lat.values
    src_lon = ds.lon.values

    if lat is None:
        dlat = np.sign(src_lat[1] - src_lat[0]) * resolution
        lat = np.arange(src_lat[0], src_lat[-1] + dlat, dlat)

    if lon is None:
        dlon = np.sign(src_lon[1] - src_lon[0]) * resolution
        lon = np.arange(src_lon[0], min(360, src_lon[-1] + dlon), dlon)

    # from IPython import embed; embed()
    if len(src_lat) == len(lat) and len(src_lon) == len(lon):
        return ds

    print(f"\033[0;32mResize from {len(src_lat)} x {len(src_lon)} to {len(lat)} x {len(lon)}\033[0m")

    if fill_value:
        ds = ds.interp(lat=lat, lon=lon, kwargs={"fill_value": "extrapolate"})
    else:
        ds = ds.interp(lat=lat, lon=lon)

    return ds


def spatial_interp(ds, scaling_factor):
    lat = ds.lat.values
    lon = ds.lon.values
    new_lat_size = max(1, int(len(lat) / scaling_factor))
    new_lon_size = max(1, int(len(lon) / scaling_factor))
    new_lat = np.linspace(lat[0], lat[-1], new_lat_size)
    new_lon = np.linspace(lon[0], lon[-1], new_lon_size)
    ds_interp = ds.interp(lat=new_lat, lon=new_lon, method="linear")
    return ds_interp


def cumsum_to_accum(ds, start_minute=10):
    diff = ds.diff(dim="time", n=1)
    mask = (ds.time.dt.minute == start_minute).data
    accum = ds.sel(time=ds.time[mask])
    diff = diff.sel(time=ds.time[~mask])
    accum = xr.concat([accum, diff], "time")
    accum = accum.sortby("time")
    return accum


def filter_complete_hours(ds, time_resolution="6min"):
    times = pd.DatetimeIndex(ds.time.values)
    time_interval = pd.Timedelta(time_resolution)

    # Check if times follow a regular interval by comparing fractional and integer divisions
    time_diffs = (times - times[0]) / time_interval
    valid_times = times[time_diffs.astype(int) == time_diffs]

    if len(valid_times) == 0:
        print(f"No valid {time_resolution} interval data found. Returning original dataset.")
        return ds

    # Select only the valid time points
    ds_valid = ds.sel(time=valid_times)

    # Calculate expected number of points per hour
    points_per_hour = pd.Timedelta("1h") / time_interval

    # Count timestamps per hour and find complete hours
    timestamps_per_hour = ds_valid.time.groupby(ds_valid.time.dt.floor("h")).count()
    complete_hours = timestamps_per_hour[timestamps_per_hour == points_per_hour].floor

    if len(complete_hours) == 0:
        print("No complete hours found. Returning original dataset.")
        return ds_valid

    is_in_complete_hour = ds_valid.time.dt.floor("h").isin(complete_hours)
    ds_complete_hours = ds_valid.where(is_in_complete_hour, drop=True)
    return ds_complete_hours


def resample_dataarray(ds, source_resolution="6min", target_resolution="10min"):
    # First filter to only include complete hours
    ds_filtered = filter_complete_hours(ds, source_resolution)
    # Then resample to the target resolution using linear interpolation
    ds_resampled = ds_filtered.resample(time=target_resolution).interpolate("linear")
    return ds_resampled


def crop_dataarray(ds, lat, lon, tol=0.01):
    src_lat = ds.lat.values
    src_lon = ds.lon.values
    dst_lat = np.asarray(lat)
    dst_lon = np.asarray(lon)

    def _find_boundary_indices(src_coords, target_min, target_max, tol):
        """Helper function to find array indices for boundary values"""
        min_candidates = np.where(np.abs(src_coords - target_min) <= tol)[0]
        max_candidates = np.where(np.abs(src_coords - target_max) <= tol)[0]
        if not min_candidates.size or not max_candidates.size:
            raise ValueError("Target coordinates not found in source data")
        start, end = sorted([int(min_candidates[0]), int(max_candidates[0])])
        return start, end

    lat_start, lat_end = _find_boundary_indices(src_lat, dst_lat[0], dst_lat[-1], tol)
    lon_start, lon_end = _find_boundary_indices(src_lon, dst_lon[0], dst_lon[-1], tol)

    cropped = ds.isel(lat=slice(lat_start, lat_end + 1), lon=slice(lon_start, lon_end + 1))
    cropped = cropped.assign_coords(lat=dst_lat, lon=dst_lon)
    return cropped


def aggregate(ds, start_step, end_step, step_freq=24, dim="step"):
    end_step += 1
    steps = np.arange(start_step, end_step)
    lead_times = steps * step_freq
    print(f"lead_time: {lead_times}")
    init_time = ds.time[:-end_step]
    new_ds = []
    for hour in lead_times:
        valid_time = init_time + pd.Timedelta(hours=hour)
        valid_time, ind1, ind2 = np.intersect1d(valid_time, ds.time, return_indices=True)
        v = ds.sel(time=valid_time)
        v = v.assign_coords(time=init_time[ind1])
        new_ds.append(v)
    new_ds = xr.concat(new_ds, dim)
    coord = steps if dim == "step" else lead_times
    new_ds = new_ds.assign_coords({dim: coord})
    return new_ds


def area_mean(ds, lat=None, lon=None, mask=None):

    if mask is None:
        v = ds.sel(lat=lat, lon=lon)
    else:
        v = ds * mask

    if "lat" in v.dims:
        weights = np.cos(np.deg2rad(np.abs(v.lat)))
        v = v.weighted(weights).mean("lat").mean("lon")
    else:
        v = v.mean("lon")
    return v


def weekly_mean(ds):
    lead_times = 24 * np.arange(15, 43, dtype=np.int32)
    w34 = lead_times[:14]
    w56 = lead_times[14:]
    lead_times = np.split(lead_times, 4)
    lead_times.extend([w34, w56])
    new_ds = []
    for lead_time in lead_times:
        print(f"{len(lead_time)}: {lead_time}")
        v = ds.sel(lead_time=lead_time).mean("lead_time")
        new_ds.append(v)
    new_ds = xr.concat(new_ds, "lead_time")
    new_ds = new_ds.assign_coords(lead_time=["w3", "w4", "w5", "w6", "w34", "w56"])
    return new_ds


def season_mean(data_name, years=("19801231", "20100228")):
    ds = xr.open_zarr(data_name).data
    mean = xr.open_dataarray(data_name + "/mean.nc")
    std = xr.open_dataarray(data_name + "/std.nc")

    ds = ds.sel(time=slice(*years))
    ds = ds.sel(time=ds.time.dt.season == "DJF")
    ds = ds * std + mean
    tp = ds.data[:, -1]
    ds.data[:, -1] = np.exp(tp) - 1

    idx = np.arange(len(ds.time))
    v1 = ds.isel(time=idx[:-1]).mean("time")
    v2 = ds.isel(time=idx[1:]).mean("time")
    v = xr.concat([v1, v2], "time")
    v = v.assign_coords(time=ds.time[-2:])
    # save_nc(v, "initial.nc")


def get_doy(ds, no_leap_year=False):
    doy = ds.time.dt.dayofyear.values
    if no_leap_year:
        doy = ds.time.dt.dayofyear.values
        mask = (ds.time.dt.is_leap_year) & (ds.time.dt.month.values > 2)
        doy[mask] -= 1
    return doy


def sel_like(ds, ref, no_leap_year=True):
    doy = get_doy(ref, no_leap_year=no_leap_year)
    if "doy" in ds.dims:
        ds = ds.sel(doy=doy)
        ds = ds.rename({"doy": "time"})
    elif "dayofyear" in ds.dims:
        ds = ds.sel(dayofyear=doy)
        ds = ds.rename({"dayofyear": "time"})
    ds = ds.assign_coords(time=ref.time)
    ds = chunk(ds, time=1)
    return ds


def climate_mean(ds, no_leap_year=True):
    ds.coords["doy"] = ("time", get_doy(ds, no_leap_year=no_leap_year))
    clim = ds.groupby("doy").mean("time")
    groups = [len(v.time) for _, v in ds.groupby("doy")]
    is_same_size = min(groups) == max(groups)
    assert is_same_size, groups
    return clim


def climate_std(ds, no_leap_year=True):
    ds.coords["doy"] = ("time", get_doy(ds, no_leap_year=no_leap_year))
    dg = ds.groupby("doy")
    clim = []
    groups = []
    doys = []
    for i, v in dg:
        doys.append(i)
        groups.append(len(v.time))
        clim.append(v.std("time"))
    groups = np.array(groups[:365], dtype=np.int32)
    print(f"groups: \n {groups}")
    # assert sum(groups != ngroups) == 0, groups
    clim = xr.concat(clim, "doy")
    clim = clim.assign_coords(doy=doys)
    return clim


def tercile_edge(ds, q=[1.0 / 3.0, 2.0 / 3.0], dim="time"):
    ds.coords["doy"] = ("time", get_doy(ds, no_leap_year=True))
    dg = ds.groupby("doy")
    groups = np.array([len(v) for v in dg.groups.values()])
    print(f"dims: {ds.dims}")
    print(f"groups: {groups}")
    edge = dg.quantile(q=q, skipna=True, dim=dim)
    edge = edge.rename({"quantile": "category_edge"})
    return edge


def compute_climate(v):
    v = chunk(v, time=1)
    v = climate_mean(v, no_leap_year=True)
    v = chunk(v, doy=1)
    return v


def compute_trend(v):
    v = chunk(v, time=-1)
    v = get_trend(v)
    return v


def compute_edge(v, q):
    v = chunk(v, time=-1)
    if "member" in v.dims:
        v = tercile_edge(v, q=q, dim=("time", "member"))
    else:
        v = tercile_edge(v, q=q, dim=("time"))
    return v


def compute_prob(ds, edges, member_dim="member", tp_mask=None, q=["9/10"]):

    if "doy" in edges.dims:
        edges = sel_like(edges, ds)
        edges = chunk(edges, time=1)

    if len(q) == 1:
        above_normal = ds >= edges.sel(category_edge=q[0], drop=True)
        prob = xr.concat([above_normal], "category")
        prob = prob.assign_coords(category=["above normal"])
    else:
        below_normal = ds < edges.sel(category_edge=q[0], drop=True)
        near_normal = (ds >= edges.sel(category_edge=q[0], drop=True)) & (ds < edges.sel(category_edge=q[1], drop=True))  # normal
        below_normal = ds < edges.sel(category_edge=q[1], drop=True)
        prob = xr.concat([below_normal, near_normal, above_normal], "category")
        prob = prob.assign_coords(category=["below normal", "near normal", "above normal"])

    if member_dim in prob.dims:
        prob = prob.mean(member_dim)

    if "tp" in prob.dims and tp_mask:
        tp_mask = sel_like(tp_mask, prob)
        tp_mask = chunk(tp_mask, time=1)
        prob["tp"] = prob["tp"].where(tp_mask)

    return prob


def remove_annual_cycle(ds):
    print("Remove annual cycle ...")
    import torch

    for name in ds:
        v = ds[name]
        print(v.shape)
        x = torch.from_numpy(v.values)
        fx = torch.fft.rfft(x, dim=0)
        fx.real[3:].fill_(0)
        fx.imag[3:].fill_(0)
        v.data = torch.fft.irfft(fx, n=v.shape[0], dim=0).numpy()
        ds[name] = v
    print("Remove annual cycle done")
    return ds


def get_anomaly(ds, clim, remove_cycle=False):
    xclim = clim[ds.name]
    if remove_cycle:
        xclim = remove_annual_cycle(xclim)
    xclim = sel_like(xclim, ds)
    anno = ds - xclim
    return anno


@handle_dataset
def calc_diff_std(data, save_dir, downscaling=True):
    save_path = os.path.join(save_dir, "diff_mean.nc")

    if downscaling:
        data = resize_dataarray(data, resolution=1)

    data = chunk(data)
    data = data.astype(np.float64)

    # Calculate diff_mean
    print("Calculating diff_mean...")
    start = time.perf_counter()
    data_diff = data.diff(dim="time", n=1)
    diff_mean = data_diff.mean(("time", "lat", "lon"), skipna=True)
    diff_mean = diff_mean.load()
    calc_time = time.perf_counter() - start
    print(f"Calculation time (diff_mean): {calc_time:.2f} sec")
    print(f"Diff Mean range: {diff_mean.min().item():.6f} ~ {diff_mean.max().item():.6f}")
    save_nc(diff_mean, save_path)

    # Calculate diff_std
    print("Calculating diff_std...")
    start = time.perf_counter()
    data_diff = data.diff(dim="time", n=1)
    diff_std = data_diff.std(("time", "lat", "lon"), skipna=True)
    diff_std = diff_std.load()
    calc_time = time.perf_counter() - start
    print(f"Calculation time (diff_std): {calc_time:.2f} sec")
    print(f"Diff Std range: {diff_std.min().item():.6f} ~ {diff_std.max().item():.6f}")
    save_path = save_path.replace("diff_mean.nc", "diff_std.nc")
    save_nc(diff_std, save_path)


def calc_mean_std(data, save_dir, norm_cfg=None, downscaling=True):
    """Calculate mean and standard deviation of climate data.

    Args:
        data: xarray.DataArray - Input climate data
        save_dir: str - Directory to save/load mean and std files
        norm_cfg: dict - Configuration for normalization {var_name: (mean, std)}
        downscaling: bool - Whether to downscale data to 1 degree resolution

    Returns:
        tuple(xarray.DataArray, xarray.DataArray) - Mean and standard deviation
    """
    # Set default norm config if none provided
    if norm_cfg is None:
        norm_cfg = {"tp": (0, 1)}

    # check npy mean/std path exists
    mean_path = os.path.join(save_dir, "mean.npy")
    std_path = os.path.join(save_dir, "std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print(f"Loading cached mean and std from {mean_path} and {std_path}")
        mean = np.load(mean_path)
        mean = mean.reshape((1,) + mean.shape)
        std = np.load(std_path)
        std = std.reshape((1,) + std.shape)
        return mean, std

    mean_path = os.path.join(save_dir, "mean.nc")
    std_path = os.path.join(save_dir, "std.nc")

    # Load cached statistics if they exist
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print(f"Loading cached mean and std from {mean_path} and {std_path}")
        mean = xr.open_dataarray(mean_path)
        std = xr.open_dataarray(std_path)
        return mean, std

    # Preprocess data
    if downscaling:
        data = resize_dataarray(data, resolution=1)

    data = chunk(data)
    data = data.astype(np.float32)

    # Calculate statistics
    print("Calculating statistics...")
    start = time.perf_counter()

    if data.name in norm_cfg:
        # Use predefined normalization values
        mean_val, std_val = norm_cfg[data.name]
        template = data.isel(time=0, drop=True).mean(("lat", "lon"))
        mean = template * mean_val
        std = template * 0 + std_val
    else:
        # Use xarray's built-in methods (which use NumPy)
        mean = data.mean(("time", "lat", "lon"), skipna=True)
        std = data.std(("time", "lat", "lon"), skipna=True)

    # Load into memory
    mean = mean.load()
    std = std.load()
    calc_time = time.perf_counter() - start
    print(f"Calculation time: {calc_time:.2f} sec")
    print(f"Mean range: {mean.min().item():.6f} ~ {mean.max().item():.6f}")
    print(f"Std range: {std.min().item():.6f} ~ {std.max().item():.6f}")

    # Save results
    save_nc(mean, mean_path)
    save_nc(std, std_path)
    return mean, std


def merge_nc(src_name, dst_name):
    ds = xr.open_mfdataset(f"{src_name}/*.nc")
    if "time" in ds.dims:
        ds = ds.sortby("time")
    save_nc(ds, f"{dst_name}")


def save_nc(ds, save_name, dtype=np.float32, overwrite=False):
    """Save dataset to NetCDF file.

    Args:
        ds: xarray Dataset/DataArray to save
        save_name: Path to save file
        dtype: Data type to convert to before saving
        overwrite: If True, replace existing file if it exists. If False, raise error
    """
    if os.path.exists(save_name):
        if overwrite:
            os.remove(save_name)
        else:
            print(f"Store {save_name} already exists. Set overwrite=True to replace.")

    if "time" in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    if "channel" in ds.dims:
        ds.coords["channel"] = ds.channel.astype(str)

    ds = ds.astype(dtype)
    obj = ds.to_netcdf(save_name, compute=False)
    with ProgressBar():
        obj.compute()


def save_zarr(ds, save_name, dtype=None, overwrite=False):
    """Save dataset to Zarr store.

    Args:
        ds: xarray Dataset/DataArray to save
        save_name: Path to save store
        dtype: Optional data type to convert to before saving
        overwrite: If True, replace existing store if it exists. If False, raise error
    """
    if os.path.exists(save_name):
        if overwrite:
            shutil.rmtree(save_name, ignore_errors=True)
        else:
            print(f"Store {save_name} already exists. Set overwrite=True to replace.")

    if "time" in ds.dims:
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))

    if "channel" in ds.dims:
        ds.coords["channel"] = ds.channel.astype(str)

    ds = ds.astype(dtype)

    print(f"Save to {save_name} ...")
    obj = ds.to_zarr(save_name, compute=False)
    with ProgressBar():
        obj.compute()


def save_by_year(ds, save_dir, ftype="zarr", dtype=np.float32):
    os.makedirs(save_dir, exist_ok=True)
    y1 = pd.to_datetime(ds.time.values[0]).year
    y2 = pd.to_datetime(ds.time.values[-1]).year

    for y1 in range(y1, y2 + 1):
        t1 = f"{y1}0101"
        t2 = f"{y1}1231"
        data = ds.sel(time=slice(t1, t2))
        data = chunk(data)
        try:
            if ftype == "zarr":
                save_name = os.path.join(save_dir, f"{y1}")
                save_zarr(data, save_name, dtype=dtype, overwrite=True)
            elif ftype == "nc":
                save_name = os.path.join(save_dir, f"{y1}.nc")
                save_nc(data, save_name, dtype=dtype, overwrite=True)
        except Exception:
            print(f"Save {save_name} failed !!!")


def merge_member(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for time_str in os.listdir(data_dir):
        ds = []
        for step_str in os.listdir(f"{data_dir}/{time_str}"):
            files = glob.glob(f"{data_dir}/{time_str}/{step_str}/*.zarr")
            v = xr.open_mfdataset(files, engine="zarr", concat_dim="member", combine="nested")
            v = v.assign_coords(member=np.arange(v.member.size))
            v = chunk(v)
            ds.append(v)
        ds = xr.concat(ds, "step")
        ds = chunk(ds, step=1)
        print(ds)
        save_path = os.path.join(save_dir, f"{time_str}.zarr")
        save_zarr(ds, save_path)


def load_zarr(file_path: str, name: str | None = None) -> "xr.DataArray | None":
    if not file_path:
        return None

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load the dataset
    ds = xr.open_zarr(file_path, consolidated=True)

    # Get all data variables
    data_vars = list(ds.data_vars)

    if not data_vars:
        raise ValueError(f"No variables found in dataset: {file_path}")

    if len(data_vars) > 1:
        print(f"Multiple variables found in {file_path}: {data_vars}")
        print("Using first variable: {data_vars[0]}")

    # Get the first variable
    data = ds[data_vars[0]]

    if name is not None:
        data.name = name

    return data
