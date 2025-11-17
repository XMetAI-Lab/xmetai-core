'''
Author: DouZesheng && douzsh@gmail.com
Date: 2025-08-18 20:06:02
Description: Data utility functions for weather datasets.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
'''
import pandas as pd
import numpy as np
import os
import re
import glob
import torch
import xarray as xr
from xmetai.config import instantiate

__all__ = ["make_seq", "make_seq_inference", "make_onnx", "get_data_info", "get_cra_data_info"]


def make_seq(times, total_frames, freq=1, interval=1):
    i = 0
    inds = []

    while i < len(times) - total_frames:

        cur_sequence = [0]
        for j in range(i, i + total_frames - 1):
            if times[j + 1] - times[j] == pd.Timedelta(hours=freq):
                cur_sequence.append(j)

        if len(cur_sequence) >= total_frames - 1:
            inds.append(i)

        i += interval

    return np.array(inds, dtype=np.int32)


def make_seq_inference(times, total_frames, freq=24):
    """
    Generate sequence indices starting with specific dates in the year.

    Args:
        ds: The dataset containing time values
        total_frames: The number of consecutive frames required for each sequence

    Returns:
        numpy.ndarray: Array of indices where sequences can start
    """
    # Define the specific days in the year for inference
    specific_dates = [
        '01-03', '01-06', '01-10', '01-13', '01-17', '01-20', '01-24', '01-27', '01-31',
        '02-03', '02-07', '02-10', '02-14', '02-17', '02-21', '02-24', '02-28',
        '03-03', '03-07', '03-10', '03-14', '03-17', '03-21', '03-24', '03-28', '03-31',
        '04-04', '04-07', '04-11', '04-14', '04-18', '04-21', '04-25', '04-28',
        '05-02', '05-05', '05-09', '05-12', '05-16', '05-19', '05-23', '05-26', '05-30',
        '06-02', '06-06', '06-09', '06-13', '06-16', '06-20', '06-23', '06-27', '06-30',
        '07-04', '07-07', '07-11', '07-14', '07-18', '07-21', '07-25', '07-28',
        '08-01', '08-04', '08-08', '08-11', '08-15', '08-18', '08-22', '08-25', '08-29',
        '09-01', '09-05', '09-08', '09-12', '09-15', '09-19', '09-22', '09-26', '09-29',
        '10-03', '10-06', '10-10', '10-13', '10-17', '10-20', '10-24', '10-27', '10-31',
        '11-03', '11-07', '11-10', '11-14', '11-17', '11-21', '11-24', '11-28',
        '12-01', '12-05', '12-08', '12-12', '12-15', '12-19', '12-22', '12-26', '12-29'
    ]

    inds = []

    # Iterate through all time values in the dataset
    for i in range(len(times) - total_frames):
        # Extract the date parts from the current timestamp
        current_date = pd.Timestamp(times[i + 1])
        month_day = current_date.strftime('%m-%d')
        hour = current_date.hour

        # If the current date is one of our specific dates, check if we have enough consecutive frames
        if month_day in specific_dates:
            if freq == 6 and hour != 18:  # with 12+18 input frames, predict next 00:00 hours start.
                continue
            # Check if we have enough consecutive frames
            valid_sequence = True
            for j in range(1, total_frames):
                # Check if next frame is sequential (1 day = 24 hours typically for weather data)
                if i + j < len(times) and times[i + j] - times[i + j - 1] != pd.Timedelta(hours=freq):
                    valid_sequence = False
                    break

            # If we have a valid sequence, add this index
            if valid_sequence:
                inds.append(i)

    return np.array(inds, dtype=np.int32)


def unnormalize(data, in_names, mean, std, log_names=["tp"]):
    data = data * std + mean
    # apply for tp
    logid = in_names.index("tp")
    print(f"applying exp transformation to {logid}")
    tp = data[:, :, logid].clamp(min=0.0, max=7.0)
    tp = tp.exp() - 1
    data[:, :, logid] = tp
    return data


def make_input(
    dataloader,
    in_names,
    mean=None,
    std=None,
    log_names=["tp"],
):
    # fetch one batch of data
    batch = next(iter(dataloader))
    input_data = batch['inputs']  # (b, t, c, h, w)
    print(f"input_data: {input_data.shape}, {input_data.min()} ~ {input_data.max()}")

    if mean is not None and std is not None:
        input_data = unnormalize(input_data, in_names, mean, std, log_names=log_names)

    return input_data


def make_onnx(
    model,
    dataloader_cfg,
    output_dir,
    in_names,
    mean=None,
    std=None,
    batch_size=1,
    model_name=None,
    device="cuda",
    forward_name="export_onnx",
    embed_types=["step", "hour", "doy"],
):
    dataloader = instantiate(dataloader_cfg)
    input_data = make_input(dataloader, in_names, mean, std)
    print(f"input_data: {input_data.shape}, {input_data.min()} ~ {input_data.max()}")

    if model_name is None:
        model_name = "model_infer.onnx"
    elif not model_name.endswith(".onnx"):
        model_name += ".onnx"

    os.makedirs(os.path.dirname(os.path.join(output_dir, model_name)), exist_ok=True)

    data = dict(input=input_data)

    for embed_type in embed_types:
        data[embed_type] = np.random.rand(batch_size).astype(np.float32)

    model.export_kwargs = dict(
        data=data,
        save_name=os.path.join(output_dir, model_name),
        forward_name=forward_name,
        device=device,
    )

    # set environment variable for torch cpu mode
    torch.set_num_threads(os.cpu_count())
    torch.set_num_interop_threads(os.cpu_count())
    # set environment variable for numpy cpu mode
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
    # enable mkldnn
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.bench = True


def get_ids_and_buffer(data_dirs, in_names, ori_in_names=[], ignore_names=[]):
    LEVELS_13 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    LEVELS_37 = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350,
                 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
    # z: geopotential
    # t: temperature
    # u: zonal wind
    # v: meridional wind
    # q: specific humidity
    pl_names = ['z', 't', 'u', 'v', 'q']  # with 5 upper-air atmospheric variables
    accum_name = ['ssr', 'ssrd', 'fdir', 'ttr', 'tp']
    ocean_name = ["sst", "mdts", "mdww", "mpts", "mpww", "shts", "shww"]
    land_name = ["stl1", "stl2", "stl3", "stl4", "swvl1", "swvl2", "swvl3", "swvl4"]  # soil temperature and water

    logid = [in_names.index(n) for n in ["tp"] if n in in_names]
    accumid = [in_names.index(n) for n in accum_name if n in in_names]
    maskid = [in_names.index(n) for n in ocean_name if n in in_names]
    landid = [in_names.index(n) for n in land_name if n in in_names]

    out_names = in_names  # + ws_name

    # Calculate channel weights
    channel_weights = np.ones(len(in_names), dtype=np.float32)
    for name in in_names:
        if any(pl in name for pl in pl_names):
            if any(level in name for level in map(str, LEVELS_13)):
                # for 10u/10v/100u/100v/u10m/v10m, assign weight 1.0
                if name[-1] != "0":
                    continue
                level = int(name[1:])
                channel_weights[in_names.index(name)] = max(0.2, level / 1000)  # min weight 0.2 for upper-air variables
    channel_weights[landid] *= 0.33  # reduce land variables' weight
    channel_weights[maskid] *= 0.67  # reduce ocean variables' weight
    channel_weights /= channel_weights.max()  # normalize to [0, 1]
    print(f"Channel weights: {channel_weights}")

    buffers = {"channel_weights": channel_weights.reshape((-1, 1, 1))}
    for k in ["mean", "std", "weight", "mask", "land_mask", "const"]:
        for data_dir in data_dirs:
            # Load the buffer file if it exists
            file_name = os.path.join(data_dir, f"{k}.nc")
            if os.path.exists(file_name):
                v = xr.open_dataarray(file_name).values
            elif os.path.exists(os.path.join(data_dir, f"{k}.npy")):
                v = np.load(os.path.join(data_dir, f"{k}.npy"))
            else:
                print(f"File {file_name} not found, using default value for {k}.")
                continue
            if k == "weight":
                v = v.reshape(1, -1, 1)
            elif k in ['mean', 'std']:
                v = v.reshape(-1, 1, 1)
                if k not in buffers:
                    buffers[k] = v
                else:  # concat if already exists
                    buffers[k] = np.concatenate([buffers[k], v], axis=0)
                continue
            if k in buffers:
                print(f"{k} already exists, updated by {file_name}")
            buffers[k] = v

    # update mean/std for ignored variables.
    if len(ignore_names) > 0:
        for k in ['mean', 'std']:
            if k in buffers:
                buffers[k] = buffers[k][~np.isin(ori_in_names, ignore_names)]

    return {
        "in_names": in_names,
        "out_names": out_names,
        "logid": logid,
        "accumid": accumid,
        "maskid": maskid,
        "landid": landid,
        "buffers": buffers,
    }


def get_data_info(data_dirs, ignore_names=[]):
    coord = "level"
    if coord not in xr.open_zarr(data_dirs[0]).coords:
        coord = "channel"
    ori_in_names = sum([xr.open_zarr(d).coords[coord].values.tolist() for d in data_dirs], [])
    in_names = [n for n in ori_in_names if n not in ignore_names]
    return get_ids_and_buffer(data_dirs, in_names, ori_in_names=ori_in_names, ignore_names=ignore_names)


def get_cra_data_info(data_dir, in_names=["tp"]):
    # get all grib2 files and sort by datetime(*yyyymmddhh.grib2)
    pattern = re.compile(r'(\d{10})\.grib2$')
    file_list = glob.glob(os.path.join(data_dir, "**", "*.grib2"), recursive=True)
    file_map = {}
    for file_path in file_list:
        match = pattern.search(file_path)
        if match:
            datetime_str = match.group(1)
            file_map[datetime_str] = file_path
    sorted_datetimes = sorted(file_map.keys())
    sorted_files = [file_map[dt] for dt in sorted_datetimes]
    start_time = sorted_datetimes[0]
    end_time = sorted_datetimes[-1]
    print(f"Total files found: {len(sorted_files)}, starting from {start_time} to {end_time}")

    # check if file count matches expected count
    start_pd = pd.to_datetime(start_time, format="%Y%m%d%H")
    end_pd = pd.to_datetime(end_time, format="%Y%m%d%H")
    expected_frames = int(((end_pd - start_pd).total_seconds() // 3600) / 6) + 1
    if len(sorted_files) != expected_frames:
        print(f"Warning: Expected {expected_frames} files based on time range, but found {len(sorted_files)} files.")

    return get_ids_and_buffer([data_dir], in_names)
