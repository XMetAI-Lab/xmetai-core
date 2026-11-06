"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-08-18 20:06:02
Description: Data utility functions for weather datasets.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import os
import re

import numpy as np
import pandas as pd
import torch
import xarray as xr

from xmetai.config import instantiate
from xmetai.utils.reanalyse import LEVELS_13

__all__ = [
    "make_seq",
    "make_seq_inference",
    "unnormalize",
    "make_input",
    "make_onnx",
    "get_data_info",
]


def make_seq(times, total_frames, freq=1, interval=1, freq_unit="hours"):
    step = pd.Timedelta(**{freq_unit: freq})
    i = 0
    inds = []

    while i < len(times) - total_frames:

        cur_sequence = [0]
        for j in range(i, i + total_frames - 1):
            if times[j + 1] - times[j] == step:
                cur_sequence.append(j)

        if len(cur_sequence) >= total_frames - 1:
            inds.append(i)

        i += interval

    return np.array(inds, dtype=np.int32)


def make_seq_inference(times, total_frames, freq=24, freq_unit="hours"):
    """
    Generate sequence indices starting with specific dates in the year.

    Args:
        ds: The dataset containing time values
        total_frames: The number of consecutive frames required for each sequence

    Returns:
        numpy.ndarray: Array of indices where sequences can start
    """
    # Define the specific days in the year for inference
    # fmt: off
    specific_dates = [
        "01-03", "01-06", "01-10", "01-13", "01-17", "01-20", "01-24", "01-27", "01-31",
        "02-03", "02-07", "02-10", "02-14", "02-17", "02-21", "02-24", "02-28",
        "03-03", "03-07", "03-10", "03-14", "03-17", "03-21", "03-24", "03-28", "03-31",
        "04-04", "04-07", "04-11", "04-14", "04-18", "04-21", "04-25", "04-28",
        "05-02", "05-05", "05-09", "05-12", "05-16", "05-19", "05-23", "05-26", "05-30",
        "06-02", "06-06", "06-09", "06-13", "06-16", "06-20", "06-23", "06-27", "06-30",
        "07-04", "07-07", "07-11", "07-14", "07-18", "07-21", "07-25", "07-28",
        "08-01", "08-04", "08-08", "08-11", "08-15", "08-18", "08-22", "08-25", "08-29",
        "09-01", "09-05", "09-08", "09-12", "09-15", "09-19", "09-22", "09-26", "09-29",
        "10-03", "10-06", "10-10", "10-13", "10-17", "10-20", "10-24", "10-27", "10-31",
        "11-03", "11-07", "11-10", "11-14", "11-17", "11-21", "11-24", "11-28",
        "12-01", "12-05", "12-08", "12-12", "12-15", "12-19", "12-22", "12-26", "12-29",
    ]
    # fmt: on

    inds = []

    # Iterate through all time values in the dataset
    for i in range(len(times) - total_frames):
        # Extract the date parts from the current timestamp
        current_date = pd.Timestamp(times[i + 1])
        month_day = current_date.strftime("%m-%d")
        hour = current_date.hour

        # If the current date is one of our specific dates, check if we have enough consecutive frames
        if month_day in specific_dates:
            if freq_unit == "hours" and freq == 6 and hour != 18:  # with 12+18 input frames, predict next 00:00 hours start.
                continue
            # Check if we have enough consecutive frames
            valid_sequence = True
            for j in range(1, total_frames):
                # Check if next frame is sequential (1 day = 24 hours typically for weather data)
                if i + j < len(times) and times[i + j] - times[i + j - 1] != pd.Timedelta(**{freq_unit: freq}):
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
    input_data = batch["inputs"]  # (b, t, c, h, w)
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
    extra_inputs=None,
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
    if extra_inputs is not None:
        data.update(extra_inputs)

    model.export_kwargs = dict(
        data=data,
        save_name=os.path.join(output_dir, model_name),
        forward_name=forward_name,
        device=device,
    )

    # set environment variable for torch cpu mode
    torch.set_num_threads(os.cpu_count())
    # set environment variable for numpy cpu mode
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
    # enable mkldnn
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.bench = True


def get_data_info(data_dirs, ignore_names=[]):
    pl_names = {"z", "t", "u", "v", "q"}
    accum_names = {"ssr", "ssrd", "fdir", "ttr", "tp"}
    ocean_names = {"sst", "mdts", "mdww", "mpts", "mpww", "shts", "shww"}
    land_names = {
        "stl1",
        "stl2",
        "stl3",
        "stl4",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
    }

    stores = []
    store_channel_names = []
    try:
        for data_dir in data_dirs:
            # Metadata reads must not start Dask threads before training workers fork.
            try:
                store = xr.open_zarr(data_dir, consolidated=True, chunks=None)
            except (KeyError, ValueError):
                store = xr.open_zarr(data_dir, consolidated=False, chunks=None)
            stores.append(store)

            if "channel" not in store.coords:
                raise ValueError(f"{data_dir} does not have channel")
            store_channel_names.append(store.coords["channel"].values.tolist())

        ori_in_names = sum(store_channel_names, [])
        in_names = [name for name in ori_in_names if name not in ignore_names]
        logid = [i for i, name in enumerate(in_names) if name == "tp"]
        accumid = [i for i, name in enumerate(in_names) if name in accum_names]
        maskid = [i for i, name in enumerate(in_names) if name in ocean_names]
        landid = [i for i, name in enumerate(in_names) if name in land_names]

        channel_weights = np.ones(len(in_names), dtype=np.float32)
        level_names = {str(level) for level in LEVELS_13}
        for i, name in enumerate(in_names):
            match = re.match(r"^([A-Za-z]+)_?(\d+)$", name)
            if match is None:
                continue
            var_name, level_name = match.groups()
            if var_name in pl_names and level_name in level_names:
                channel_weights[i] = max(0.2, int(level_name) / 1000)
        channel_weights[landid] *= 0.33
        channel_weights[maskid] *= 0.67
        channel_weights /= channel_weights.max()
        print(f"Channel weights: {channel_weights}")

        buffers = {"channel_weights": channel_weights.reshape((-1, 1, 1))}
        for key in ["mean", "std", "weight", "mask", "const"]:
            for data_dir, data_names, store in zip(data_dirs, store_channel_names, stores):
                file_name = os.path.join(data_dir, f"{key}.nc")
                npy_name = os.path.join(data_dir, f"{key}.npy")

                if key in {"mean", "std"}:
                    values = None

                    # 1. 新 Zarr 格式：mean/std 作为变量（优先）
                    if key in store:
                        values = store[key].values

                    # 2. 兼容旧格式：mean.nc/std.nc
                    elif os.path.exists(file_name):
                        with xr.open_dataarray(file_name) as data_array:
                            if "channel" in data_array.coords:
                                sidecar_names = data_array.coords["channel"].values.tolist()
                                if all(name in sidecar_names for name in data_names):
                                    values = data_array.sel(channel=data_names).values

                            if values is None and data_array.size == len(data_names):
                                values = data_array.values.reshape(-1)

                    # 3. 兼容旧格式：mean.npy/std.npy
                    elif os.path.exists(npy_name):
                        values = np.load(npy_name).reshape(-1)

                    if values is None:
                        if "data" not in store or key not in store["data"].attrs:
                            raise ValueError(f"Cannot align {key!r} statistics for {data_dir}: no matching sidecar or data attribute.")
                        attr_values = np.asarray(store["data"].attrs[key], dtype=np.float32).reshape(-1)
                        if attr_values.size % len(data_names) != 0:
                            raise ValueError(
                                f"Cannot align {key!r} statistics for {data_dir}: attribute length {attr_values.size} "
                                f"is not a multiple of {len(data_names)} channels."
                            )
                        blocks = attr_values.reshape(-1, len(data_names))
                        if blocks.shape[0] > 1 and not np.allclose(blocks, blocks[:1], rtol=0.0, atol=0.0):
                            raise ValueError(f"Cannot align {key!r} statistics for {data_dir}: repeated attribute blocks are inconsistent.")
                        values = blocks[0]
                        # raise ValueError(f"{key} not found in {data_dir}")

                elif key == "mask" and "mask" in store:
                    if "mask_channel" not in store:
                        raise ValueError(f"{data_dir} has mask but no mask_channel")
                    mask_names = store["mask_channel"].values.tolist()
                    mask = store["mask"].values
                    if "sea_mask" in mask_names:
                        buffers["mask"] = mask[mask_names.index("sea_mask")]
                    if "land_mask" in mask_names:
                        buffers["land_mask"] = mask[mask_names.index("land_mask")]
                    continue
                elif os.path.exists(file_name):
                    with xr.open_dataarray(file_name) as data_array:
                        values = data_array.values
                elif os.path.exists(npy_name):
                    values = np.load(npy_name)
                else:
                    print(f"File {file_name} or Zarr variable '{key}' not found, using default value for {key}.")
                    continue

                if key == "weight":
                    values = values.reshape(1, -1, 1)
                elif key in {"mean", "std"}:
                    values = values.reshape(-1, 1, 1)
                    buffers[key] = np.concatenate([buffers[key], values], axis=0) if key in buffers else values
                    continue
                if key in buffers:
                    print(f"{key} already exists, updated by {file_name}")
                buffers[key] = values

        if ignore_names:
            keep_channels = ~np.isin(ori_in_names, ignore_names)
            for key in ["mean", "std"]:
                if key in buffers:
                    buffers[key] = buffers[key][keep_channels]

        return {
            "in_names": in_names,
            "out_names": in_names,
            "logid": logid,
            "accumid": accumid,
            "maskid": maskid,
            "landid": landid,
            "buffers": buffers,
        }
    finally:
        for store in stores:
            store.close()
