#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run UNet3D precipitation correction and export tp-only Zarr v3 output."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

XMETAI_ROOT = Path(__file__).resolve().parent.parent
if os.fspath(XMETAI_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(XMETAI_ROOT))

from xmetai.data.weather.prcpcorrect_dataset import PrcpCorrectionUNet3DDataset  # noqa: E402
from xmetai.models.prcpcorrect_unet3d import PrcpCorrectionUNet3D  # noqa: E402

LOGGER = logging.getLogger("prcpcorrect_unet3d_infer")
OUTPUT_SCHEMA = "xmetai.prcp_correction.unet3d_prediction_v1"


def _env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Environment variable {name} is required.")
    return Path(value).expanduser().resolve()


def _resolve_paths() -> tuple[Path, Path, Path, Path, Path]:
    data_root = _env_path("PRCP_DATA_ROOT")
    checkpoint = _env_path("INIT_CHECKPOINT")
    output = _env_path("INFER_OUTPUT")
    fengq = data_root / "FENGQ15.zarr"
    input_min = _env_path("FENGQ_MIN_PATH")
    input_max = _env_path("FENGQ_MAX_PATH")

    for name, path in (("checkpoint", checkpoint), ("FENGQ15", fengq), ("input minimum", input_min), ("input maximum", input_max)):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    return checkpoint, output, fengq, input_min, input_max


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    expected_min = model.input_min.detach().cpu().clone()
    expected_max = model.input_max.detach().cpu().clone()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if state_dict and all(name.startswith("module.") for name in state_dict):
        state_dict = {name.removeprefix("module."): value for name, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    if not torch.equal(model.input_min.cpu(), expected_min) or not torch.equal(model.input_max.cpu(), expected_max):
        raise ValueError("The selected min/max statistics do not match those stored in the checkpoint.")


def _create_coordinate(
    group: zarr.Group,
    source: zarr.Group,
    name: str,
    values: np.ndarray,
    dimensions: list[str],
) -> None:
    array = group.create_array(name, data=values, fill_value=None, dimension_names=dimensions)
    array.attrs.update(dict(source[name].attrs))
    array.attrs["_ARRAY_DIMENSIONS"] = dimensions


def _create_output(
    dataset: PrcpCorrectionUNet3DDataset,
    output_path: Path,
    checkpoint_path: Path,
) -> zarr.Array:
    source = dataset.fengq_group
    indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    steps, _, height, width = source["input"].shape[1:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(os.fspath(output_path), mode="w-", zarr_format=3)
    group.attrs.update(
        schema=OUTPUT_SCHEMA,
        normalization="none",
        time_zone="Beijing time",
        variable="tp",
        units="mm",
        checkpoint=checkpoint_path.name,
        evaluation_observation_mask="CMPAS valid_mask; predictions cover the full grid",
    )
    prediction = group.create_array(
        "input",
        shape=(len(indices), steps, 1, height, width),
        chunks=(1, steps, 1, height, width),
        dtype=np.float32,
        fill_value=np.nan,
        dimension_names=["time", "dtime", "pl", "lat", "lon"],
    )
    prediction.attrs.update(
        _ARRAY_DIMENSIONS=["time", "dtime", "pl", "lat", "lon"],
        long_name="UNet3D corrected six-hour accumulated precipitation",
        units="mm",
    )

    selected_times = source["time"].get_orthogonal_selection((indices,))
    selected_valid_times = source["valid_time"].get_orthogonal_selection((indices, slice(None)))
    _create_coordinate(group, source, "time", selected_times, ["time"])
    _create_coordinate(group, source, "dtime", source["dtime"][:], ["dtime"])
    _create_coordinate(group, source, "valid_time", selected_valid_times, ["time", "dtime"])
    precipitation = group.create_array("pl", data=np.asarray(["tp"], dtype="U2"), fill_value=None, dimension_names=["pl"])
    precipitation.attrs["_ARRAY_DIMENSIONS"] = ["pl"]
    _create_coordinate(group, source, "lat", source["lat"][:], ["lat"])
    _create_coordinate(group, source, "lon", source["lon"][:], ["lon"])
    return prediction


def run_inference() -> Path:
    if zarr.__version__.split(".", 1)[0] != "3":
        raise RuntimeError(f"UNet3D precipitation inference requires Zarr 3, got {zarr.__version__}.")

    checkpoint, output, fengq, input_min, input_max = _resolve_paths()
    split_csv = os.environ.get("PRCP_TEST_CSV")
    start_time = os.environ.get("TEST_START")
    end_time = os.environ.get("TEST_END")
    batch_size = int(os.environ.get("INFER_BATCH_SIZE", "1"))
    num_workers = int(os.environ.get("PRCP_NUM_WORKERS", "4"))
    base_channels = int(os.environ.get("BASE_CHANNELS", "128"))
    device = os.environ.get("DEVICE", "cuda")
    if split_csv and (start_time or end_time):
        raise ValueError("PRCP_TEST_CSV cannot be combined with TEST_START or TEST_END.")
    if batch_size < 1 or num_workers < 0 or base_channels < 1:
        raise ValueError("batch size and base channels must be positive; num workers must be non-negative.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    dataset = PrcpCorrectionUNet3DDataset(
        fengq_zarr_path=os.fspath(fengq),
        cmpas_zarr_path=None,
        input_min=os.fspath(input_min),
        input_max=os.fspath(input_max),
        split_csv=os.fspath(Path(split_csv).expanduser().resolve()) if split_csv else None,
        start_time=start_time,
        end_time=end_time,
    )
    channels = int(np.load(input_min, mmap_mode="r").shape[0])
    model = PrcpCorrectionUNet3D(
        input_min=os.fspath(input_min),
        input_max=os.fspath(input_max),
        in_channels=channels,
        base_channels=base_channels,
        precision="fp32",
        device=device,
    )
    _load_checkpoint(model, checkpoint)
    model.to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    prediction = _create_output(dataset, output, checkpoint)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            values = model(batch)["output"].float().cpu().numpy()
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Inference produced non-finite or negative precipitation.")
            prediction[offset : offset + len(values)] = values
            offset += len(values)
            LOGGER.info("Wrote %d/%d samples", offset, len(dataset))

    zarr.consolidate_metadata(os.fspath(output))
    LOGGER.info("Inference completed: %s", output)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_inference()


if __name__ == "__main__":
    main()
