"""Unit tests for GraphCastDataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xmetai.data import GraphCastDataset


@pytest.mark.numerical
def test_graphcast_dataset_batch_keys(tmp_path):
    time = pd.date_range("2000-01-01", periods=10, freq="6h")
    level = [1000, 850]
    lat = np.linspace(90, -90, 4, dtype=np.float32)
    lon = np.linspace(0, 360, 8, dtype=np.float32, endpoint=False)

    data = np.random.randn(len(time), len(level), len(lat), len(lon)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=["time", "level", "lat", "lon"],
        coords={"time": time, "level": level, "lat": lat, "lon": lon},
        name="data",
    )
    ds = xr.Dataset({"data": da})

    zarr_path = tmp_path / "toy.zarr"
    ds.to_zarr(zarr_path, mode="w")

    dataset = GraphCastDataset(
        data_name=str(zarr_path),
        hist_frames=2,
        fcst_frames=1,
        freq=6,
        interval=1,
        training=True,
        years=("20000101", "20000103"),
    )

    sample = dataset[0]
    assert "inputs" in sample
    assert "targets" in sample
    assert "edge_index" in sample
    assert "edge_attr" in sample
    assert "node_static" in sample
    assert "grid_shape" in sample

    inputs = sample["inputs"]
    targets = sample["targets"]
    assert inputs.ndim == 4
    assert targets.ndim == 4
    assert sample["edge_index"].shape[0] == 2
    assert sample["edge_attr"].shape[1] == 3
