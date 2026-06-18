import numpy as np
import pandas as pd

from xmetai.data.weather.meteonet_dataset import MeteoNetRadarDataset


def _write_meteonet_npz(path, start, frame_values, shape=(3, 4), missing_offsets=(), data_overrides=()):
    dates = [start + pd.Timedelta(minutes=5 * i) for i in range(len(frame_values))]
    data = np.stack([np.full(shape, int(value), dtype=np.int16) for value in frame_values], axis=0)
    for frame, row, col, value in data_overrides:
        data[frame, row, col] = value
    prob = np.stack([np.full(shape, 50 + i, dtype=np.int16) for i in range(len(frame_values))], axis=0)
    height = np.stack([np.full(shape, 1000 + i, dtype=np.int16) for i in range(len(frame_values))], axis=0)
    miss_dates = [dates[i] for i in missing_offsets]

    np.savez_compressed(
        path,
        data=data,
        prob=prob,
        height=height,
        dates=np.asarray([date.to_pydatetime() for date in dates], dtype=object),
        miss_dates=np.asarray([date.to_pydatetime() for date in miss_dates], dtype=object),
    )


def test_meteonet_radar_dataset_discovers_npz_dates_and_sequence_shapes(tmp_path):
    start = pd.Timestamp("2018-02-01 00:00:00")
    _write_meteonet_npz(tmp_path / "reflectivity_new_SE_2018_02.2.npz", start + pd.Timedelta(minutes=20), [140, 150, 160, 170])
    _write_meteonet_npz(tmp_path / "reflectivity_new_SE_2018_02.1.npz", start, [100, 110, 120, 130], data_overrides=((0, 0, 1, -200),))

    dataset = MeteoNetRadarDataset(
        data_paths=tmp_path,
        hist_frames=2,
        fcst_frames=[1, 2],
        freq=5,
        freq_unit="minutes",
        interval=-1,
        training=False,
        crop=(0, 2, 1, 3),
    )
    sample = dataset[0]

    assert len(dataset) == 5
    assert sample["inputs"].shape == (2, 1, 2, 2)
    assert sample["targets"].shape == (2, 1, 2, 2)
    assert sample["grid_shape"].tolist() == [2, 2]
    assert sample["times"].item() == (start + pd.Timedelta(minutes=5)).value
    assert float(sample["inputs"][0, 0, 0, 0]) == 0.0
    assert float(sample["inputs"][1, 0, 0, 0]) == 11.0
    assert float(sample["targets"][1, 0, 0, 0]) == 13.0


def test_meteonet_radar_dataset_treats_miss_dates_as_sequence_breaks(tmp_path):
    start = pd.Timestamp("2018-02-01 00:00:00")
    _write_meteonet_npz(tmp_path / "reflectivity_new_SE_2018_02.1.npz", start, [100, 110, 120, 130, 140, 150], missing_offsets=(2,))

    dataset = MeteoNetRadarDataset(
        data_paths=tmp_path,
        hist_frames=2,
        fcst_frames=1,
        freq=5,
        freq_unit="minutes",
        interval=1,
        training=True,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["times"].item() == (start + pd.Timedelta(minutes=20)).value
    assert float(sample["inputs"][0, 0, 0, 0]) == 13.0
    assert float(sample["inputs"][1, 0, 0, 0]) == 14.0
    assert float(sample["targets"][0, 0, 0, 0]) == 15.0


def test_meteonet_radar_dataset_supports_multiple_channels(tmp_path):
    start = pd.Timestamp("2018-02-01 00:00:00")
    _write_meteonet_npz(tmp_path / "reflectivity_new_SE_2018_02.1.npz", start, [120, 130])

    dataset = MeteoNetRadarDataset(
        data_paths=tmp_path,
        hist_frames=1,
        fcst_frames=1,
        freq=5,
        freq_unit="minutes",
        training=False,
        inference_only=True,
        channels=("data", "prob", "height"),
        cache_size=0,
    )
    sample = dataset[0]

    assert sample["inputs"].shape == (1, 3, 3, 4)
    assert float(sample["inputs"][0, 0, 0, 0]) == 12.0
    assert float(sample["inputs"][0, 1, 0, 0]) == 50.0
    assert float(sample["inputs"][0, 2, 0, 0]) == 1000.0
