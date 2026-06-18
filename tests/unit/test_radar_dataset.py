import bz2
import json
import struct

import numpy as np
import pandas as pd
from PIL import Image

from xmetai.data.weather.radar_dataset import RadarDataset
from xmetai.utils.radar import RADAR_BBOX_CHINA, RADAR_BBOX_EAST_CHINA, RADAR_BBOX_REGIONS, read_radar_header, read_radar_reflectivity, save_radar_png


def _write_radar_file(path, timestamp, raw):
    raw = np.asarray(raw, dtype="<i2")
    height, width = raw.shape
    payload = bz2.compress(raw.tobytes())
    header = bytearray(256)
    header[:4] = b"MOC\x00"
    header[0x10:0x18] = b"CREF\x00\x00\x00\x00"
    header[0x18 : 0x18 + len(b"Composite Reflectivity")] = b"Composite Reflectivity"
    struct.pack_into("<I", header, 0x08, 256 + len(payload))
    struct.pack_into("<I", header, 0x58, 256)
    struct.pack_into("<I", header, 0x5C, len(payload))
    struct.pack_into("<6H", header, 0x64, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second)
    struct.pack_into("<I", header, 0x70, timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second)
    struct.pack_into("<i", header, 0x80, 70000)
    struct.pack_into("<i", header, 0x84, 60000)
    struct.pack_into("<i", header, 0x88, 70000 + (width - 1) * 10)
    struct.pack_into("<i", header, 0x8C, 70000)
    struct.pack_into("<i", header, 0x90, 60000)
    struct.pack_into("<I", header, 0x94, width)
    struct.pack_into("<I", header, 0x98, height)
    struct.pack_into("<I", header, 0x9C, 100)
    struct.pack_into("<I", header, 0xA0, 100)
    struct.pack_into("<I", header, 0xAC, raw.nbytes)
    path.write_bytes(bytes(header) + payload)


def test_read_radar_header_and_reflectivity(tmp_path):
    timestamp = pd.Timestamp("2019-12-16 00:00:00")
    raw = np.array(
        [
            [-32768, -1280, 0, 55],
            [100, 250, 580, -1280],
        ],
        dtype=np.int16,
    )
    path = tmp_path / "ACHN_CR_20191216_000000.bin"
    _write_radar_file(path, timestamp, raw)

    header = read_radar_header(path)
    data = read_radar_reflectivity(path, header=header, fill_value=None)
    filled = read_radar_reflectivity(path, header=header, fill_value=0.0)

    assert header.product == "CREF"
    assert header.timestamp == timestamp
    assert header.width == 4
    assert header.height == 2
    assert np.isclose(header.lon_step, 0.01)
    assert data[0, 0] == -32768
    assert data[0, 1] == -128
    assert data[0, 3] == 5.5
    assert data[1, 2] == 58.0
    assert filled[0, 0] == 0.0
    assert filled[0, 1] == 0.0


def test_read_radar_reflectivity_qc_removes_speckles_and_invalid_values(tmp_path):
    timestamp = pd.Timestamp("2019-12-16 00:00:00")
    raw = np.zeros((5, 5), dtype=np.int16)
    raw[1, 1] = 200
    raw[3, 1] = 200
    raw[3, 2] = 200
    raw[0, 4] = 900
    path = tmp_path / "ACHN_CR_20191216_000000.bin"
    _write_radar_file(path, timestamp, raw)

    raw_data = read_radar_reflectivity(path, fill_value=0.0, qc=False)
    qc_data = read_radar_reflectivity(path, fill_value=0.0, qc=True)

    assert raw_data[1, 1] == 20.0
    assert raw_data[0, 4] == 90.0
    assert qc_data[1, 1] == 0.0
    assert qc_data[0, 4] == 0.0
    assert qc_data[3, 1] == 20.0
    assert qc_data[3, 2] == 20.0


def test_save_radar_png(tmp_path):
    output_path = tmp_path / "radar.png"
    boundary_path = tmp_path / "boundary.geojson"
    boundary_path.write_text(json.dumps({"type": "LineString", "coordinates": [[0.0, 1.0], [1.0, 0.0]]}), encoding="utf-8")
    result_path = save_radar_png(
        np.zeros((16, 16), dtype=np.float32),
        output_path,
        bbox=(0.0, 0.0, 1.0, 1.0),
        boundary_geojson=boundary_path,
        boundary_color=(0, 0, 0, 255),
    )

    assert result_path == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    rgba = np.asarray(Image.open(output_path).convert("RGBA"))
    assert np.any(np.all(rgba == np.array([0, 0, 0, 255], dtype=np.uint8), axis=-1))


def test_common_radar_bbox_constants():
    assert RADAR_BBOX_REGIONS["全国"] == RADAR_BBOX_CHINA
    assert RADAR_BBOX_REGIONS["huadong"] == RADAR_BBOX_EAST_CHINA
    assert RADAR_BBOX_REGIONS["华东"] == RADAR_BBOX_EAST_CHINA


def test_radar_dataset_sequence_shapes(tmp_path):
    start = pd.Timestamp("2019-12-16 00:00:00")
    for i in range(4):
        timestamp = start + pd.Timedelta(minutes=6 * i)
        raw = np.full((3, 4), 100 + i * 10, dtype=np.int16)
        path = tmp_path / timestamp.strftime("ACHN_CR_%Y%m%d_%H%M%S.bin")
        _write_radar_file(path, timestamp, raw)

    dataset = RadarDataset(
        data_paths=tmp_path,
        hist_frames=2,
        fcst_frames=[1, 2],
        freq=6,
        freq_unit="minutes",
        interval=-1,
        training=False,
        years=("20191216", "20191216"),
        crop=(1, 3, 1, 4),
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["inputs"].shape == (2, 1, 2, 3)
    assert sample["targets"].shape == (2, 1, 2, 3)
    assert sample["grid_shape"].tolist() == [2, 3]
    assert sample["times"].item() == (start + pd.Timedelta(minutes=6)).value
    assert float(sample["inputs"][0, 0, 0, 0]) == 10.0
    assert float(sample["targets"][1, 0, 0, 0]) == 13.0


def test_radar_dataset_echo_random_crop_contains_echo(tmp_path):
    timestamp = pd.Timestamp("2019-12-16 00:00:00")
    raw = np.zeros((8, 8), dtype=np.int16)
    raw[6:8, 6:8] = 500
    path = tmp_path / "ACHN_CR_20191216_000000.bin"
    _write_radar_file(path, timestamp, raw)

    dataset = RadarDataset(
        data_paths=tmp_path,
        hist_frames=1,
        fcst_frames=1,
        freq=6,
        freq_unit="minutes",
        training=True,
        inference_only=True,
        random_crop_size=(4, 4),
        echo_crop_prob=1.0,
        echo_threshold_dbz=20.0,
    )
    sample = dataset[0]

    assert sample["inputs"].shape == (1, 1, 4, 4)
    assert sample["grid_shape"].tolist() == [4, 4]
    assert "crop_origin" in sample
    assert float(sample["inputs"].max()) == 50.0
