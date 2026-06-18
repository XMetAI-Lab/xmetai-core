"""CINRAD composite reflectivity binary readers and lightweight QC."""

from __future__ import annotations

import bz2
import json
import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_BOUNDARY_COLOR",
    "DEFAULT_CHINA_BOUNDARY_GEOJSON",
    "RadarGridHeader",
    "RadarQCConfig",
    "RadarRecord",
    "RadarBBox",
    "RADAR_BBOX_CENTRAL_CHINA",
    "RADAR_BBOX_CHINA",
    "RADAR_BBOX_EAST_CHINA",
    "RADAR_BBOX_NORTHWEST_CHINA",
    "RADAR_BBOX_NORTH_CHINA",
    "RADAR_BBOX_NORTHEAST_CHINA",
    "RADAR_BBOX_REGIONS",
    "RADAR_BBOX_SOUTHWEST_CHINA",
    "RADAR_BBOX_SOUTH_CHINA",
    "apply_reflectivity_qc",
    "bbox_to_slices",
    "discover_radar_files",
    "overlay_radar_boundaries",
    "render_radar_rgba",
    "read_radar_header",
    "read_radar_reflectivity",
    "save_radar_png",
]

DEFAULT_CHINA_BOUNDARY_GEOJSON = Path(__file__).resolve().parent / "assets/china_100000_full_wgs84.geojson"
DEFAULT_BOUNDARY_COLOR = (31, 41, 55, 204)
_RADAR_TIME_RE = re.compile(r"(?P<date>\d{8})[_-](?P<time>\d{6})")

RadarBBox = tuple[float, float, float, float]

# Common display/crop regions in (lon_min, lat_min, lon_max, lat_max).
RADAR_BBOX_CHINA: RadarBBox = (70.0, 17.0, 137.0, 55.0)
RADAR_BBOX_NORTH_CHINA: RadarBBox = (97.0, 34.0, 126.5, 54.0)
RADAR_BBOX_NORTHEAST_CHINA: RadarBBox = (118.0, 38.0, 136.0, 54.0)
RADAR_BBOX_EAST_CHINA: RadarBBox = (113.0, 21.5, 125.0, 38.5)
RADAR_BBOX_CENTRAL_CHINA: RadarBBox = (108.0, 24.5, 117.0, 36.5)
RADAR_BBOX_SOUTH_CHINA: RadarBBox = (104.0, 18.0, 118.5, 26.5)
RADAR_BBOX_SOUTHWEST_CHINA: RadarBBox = (78.0, 21.0, 111.0, 37.0)
RADAR_BBOX_NORTHWEST_CHINA: RadarBBox = (73.0, 31.0, 112.0, 50.0)

RADAR_BBOX_REGIONS: dict[str, RadarBBox] = {
    "china": RADAR_BBOX_CHINA,
    "全国": RADAR_BBOX_CHINA,
    "north_china": RADAR_BBOX_NORTH_CHINA,
    "huabei": RADAR_BBOX_NORTH_CHINA,
    "华北": RADAR_BBOX_NORTH_CHINA,
    "northeast_china": RADAR_BBOX_NORTHEAST_CHINA,
    "dongbei": RADAR_BBOX_NORTHEAST_CHINA,
    "东北": RADAR_BBOX_NORTHEAST_CHINA,
    "east_china": RADAR_BBOX_EAST_CHINA,
    "huadong": RADAR_BBOX_EAST_CHINA,
    "华东": RADAR_BBOX_EAST_CHINA,
    "central_china": RADAR_BBOX_CENTRAL_CHINA,
    "huazhong": RADAR_BBOX_CENTRAL_CHINA,
    "华中": RADAR_BBOX_CENTRAL_CHINA,
    "south_china": RADAR_BBOX_SOUTH_CHINA,
    "huanan": RADAR_BBOX_SOUTH_CHINA,
    "华南": RADAR_BBOX_SOUTH_CHINA,
    "southwest_china": RADAR_BBOX_SOUTHWEST_CHINA,
    "xinan": RADAR_BBOX_SOUTHWEST_CHINA,
    "西南": RADAR_BBOX_SOUTHWEST_CHINA,
    "northwest_china": RADAR_BBOX_NORTHWEST_CHINA,
    "xibei": RADAR_BBOX_NORTHWEST_CHINA,
    "西北": RADAR_BBOX_NORTHWEST_CHINA,
}


@dataclass(frozen=True)
class RadarGridHeader:
    """Fixed 256-byte header fields used by ACHN_CR composite reflectivity files."""

    path: Path
    product: str
    product_name: str
    header_size: int
    compressed_size: int
    data_bytes: int
    timestamp: pd.Timestamp | None
    left_lon: float
    top_lat: float
    right_lon: float
    center_lon: float
    center_lat: float
    width: int
    height: int
    lon_step: float
    lat_step: float

    @property
    def bottom_lat(self) -> float:
        return self.top_lat - self.lat_step * (self.height - 1)

    @property
    def lon(self) -> np.ndarray:
        return self.left_lon + np.arange(self.width, dtype=np.float32) * np.float32(self.lon_step)

    @property
    def lat(self) -> np.ndarray:
        return self.top_lat - np.arange(self.height, dtype=np.float32) * np.float32(self.lat_step)


@dataclass(frozen=True)
class RadarRecord:
    time: pd.Timestamp
    path: Path
    header: RadarGridHeader


@dataclass(frozen=True)
class RadarQCConfig:
    """Conservative QC knobs for 2D composite reflectivity grids."""

    enabled: bool = True
    min_dbz: float = 0.0
    max_dbz: float = 75.0
    speckle_dbz: float = 5.0
    speckle_window: int = 3
    speckle_min_neighbors: int = 2
    spike_dbz: float = 15.0
    spike_delta_dbz: float = 25.0
    spike_max_neighbors: int = 2
    fill_value: float = 0.0


def _as_paths(data_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(data_paths, (str, Path)):
        data_paths = [data_paths]
    return [Path(path).expanduser() for path in data_paths]


def _extract_geojson_lines(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if coords is None:
        return []
    if geom_type == "LineString":
        return [[(float(lon), float(lat)) for lon, lat, *_ in coords]]
    if geom_type == "MultiLineString":
        return [[(float(lon), float(lat)) for lon, lat, *_ in line] for line in coords]
    if geom_type == "Polygon":
        return [[(float(lon), float(lat)) for lon, lat, *_ in ring] for ring in coords]
    if geom_type == "MultiPolygon":
        lines = []
        for polygon in coords:
            lines.extend([[(float(lon), float(lat)) for lon, lat, *_ in ring] for ring in polygon])
        return lines
    if geom_type == "GeometryCollection":
        lines = []
        for child in geometry.get("geometries", []):
            lines.extend(_extract_geojson_lines(child))
        return lines
    return []


@lru_cache(maxsize=8)
def _load_geojson_lines(path: str) -> tuple[tuple[tuple[float, float], ...], ...]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)

    lines = []
    geojson_type = geojson.get("type")
    if geojson_type == "FeatureCollection":
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry")
            if geometry is not None:
                lines.extend(_extract_geojson_lines(geometry))
    elif geojson_type == "Feature":
        geometry = geojson.get("geometry")
        if geometry is not None:
            lines.extend(_extract_geojson_lines(geometry))
    else:
        lines.extend(_extract_geojson_lines(geojson))
    return tuple(tuple(line) for line in lines if len(line) >= 2)


def _rgba_color(color: Sequence[int]) -> tuple[int, int, int, int]:
    if len(color) not in (3, 4):
        raise ValueError(f"boundary_color must be RGB or RGBA, got {color}")
    channels = tuple(int(channel) for channel in color)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"boundary_color channels must be in [0, 255], got {color}")
    if len(channels) == 3:
        r, g, b = channels
        return r, g, b, 255
    r, g, b, a = channels
    return r, g, b, a


def _selection_origin(header: RadarGridHeader, bbox: Sequence[float] | None) -> tuple[int, int]:
    if bbox is None:
        return 0, 0
    row_slice, col_slice = bbox_to_slices(header, bbox)
    return int(row_slice.start), int(col_slice.start)


def overlay_radar_boundaries(
    rgba: np.ndarray,
    *,
    bbox: Sequence[float] | None = None,
    header: RadarGridHeader | None = None,
    stride: int = 1,
    boundary_geojson: str | Path = DEFAULT_CHINA_BOUNDARY_GEOJSON,
    boundary_color: Sequence[int] = DEFAULT_BOUNDARY_COLOR,
    boundary_width: int = 1,
) -> np.ndarray:
    """Overlay lon/lat GeoJSON line boundaries on a radar RGBA image."""

    image_array = np.asarray(rgba, dtype=np.uint8)
    if image_array.ndim != 3 or image_array.shape[2] != 4:
        raise ValueError(f"Boundary overlay expects an RGBA image with shape (H, W, 4), got {image_array.shape}")
    if header is None and bbox is None:
        raise ValueError("Boundary overlay requires either a radar header or bbox=(lon_min, lat_min, lon_max, lat_max).")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if boundary_width <= 0:
        raise ValueError(f"boundary_width must be positive, got {boundary_width}")

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("overlay_radar_boundaries requires Pillow to draw administrative boundaries.") from exc

    height, width = image_array.shape[:2]
    if header is not None:
        row_start, col_start = _selection_origin(header, bbox)

        def project(lon: float, lat: float) -> tuple[float, float]:
            x = ((lon - header.left_lon) / header.lon_step - col_start) / stride
            y = ((header.top_lat - lat) / header.lat_step - row_start) / stride
            return x, y

    else:
        if bbox is None or len(bbox) != 4:
            raise ValueError(f"bbox must be (lon_min, lat_min, lon_max, lat_max), got {bbox}")
        lon_min, lat_min, lon_max, lat_max = [float(v) for v in bbox]
        if lon_min >= lon_max or lat_min >= lat_max:
            raise ValueError(f"Invalid bbox bounds: {bbox}")
        x_scale = (width - 1) / (lon_max - lon_min) if width > 1 else 0.0
        y_scale = (height - 1) / (lat_max - lat_min) if height > 1 else 0.0

        def project(lon: float, lat: float) -> tuple[float, float]:
            return (lon - lon_min) * x_scale, (lat_max - lat) * y_scale

    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image, mode="RGBA")
    color = _rgba_color(boundary_color)
    for line in _load_geojson_lines(str(Path(boundary_geojson).expanduser())):
        points = [project(lon, lat) for lon, lat in line]
        if any(-width <= x <= 2 * width and -height <= y <= 2 * height for x, y in points):
            draw.line(points, fill=color, width=boundary_width, joint="curve")
    return np.asarray(image)


def render_radar_rgba(
    data: np.ndarray,
    *,
    bbox: Sequence[float] | None = None,
    boundary_geojson: str | Path | None = None,
    boundary_color: Sequence[int] = DEFAULT_BOUNDARY_COLOR,
    boundary_width: int = 1,
) -> np.ndarray:
    """Render a 2D dBZ field with the shared radar reflectivity color scale."""

    field = np.asarray(data, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"radar PNG rendering expects a 2D field, got shape {field.shape}")

    from xmetai.utils.colormaps import radar_cmap, radar_norm

    rgba = np.asarray(radar_cmap(radar_norm(field), bytes=True), dtype=np.uint8)
    if boundary_geojson is not None:
        rgba = overlay_radar_boundaries(
            rgba,
            bbox=bbox,
            boundary_geojson=boundary_geojson,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )
    return rgba


def save_radar_png(
    data: np.ndarray,
    output_path: str | Path,
    *,
    bbox: Sequence[float] | None = None,
    boundary_geojson: str | Path | None = None,
    boundary_color: Sequence[int] = DEFAULT_BOUNDARY_COLOR,
    boundary_width: int = 1,
) -> Path:
    """Save a 2D dBZ field as a radar-colored PNG."""

    rgba = render_radar_rgba(data, bbox=bbox, boundary_geojson=boundary_geojson, boundary_color=boundary_color, boundary_width=boundary_width)
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("save_radar_png requires Pillow to export PNG files.") from exc
    Image.fromarray(rgba).save(path)
    return path


def _build_qc_config(qc: bool | RadarQCConfig | dict, qc_config: RadarQCConfig | dict | None = None) -> RadarQCConfig:
    if isinstance(qc, RadarQCConfig):
        if qc_config is not None:
            raise ValueError("qc_config cannot be used when qc is already a RadarQCConfig")
        return qc

    params = {}
    if isinstance(qc, dict):
        params.update(qc)
    else:
        params["enabled"] = bool(qc)

    if qc_config is not None:
        params.update(qc_config.__dict__ if isinstance(qc_config, RadarQCConfig) else qc_config)
    return RadarQCConfig(**params)


def _decode_ascii(buf: bytes) -> str:
    return buf.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()


def _read_i32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<i", buf, offset)[0]


def _read_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from("<I", buf, offset)[0]


def _parse_header_time(buf: bytes) -> pd.Timestamp | None:
    year, month, day, hour, minute, second = struct.unpack_from("<6H", buf, 0x64)
    if year <= 0 or month <= 0 or day <= 0:
        return None
    try:
        return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    except ValueError:
        return None


def _parse_filename_time(path: Path) -> pd.Timestamp | None:
    match = _RADAR_TIME_RE.search(path.name)
    if match is None:
        return None
    return pd.to_datetime(f"{match.group('date')}{match.group('time')}", format="%Y%m%d%H%M%S")


def _parse_time_bound(value: object, *, is_end: bool) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value

    text = str(value)
    formats = ["%Y%m%d-%H%M%S", "%Y%m%d-%H%M", "%Y%m%d-%H", "%Y%m%d_%H%M%S", "%Y%m%d%H%M%S", "%Y%m%d"]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(text, format=fmt)
        except ValueError:
            continue
        if fmt == "%Y%m%d" and is_end:
            return parsed + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return parsed
    return pd.Timestamp(value)


def _time_range(years: Sequence[object] | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if years is None:
        return None, None
    if len(years) != 2:
        raise ValueError(f"years must contain start and end values, got {years}")
    return _parse_time_bound(years[0], is_end=False), _parse_time_bound(years[1], is_end=True)


def read_radar_header(path: str | Path) -> RadarGridHeader:
    """Read the 256-byte ACHN_CR header without decompressing the grid payload."""

    radar_path = Path(path).expanduser()
    with radar_path.open("rb") as handle:
        buf = handle.read(256)
    if len(buf) < 256:
        raise ValueError(f"Radar file is shorter than the 256-byte header: {radar_path}")
    if buf[:3] != b"MOC":
        raise ValueError(f"Unsupported radar product magic {buf[:4]!r} in {radar_path}")

    header_size = _read_u32(buf, 0x58) or 256
    width = _read_u32(buf, 0x94)
    height = _read_u32(buf, 0x98)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid radar grid shape ({height}, {width}) in {radar_path}")

    coord_scale = 1000.0
    resolution_scale = 10000.0
    left_lon = _read_i32(buf, 0x80) / coord_scale
    top_lat = _read_i32(buf, 0x84) / coord_scale
    right_lon = _read_i32(buf, 0x88) / coord_scale
    lon_step = _read_u32(buf, 0x9C) / resolution_scale
    lat_step = _read_u32(buf, 0xA0) / resolution_scale
    if lon_step <= 0 and width > 1:
        lon_step = (right_lon - left_lon) / (width - 1)
    if lat_step <= 0:
        lat_step = lon_step

    return RadarGridHeader(
        path=radar_path,
        product=_decode_ascii(buf[0x10:0x18]),
        product_name=_decode_ascii(buf[0x18:0x58]),
        header_size=header_size,
        compressed_size=_read_u32(buf, 0x5C),
        data_bytes=_read_u32(buf, 0xAC),
        timestamp=_parse_header_time(buf),
        left_lon=left_lon,
        top_lat=top_lat,
        right_lon=right_lon,
        center_lon=_read_i32(buf, 0x8C) / coord_scale,
        center_lat=_read_i32(buf, 0x90) / coord_scale,
        width=width,
        height=height,
        lon_step=lon_step,
        lat_step=lat_step,
    )


def bbox_to_slices(header: RadarGridHeader, bbox: Sequence[float]) -> tuple[slice, slice]:
    """Convert ``(lon_min, lat_min, lon_max, lat_max)`` to row/column slices."""

    if len(bbox) != 4:
        raise ValueError(f"bbox must be (lon_min, lat_min, lon_max, lat_max), got {bbox}")
    lon_min, lat_min, lon_max, lat_max = [float(v) for v in bbox]
    if lon_min > lon_max or lat_min > lat_max:
        raise ValueError(f"Invalid bbox bounds: {bbox}")

    col_start = int(np.floor((lon_min - header.left_lon) / header.lon_step))
    col_stop = int(np.ceil((lon_max - header.left_lon) / header.lon_step)) + 1
    row_start = int(np.floor((header.top_lat - lat_max) / header.lat_step))
    row_stop = int(np.ceil((header.top_lat - lat_min) / header.lat_step)) + 1

    row_start = max(row_start, 0)
    col_start = max(col_start, 0)
    row_stop = min(row_stop, header.height)
    col_stop = min(col_stop, header.width)
    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError(f"bbox {bbox} does not overlap radar grid {header.path}")
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _crop_to_slices(crop: Sequence[int]) -> tuple[slice, slice]:
    if len(crop) != 4:
        raise ValueError(f"crop must be (row_start, row_stop, col_start, col_stop), got {crop}")
    row_start, row_stop, col_start, col_stop = [int(v) for v in crop]
    if row_start < 0 or col_start < 0 or row_stop <= row_start or col_stop <= col_start:
        raise ValueError(f"Invalid crop: {crop}")
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _validate_odd_window(name: str, window: int) -> int:
    if window < 3 or window % 2 != 1:
        raise ValueError(f"{name} must be an odd integer >= 3, got {window}")
    return window // 2


def _neighbor_count(mask: np.ndarray, window: int) -> np.ndarray:
    radius = _validate_odd_window("window", window)
    padded = np.pad(mask.astype(np.uint8, copy=False), radius, mode="constant", constant_values=0)
    out_dtype = np.uint16 if window * window > np.iinfo(np.uint8).max else np.uint8
    counts = np.zeros(mask.shape, dtype=out_dtype)
    height, width = mask.shape
    for row_shift in range(window):
        for col_shift in range(window):
            counts += padded[row_shift : row_shift + height, col_shift : col_shift + width]
    return counts


def _neighbor_max(data: np.ndarray, window: int) -> np.ndarray:
    radius = _validate_odd_window("window", window)
    padded = np.pad(data, radius, mode="constant", constant_values=-np.inf)
    neighbor_max = np.full(data.shape, -np.inf, dtype=data.dtype)
    height, width = data.shape
    for row_shift in range(window):
        for col_shift in range(window):
            if row_shift == radius and col_shift == radius:
                continue
            np.maximum(neighbor_max, padded[row_shift : row_shift + height, col_shift : col_shift + width], out=neighbor_max)
    return neighbor_max


def apply_reflectivity_qc(
    data: np.ndarray,
    qc: bool | RadarQCConfig | dict = True,
    *,
    qc_config: RadarQCConfig | dict | None = None,
    static_clutter_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply conservative single-field QC to a 2D composite reflectivity grid."""

    config = _build_qc_config(qc, qc_config)
    if not config.enabled:
        return data

    clean = np.asarray(data, dtype=np.float32).copy()
    invalid = (~np.isfinite(clean)) | (clean < config.min_dbz) | (clean > config.max_dbz)
    clean[invalid] = config.fill_value

    if static_clutter_mask is not None:
        if static_clutter_mask.shape != clean.shape:
            raise ValueError(f"static_clutter_mask shape {static_clutter_mask.shape} does not match data shape {clean.shape}")
        clean[static_clutter_mask] = config.fill_value

    echo = clean >= config.speckle_dbz
    if np.any(echo):
        support = _neighbor_count(echo, config.speckle_window)
        clean[echo & (support < config.speckle_min_neighbors)] = config.fill_value

        if config.spike_delta_dbz > 0:
            echo_after_speckle = clean >= config.speckle_dbz
            local_support = _neighbor_count(echo_after_speckle, config.speckle_window)
            neighbor_source = np.where(clean >= config.min_dbz, clean, -np.inf).astype(np.float32, copy=False)
            local_neighbor_max = _neighbor_max(neighbor_source, config.speckle_window)
            spikes = (clean >= config.spike_dbz) & (clean - local_neighbor_max >= config.spike_delta_dbz) & (local_support <= config.spike_max_neighbors)
            clean[spikes] = config.fill_value

    return clean


def _read_compressed_payload(path: Path, header: RadarGridHeader) -> bytes:
    with path.open("rb") as handle:
        handle.seek(header.header_size)
        compressed = handle.read(header.compressed_size or -1)
    return bz2.decompress(compressed)


def read_radar_reflectivity(
    path: str | Path,
    *,
    header: RadarGridHeader | None = None,
    scale_factor: float = 10.0,
    missing_values: Sequence[int] = (-32768,),
    no_echo_values: Sequence[int] = (-1280,),
    fill_value: float | None = None,
    bbox: Sequence[float] | None = None,
    crop: Sequence[int] | None = None,
    stride: int = 1,
    dtype: np.dtype | type = np.float32,
    qc: bool | RadarQCConfig | dict = False,
    qc_config: RadarQCConfig | dict | None = None,
    static_clutter_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Read one composite reflectivity file as a ``(lat, lon)`` dBZ array."""

    radar_path = Path(path).expanduser()
    header = read_radar_header(radar_path) if header is None else header
    payload = _read_compressed_payload(radar_path, header)
    expected_bytes = header.height * header.width * np.dtype("<i2").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"Unexpected radar payload size for {radar_path}: got {len(payload)}, expected {expected_bytes}")

    raw = np.frombuffer(payload, dtype="<i2").reshape(header.height, header.width)
    if bbox is not None and crop is not None:
        raise ValueError("bbox and crop are mutually exclusive")
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        raw = raw[row_slice, col_slice]
    elif crop is not None:
        row_slice, col_slice = _crop_to_slices(crop)
        raw = raw[row_slice, col_slice]
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if stride > 1:
        raw = raw[::stride, ::stride]

    missing_mask = np.isin(raw, np.asarray(missing_values, dtype=np.int16))
    no_echo_mask = np.isin(raw, np.asarray(no_echo_values, dtype=np.int16))
    special_mask = missing_mask | no_echo_mask

    data = raw.astype(dtype, copy=True)
    data[~special_mask] = data[~special_mask] / scale_factor
    if fill_value is None:
        if len(missing_values) > 0:
            data[missing_mask] = float(missing_values[0])
        if len(no_echo_values) > 0:
            data[no_echo_mask] = float(no_echo_values[0]) / scale_factor
    else:
        data[special_mask] = fill_value
    return apply_reflectivity_qc(data, qc=qc, qc_config=qc_config, static_clutter_mask=static_clutter_mask)


def discover_radar_files(
    data_paths: str | Path | Iterable[str | Path],
    *,
    pattern: str = "*.bin",
    years: Sequence[object] | None = None,
) -> list[RadarRecord]:
    """Find radar files under files/directories and return sorted timestamp records."""

    files: list[Path] = []
    for path in _as_paths(data_paths):
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob(pattern) if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Radar path does not exist: {path}")
    files = sorted(set(files))
    if len(files) == 0:
        raise ValueError(f"No radar files found under {data_paths} with pattern {pattern!r}")

    start_time, end_time = _time_range(years)
    records: list[RadarRecord] = []
    seen: dict[pd.Timestamp, Path] = {}
    for path in files:
        header = read_radar_header(path)
        timestamp = _parse_filename_time(path) or header.timestamp
        if timestamp is None:
            raise ValueError(f"Could not infer timestamp from filename or header: {path}")
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue
        if timestamp in seen:
            raise ValueError(f"Duplicate radar timestamp {timestamp}: {seen[timestamp]} and {path}")
        seen[timestamp] = path
        records.append(RadarRecord(time=timestamp, path=path, header=header))

    records.sort(key=lambda record: record.time)
    if len(records) == 0:
        raise ValueError(f"No radar files remain after time filtering with years={years}")
    return records


def _make_sequence_indices(times: Sequence[pd.Timestamp], total_frames: int, step: pd.Timedelta, interval: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    stride = 1 if interval < 0 else interval
    if stride <= 0:
        raise ValueError(f"interval must be positive or -1 for dense inference, got {interval}")

    indices = []
    for start in range(0, len(times) - total_frames + 1, stride):
        valid = all(times[start + offset + 1] - times[start + offset] == step for offset in range(total_frames - 1))
        if valid:
            indices.append(start)
    return np.asarray(indices, dtype=np.int32)


def _validate_common_grid(records: Sequence[RadarRecord]) -> RadarGridHeader:
    first = records[0].header
    for record in records[1:]:
        header = record.header
        same_grid = (
            header.width == first.width
            and header.height == first.height
            and np.isclose(header.left_lon, first.left_lon)
            and np.isclose(header.top_lat, first.top_lat)
            and np.isclose(header.lon_step, first.lon_step)
            and np.isclose(header.lat_step, first.lat_step)
        )
        if not same_grid:
            raise ValueError(f"Radar grid mismatch between {first.path} and {header.path}")
    return first


def _slice_spatial_array(array: np.ndarray, header: RadarGridHeader, bbox: Sequence[float] | None, crop: Sequence[int] | None, stride: int) -> np.ndarray:
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        array = array[row_slice, col_slice]
    elif crop is not None:
        row_slice, col_slice = _crop_to_slices(crop)
        array = array[row_slice, col_slice]
    if stride > 1:
        array = array[::stride, ::stride]
    return array
