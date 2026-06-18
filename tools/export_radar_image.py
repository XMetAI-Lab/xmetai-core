#!/usr/bin/env python
"""Export CINRAD composite reflectivity binaries to radar-colored PNG/GIF images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import ImageColor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xmetai.utils.radar import (  # noqa: E402
    DEFAULT_CHINA_BOUNDARY_GEOJSON,
    RadarQCConfig,
    bbox_to_slices,
    discover_radar_files,
    overlay_radar_boundaries,
    read_radar_header,
    read_radar_reflectivity,
    render_radar_rgba,
)

logger = logging.getLogger(__name__)


def _parse_bbox(values: Sequence[str] | None) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--bbox expects four values: lon_min lat_min lon_max lat_max")
    return tuple(float(value) for value in values)


def _parse_color(value: str) -> tuple[int, int, int, int]:
    color = ImageColor.getcolor(value, "RGBA")
    return tuple(int(channel) for channel in color)


def _output_path(input_path: Path, output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        return output
    directory = output_dir if output_dir is not None else input_path.parent
    return directory / f"{input_path.stem}.png"


def _import_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("tools/export_radar_image.py requires Pillow to export animated GIF files.") from exc
    return Image


def _load_clutter_mask(mask_path: Path | None, header, bbox: Sequence[float] | None, stride: int) -> np.ndarray | None:
    if mask_path is None:
        return None
    mask = np.load(mask_path).astype(bool, copy=False)
    expected_shape = (header.height, header.width)
    if mask.shape != expected_shape:
        raise ValueError(f"Clutter mask {mask_path} shape {mask.shape} does not match radar grid {expected_shape}")
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        mask = mask[row_slice, col_slice]
    if stride > 1:
        mask = mask[::stride, ::stride]
    return mask


def export_one(
    path: Path,
    output_path: Path,
    *,
    bbox: Sequence[float] | None,
    stride: int,
    fill_value: float | None,
    qc_config: RadarQCConfig,
    clutter_mask_path: Path | None,
    boundary_geojson: Path | None,
    boundary_color: tuple[int, int, int, int],
    boundary_width: int,
) -> None:
    header = read_radar_header(path)
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        logger.info("Cropping %s to rows=%s cols=%s", path.name, row_slice, col_slice)
    clutter_mask = _load_clutter_mask(clutter_mask_path, header, bbox, stride)
    data = read_radar_reflectivity(path, header=header, bbox=bbox, stride=stride, fill_value=fill_value, qc=qc_config, static_clutter_mask=clutter_mask)
    rgba = render_radar_rgba(data)
    if boundary_geojson is not None:
        rgba = overlay_radar_boundaries(
            rgba,
            header=header,
            bbox=bbox,
            stride=stride,
            boundary_geojson=boundary_geojson,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _import_pillow_image().fromarray(rgba).save(output_path)
    logger.info("Wrote %s", output_path)


def _read_frame_rgba(
    path: Path,
    *,
    bbox: Sequence[float] | None,
    stride: int,
    fill_value: float | None,
    qc_config: RadarQCConfig,
    clutter_mask_path: Path | None,
    boundary_geojson: Path | None,
    boundary_color: tuple[int, int, int, int],
    boundary_width: int,
) -> np.ndarray:
    header = read_radar_header(path)
    if bbox is not None:
        row_slice, col_slice = bbox_to_slices(header, bbox)
        logger.debug("Cropping %s to rows=%s cols=%s", path.name, row_slice, col_slice)
    clutter_mask = _load_clutter_mask(clutter_mask_path, header, bbox, stride)
    data = read_radar_reflectivity(path, header=header, bbox=bbox, stride=stride, fill_value=fill_value, qc=qc_config, static_clutter_mask=clutter_mask)
    rgba = render_radar_rgba(data)
    if boundary_geojson is None:
        return rgba
    return overlay_radar_boundaries(
        rgba,
        header=header,
        bbox=bbox,
        stride=stride,
        boundary_geojson=boundary_geojson,
        boundary_color=boundary_color,
        boundary_width=boundary_width,
    )


def _gif_output_path(records, output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        return output
    directory = output_dir if output_dir is not None else Path.cwd()
    start = records[0].time.strftime("%Y%m%d_%H%M%S")
    end = records[-1].time.strftime("%Y%m%d_%H%M%S")
    return directory / f"radar_{start}_{end}.gif"


def export_gif(
    records,
    output_path: Path,
    *,
    bbox: Sequence[float] | None,
    stride: int,
    fill_value: float | None,
    qc_config: RadarQCConfig,
    clutter_mask_path: Path | None,
    boundary_geojson: Path | None,
    boundary_color: tuple[int, int, int, int],
    boundary_width: int,
    duration_ms: int,
    loop: int,
) -> None:
    if len(records) == 0:
        raise ValueError("No radar records selected for GIF export.")
    if duration_ms <= 0:
        raise ValueError(f"GIF frame duration must be positive, got {duration_ms}")

    Image = _import_pillow_image()
    frames = []
    for i, record in enumerate(records):
        rgba = _read_frame_rgba(
            record.path,
            bbox=bbox,
            stride=stride,
            fill_value=fill_value,
            qc_config=qc_config,
            clutter_mask_path=clutter_mask_path,
            boundary_geojson=boundary_geojson,
            boundary_color=boundary_color,
            boundary_width=boundary_width,
        )
        frames.append(Image.fromarray(rgba).convert("P", palette=Image.Palette.ADAPTIVE))
        logger.info("Loaded GIF frame %d/%d: %s", i + 1, len(records), record.path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=loop, optimize=True)
    logger.info("Wrote GIF %s with %d frames", output_path, len(frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Radar .bin file or directory containing radar files.")
    parser.add_argument("-o", "--output", help="Output PNG path for single-file input, or GIF path for directory/time-range input.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to each input file's directory.")
    parser.add_argument("--pattern", default="*.bin", help="Directory search pattern. Default: %(default)s")
    parser.add_argument("--years", nargs=2, metavar=("START", "END"), help="Optional inclusive time range, e.g. 20191216 20191216.")
    parser.add_argument("--bbox", nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"), help="Optional lon/lat crop.")
    parser.add_argument("--stride", type=int, default=1, help="Spatial downsampling stride for quick previews.")
    parser.add_argument("--fill-value", type=float, default=None, help="Replace missing/no-echo pixels before coloring. Default keeps them white.")
    parser.add_argument("--no-qc", action="store_true", help="Disable conservative reflectivity QC and export the raw decoded field.")
    parser.add_argument("--qc-min-dbz", type=float, default=0.0, help="Minimum valid reflectivity after QC.")
    parser.add_argument("--qc-max-dbz", type=float, default=75.0, help="Maximum valid reflectivity after QC.")
    parser.add_argument("--qc-speckle-dbz", type=float, default=5.0, help="Reflectivity threshold used for speckle support checks.")
    parser.add_argument("--qc-speckle-min-neighbors", type=int, default=2, help="Minimum echo pixels in the local QC window, including the center pixel.")
    parser.add_argument("--qc-spike-delta-dbz", type=float, default=25.0, help="Remove isolated spikes this many dBZ above all neighboring pixels.")
    parser.add_argument("--clutter-mask", help="Optional .npy boolean mask for known static clutter pixels on the full radar grid.")
    parser.add_argument("--china-boundary", action="store_true", help="Overlay the bundled WGS-84 China provincial administrative boundary GeoJSON.")
    parser.add_argument("--boundary-geojson", help="Optional GeoJSON administrative boundary file to overlay in lon/lat coordinates.")
    parser.add_argument("--boundary-color", default="#1f2937cc", help="Boundary line color accepted by Pillow, e.g. '#1f2937cc' or 'black'.")
    parser.add_argument("--boundary-width", type=int, default=1, help="Boundary line width in output pixels.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to export in directory mode.")
    parser.add_argument("--gif", action="store_true", help="Export selected directory records as one animated GIF.")
    parser.add_argument("--png-frames", action="store_true", help="Write individual PNG frames instead of auto GIF for directory/time-range input.")
    parser.add_argument("--gif-duration-ms", type=int, default=200, help="GIF frame duration in milliseconds.")
    parser.add_argument("--gif-loop", type=int, default=0, help="GIF loop count. 0 means loop forever.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    input_path = Path(args.input).expanduser()
    output = Path(args.output).expanduser() if args.output is not None else None
    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else None
    bbox = _parse_bbox(args.bbox)
    clutter_mask_path = Path(args.clutter_mask).expanduser() if args.clutter_mask is not None else None
    if args.china_boundary and args.boundary_geojson is not None:
        raise ValueError("--china-boundary and --boundary-geojson are mutually exclusive")
    boundary_geojson = DEFAULT_CHINA_BOUNDARY_GEOJSON if args.china_boundary else None
    if args.boundary_geojson is not None:
        boundary_geojson = Path(args.boundary_geojson).expanduser()
    boundary_color = _parse_color(args.boundary_color)
    qc_config = RadarQCConfig(
        enabled=not args.no_qc,
        min_dbz=args.qc_min_dbz,
        max_dbz=args.qc_max_dbz,
        speckle_dbz=args.qc_speckle_dbz,
        speckle_min_neighbors=args.qc_speckle_min_neighbors,
        spike_delta_dbz=args.qc_spike_delta_dbz,
        fill_value=args.fill_value if args.fill_value is not None else -128.0,
    )

    if input_path.is_dir():
        records = discover_radar_files(input_path, pattern=args.pattern, years=args.years)
        if args.limit is not None:
            records = records[: args.limit]
        make_gif = not args.png_frames and (args.gif or args.years is not None or (output is not None and output.suffix.lower() == ".gif"))
        if make_gif:
            export_gif(
                records,
                _gif_output_path(records, output, output_dir),
                bbox=bbox,
                stride=args.stride,
                fill_value=args.fill_value,
                qc_config=qc_config,
                clutter_mask_path=clutter_mask_path,
                boundary_geojson=boundary_geojson,
                boundary_color=boundary_color,
                boundary_width=args.boundary_width,
                duration_ms=args.gif_duration_ms,
                loop=args.gif_loop,
            )
        else:
            if output is not None:
                raise ValueError("--output can only be used with directory input when exporting a GIF. Use --output-dir for PNG frames.")
            for record in records:
                export_one(
                    record.path,
                    _output_path(record.path, None, output_dir),
                    bbox=bbox,
                    stride=args.stride,
                    fill_value=args.fill_value,
                    qc_config=qc_config,
                    clutter_mask_path=clutter_mask_path,
                    boundary_geojson=boundary_geojson,
                    boundary_color=boundary_color,
                    boundary_width=args.boundary_width,
                )
    else:
        export_one(
            input_path,
            _output_path(input_path, output, output_dir),
            bbox=bbox,
            stride=args.stride,
            fill_value=args.fill_value,
            qc_config=qc_config,
            clutter_mask_path=clutter_mask_path,
            boundary_geojson=boundary_geojson,
            boundary_color=boundary_color,
            boundary_width=args.boundary_width,
        )


if __name__ == "__main__":
    main()
