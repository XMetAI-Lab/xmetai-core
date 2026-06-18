from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["radar_cmap", "radar_norm", "radar_colors", "radar_norms"]

ArrayLike = Any


precip_colors = [
    "#04e9e7",  # 0.01 - 0.10 inches
    "#019ff4",  # 0.10 - 0.25 inches
    "#0300f4",  # 0.25 - 0.50 inches
    "#02fd02",  # 0.50 - 0.75 inches
    "#01c501",  # 0.75 - 1.00 inches
    "#008e00",  # 1.00 - 1.50 inches
    "#fdf802",  # 1.50 - 2.00 inches
    "#e5bc00",  # 2.00 - 2.50 inches
    "#fd9500",  # 2.50 - 3.00 inches
    "#fd0000",  # 3.00 - 4.00 inches
    "#d40000",  # 4.00 - 5.00 inches
    "#bc0000",  # 5.00 - 6.00 inches
    "#f800fd",  # 6.00 - 8.00 inches
    "#9854c6",  # 8.00 - 10.00 inches
    "#fdfdfd",
]  # 10.00+

radar_colors = [
    "#FFFFFF",  # 0
    "#808080",  # 0-5
    "#00FFFF",  # 5-10
    "#00BFFF",  # 10-15
    "#0000FF",  # 15-20
    "#00FF00",  # 20-25
    "#32CD32",  # 25-30
    "#008000",  # 30-35
    "#FFFF00",  # 35-40
    "#DAA520",  # 40-45
    "#FFA500",  # 45-50
    "#FF0000",  # 50-55
    "#8B0000",  # 55-60
    "#800000",  # 60-65
    "#FF00FF",  # 65-70
    "#8A2BE2",  # 70-75
    "#FFFFFF",
]  # 75+

radar_norms = [
    -32768,
    0,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    32767,
]


snow_colors = [
    "#FFFFFF",
    "#C2BFFF",
    "#867FFF",
    "#493FFF",
    "#0D00FF",
    "#0C3FBF",
    "#0B7F7F",
    "#0ABF3F",
    "#09FF00",
    "#46E800",
    "#84D200",
    "#C1BC00",
    "#FFA600",
    "#FF7C00",
    "#FF5300",
    "#FF2900",
    "#FF0000",
    "#FF004E",
    "#FF009C",
    "#FF00EA",
]

hail_colors = [
    "#FFFFFF",
    "#AEAAFF",
    "#5D55FF",
    "#0D00FF",
    "#0B55AA",
    "#0AAA55",
    "#09FF00",
    "#5BE100",
    "#ADC300",
    "#FFA600",
    "#FF6E00",
    "#FF3700",
    "#FF0000",
    "#FF0075",
    "#FF00EA",
]


def _hex_to_rgba(color: str) -> tuple[int, int, int, int]:
    text = color.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), 255


class _RadarNorm:
    """Discrete bin mapper for radar display."""

    def __init__(self, boundaries: Sequence[float], ncolors: int, clip: bool = True) -> None:
        if len(boundaries) < 2:
            raise ValueError("Radar norm expects at least two boundaries.")
        if ncolors <= 0:
            raise ValueError(f"ncolors must be positive, got {ncolors}")
        self.boundaries = np.asarray(boundaries, dtype=np.float32)
        self.ncolors = int(ncolors)
        self.clip = bool(clip)

    def __call__(self, data: ArrayLike) -> np.ndarray:
        values = np.asarray(data, dtype=np.float32)
        indices = np.searchsorted(self.boundaries, values, side="right") - 1
        if self.clip:
            indices = np.clip(indices, 0, self.ncolors - 1)
        return indices.astype(np.int16, copy=False)


class _RadarColorMap:
    """Map radar bin indices to RGB/RGBA arrays."""

    def __init__(self, colors: Sequence[str]) -> None:
        if len(colors) == 0:
            raise ValueError("Radar color map expects at least one color.")
        self.colors = colors
        self._rgba = np.asarray([_hex_to_rgba(color) for color in colors], dtype=np.uint8)

    def __call__(self, values: ArrayLike, bytes: bool = False) -> np.ndarray:
        indices = np.asarray(values)
        if not np.issubdtype(indices.dtype, np.integer):
            indices = np.floor(indices.astype(np.float32) * len(self._rgba)).astype(np.int16)
        indices = np.clip(indices, 0, len(self._rgba) - 1)
        rgba = self._rgba[indices]
        if bytes:
            return rgba
        return rgba.astype(np.float32) / 255.0


radar_cmap = _RadarColorMap(radar_colors)
radar_norm = _RadarNorm(radar_norms, ncolors=len(radar_colors), clip=True)
