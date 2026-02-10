"""Graph utilities for GraphCast-style grid graphs."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

__all__ = [
    "build_grid_edges",
    "build_edge_attr",
    "build_node_static",
    "grid_latlon",
]


def grid_latlon(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    return lat2d, lon2d


def build_grid_edges(
    height: int,
    width: int,
    neighbors: int = 8,
    periodic_lon: bool = True,
    add_self: bool = False,
) -> torch.Tensor:
    """Build grid adjacency for a 2D lat-lon grid.

    Returns edge_index with shape (2, E).
    """

    if neighbors not in (4, 8):
        raise ValueError("neighbors must be 4 or 8")

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if neighbors == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    edges_src = []
    edges_dst = []
    for i in range(height):
        for j in range(width):
            src = i * width + j
            if add_self:
                edges_src.append(src)
                edges_dst.append(src)
            for di, dj in offsets:
                ni = i + di
                nj = j + dj
                if ni < 0 or ni >= height:
                    continue
                if periodic_lon:
                    nj = nj % width
                elif nj < 0 or nj >= width:
                    continue
                dst = ni * width + nj
                edges_src.append(src)
                edges_dst.append(dst)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index


def _wrap_lon_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + math.pi) % (2 * math.pi) - math.pi


def build_edge_attr(
    lat: np.ndarray,
    lon: np.ndarray,
    edge_index: torch.Tensor,
    include_trig: bool = False,
) -> torch.Tensor:
    """Build edge attributes (dlat, dlon, distance) for each edge.

    If include_trig is True, append sin/cos of dlat and dlon.
    """

    lat2d, lon2d = grid_latlon(lat, lon)
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)

    src, dst = edge_index
    src_lat = lat_flat[src]
    src_lon = lon_flat[src]
    dst_lat = lat_flat[dst]
    dst_lon = lon_flat[dst]

    dlat = dst_lat - src_lat
    dlon = _wrap_lon_delta(dst_lon - src_lon)

    # haversine distance (unit sphere)
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    a = sin_dlat**2 + np.cos(src_lat) * np.cos(dst_lat) * sin_dlon**2
    c = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    dist = c

    feats = [dlat, dlon, dist]
    if include_trig:
        feats.extend([np.sin(dlat), np.cos(dlat), np.sin(dlon), np.cos(dlon)])
    edge_attr = np.stack(feats, axis=-1).astype(np.float32)
    return torch.from_numpy(edge_attr)


def build_node_static(lat: np.ndarray, lon: np.ndarray) -> torch.Tensor:
    """Build node static features: lat, lon, sin/cos lat/lon (radians)."""

    lat2d, lon2d = grid_latlon(lat, lon)
    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)

    feats = [
        lat_flat,
        lon_flat,
        np.sin(lat_flat),
        np.cos(lat_flat),
        np.sin(lon_flat),
        np.cos(lon_flat),
    ]
    node_static = np.stack(feats, axis=-1).astype(np.float32)
    return torch.from_numpy(node_static)
