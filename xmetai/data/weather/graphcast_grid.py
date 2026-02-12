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
    "build_coarse_mapping",
    "downsample_lat_lon",
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


def downsample_lat_lon(lat: np.ndarray, lon: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample lat/lon by averaging blocks of size factor."""

    if factor <= 1:
        return lat, lon
    height = len(lat)
    width = len(lon)
    coarse_h = math.ceil(height / factor)
    coarse_w = math.ceil(width / factor)

    coarse_lat = np.zeros(coarse_h, dtype=np.float32)
    coarse_lon = np.zeros(coarse_w, dtype=np.float32)

    for i in range(coarse_h):
        start = i * factor
        end = min((i + 1) * factor, height)
        coarse_lat[i] = lat[start:end].mean()

    for j in range(coarse_w):
        start = j * factor
        end = min((j + 1) * factor, width)
        coarse_lon[j] = lon[start:end].mean()

    return coarse_lat, coarse_lon


def build_coarse_mapping(height: int, width: int, factor: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Map fine grid nodes to coarse grid indices.

    Returns:
        fine_to_coarse: (N_f,) long tensor
        coarse_counts: (N_c,) float tensor
        coarse_h: int
        coarse_w: int
    """

    if factor <= 1:
        coarse_h = height
        coarse_w = width
    else:
        coarse_h = math.ceil(height / factor)
        coarse_w = math.ceil(width / factor)

    fine_to_coarse = torch.zeros(height * width, dtype=torch.long)
    for i in range(height):
        for j in range(width):
            ch = min(i // factor, coarse_h - 1)
            cw = min(j // factor, coarse_w - 1)
            fine_to_coarse[i * width + j] = ch * coarse_w + cw

    coarse_counts = torch.bincount(fine_to_coarse, minlength=coarse_h * coarse_w).float()
    coarse_counts = coarse_counts.clamp(min=1.0)
    return fine_to_coarse, coarse_counts, coarse_h, coarse_w
