"""Unit tests for GraphCast grid utilities."""

import numpy as np
import torch

from xmetai.data.weather.graphcast_grid import build_edge_attr, build_grid_edges, build_node_static


def test_graphcast_grid_edges_shapes():
    height, width = 4, 8
    edge_index = build_grid_edges(height, width, neighbors=8, periodic_lon=True)
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long

    lat = np.deg2rad(np.linspace(90, -90, height, dtype=np.float32))
    lon = np.deg2rad(np.linspace(0, 360, width, dtype=np.float32, endpoint=False))
    edge_attr = build_edge_attr(lat, lon, edge_index)
    assert edge_attr.shape[0] == edge_index.shape[1]
    assert edge_attr.shape[1] == 3


def test_graphcast_node_static_shape():
    height, width = 3, 5
    lat = np.deg2rad(np.linspace(60, -60, height, dtype=np.float32))
    lon = np.deg2rad(np.linspace(0, 360, width, dtype=np.float32, endpoint=False))
    node_static = build_node_static(lat, lon)
    assert node_static.shape[0] == height * width
    assert node_static.shape[1] == 6
