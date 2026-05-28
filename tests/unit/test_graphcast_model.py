"""Unit tests for GraphCastPredictor."""

import numpy as np
import torch

from xmetai.data import build_edge_attr, build_grid_edges, build_node_static
from xmetai.models import GraphCastPredictor


def _make_graph(height, width):
    lat = np.deg2rad(np.linspace(60, -60, height, dtype=np.float32))
    lon = np.deg2rad(np.linspace(0, 360, width, dtype=np.float32, endpoint=False))
    edge_index = build_grid_edges(height, width, neighbors=4, periodic_lon=True)
    edge_attr = build_edge_attr(lat, lon, edge_index)
    node_static = build_node_static(lat, lon)
    grid_shape = torch.tensor([height, width])
    return edge_index, edge_attr, node_static, grid_shape


def test_graphcast_forward_loss():
    torch.manual_seed(0)
    bsz, t_in, chans, height, width = 2, 2, 3, 4, 5
    inputs = torch.randn(bsz, t_in, chans, height, width)
    targets = torch.randn(bsz, 1, chans, height, width)
    times = torch.tensor([946684800000000000, 946684800000000000])

    edge_index, edge_attr, node_static, grid_shape = _make_graph(height, width)

    model = GraphCastPredictor(
        in_chans=chans,
        out_chans=chans,
        in_frames=t_in,
        out_frames=1,
        embed_dim=32,
        num_layers=2,
        edge_dim=3,
        static_dim=6,
        device="cpu",
        precision="fp32",
        test_frames=[1],
        test_chans=[0, 1, 2],
        test_names=["a", "b", "c"],
        step_range=[1],
    )
    model.train()

    loss = model(
        {
            "inputs": inputs,
            "targets": targets,
            "times": times,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_static": node_static,
            "grid_shape": grid_shape,
        }
    )

    assert "loss_rec" in loss


def test_graphcast_test_cascade():
    bsz, t_in, chans, height, width = 1, 2, 2, 4, 4
    inputs = torch.randn(bsz, t_in, chans, height, width)
    times = torch.tensor([946684800000000000])

    edge_index, edge_attr, node_static, grid_shape = _make_graph(height, width)

    model = GraphCastPredictor(
        in_chans=chans,
        out_chans=chans,
        in_frames=t_in,
        out_frames=1,
        embed_dim=32,
        num_layers=1,
        edge_dim=3,
        static_dim=6,
        device="cpu",
        precision="fp32",
        test_frames=[1, 2],
        test_chans=[0, 1],
        test_names=["a", "b"],
        step_range=[2],
    )
    model.eval()

    data = {
        "inputs": inputs,
        "times": times,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_static": node_static,
        "grid_shape": grid_shape,
    }
    data = model.process_data(data)
    out = model.test_cascade(data)
    assert out.shape[1] == len(model.test_frames)
    assert out.shape[-2:] == (height, width)
