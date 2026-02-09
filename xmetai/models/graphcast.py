"""GraphCast-style predictor built on BasePredictor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from xmetai.builder import MODELS
from xmetai.layers import TimestepEmbed
from xmetai.layers.graphcast_blocks import GraphCastDecoder, GraphCastEncoder, GraphCastLayer
from xmetai.loss import charbonnier_loss

from .base_predictor import BasePredictor

__all__ = ["GraphCastPredictor"]


@MODELS.register()
class GraphCastPredictor(BasePredictor):
    def __init__(
        self,
        *args,
        embed_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        neighbors: int = 8,
        edge_dim: int = 3,
        static_dim: int = 6,
        image_size: tuple[int, int] = (121, 240),
        time_embed: bool = True,
        num_pre_layers: int = 0,
        num_post_layers: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.neighbors = neighbors
        self.edge_dim = edge_dim
        self.static_dim = static_dim
        self.image_size = tuple(image_size)
        self.time_embed = time_embed
        self.num_pre_layers = num_pre_layers
        self.num_post_layers = num_post_layers

        in_dim = self.in_chans * self.in_frames
        self.encoder = GraphCastEncoder(in_dim, embed_dim, static_dim=static_dim)
        self.pre_layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_pre_layers)])
        self.layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_layers)])
        self.post_layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_post_layers)])
        self.decoder = GraphCastDecoder(embed_dim, self.out_chans)
        if self.time_embed:
            self.step_embed = TimestepEmbed(embed_dim)
            self.hour_embed = TimestepEmbed(embed_dim)
            self.doy_embed = TimestepEmbed(embed_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
            )

    def _extract_graph(self, data):
        edge_index = data.get("edge_index")
        edge_attr = data.get("edge_attr")
        node_static = data.get("node_static")
        grid_shape = data.get("grid_shape")

        if edge_index is None or edge_attr is None:
            raise ValueError("GraphCastPredictor requires edge_index and edge_attr in batch data.")

        if edge_index.dim() == 3:
            edge_index = edge_index[0]
        if edge_attr.dim() == 3:
            edge_attr = edge_attr[0]
        if node_static is not None and node_static.dim() == 3:
            node_static = node_static[0]
        if grid_shape is not None and grid_shape.dim() == 2:
            grid_shape = grid_shape[0]

        edge_index = edge_index.to(device=self.device)
        edge_attr = edge_attr.to(device=self.device, dtype=self.dtype)
        if edge_attr.size(-1) != self.edge_dim:
            raise ValueError(f"edge_attr dim mismatch: expected {self.edge_dim}, got {edge_attr.size(-1)}")
        if node_static is not None:
            node_static = node_static.to(device=self.device, dtype=self.dtype)
        if grid_shape is not None:
            grid_shape = grid_shape.to(device=self.device)
        return edge_index, edge_attr, node_static, grid_shape

    def process_data(self, data):
        batch = super().process_data(data)
        for key in ("edge_index", "edge_attr", "node_static", "grid_shape"):
            if key in data:
                batch[key] = data[key]
        return batch

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        inputs = data["input"]
        times = data["times"]
        targets = data["targets"]
        graph = self._extract_graph(data)

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames
        loss_rec = 0

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            self.reset_input(inputs)
            output = self.forward_step(inputs, times, graph, step=t)
            loss_rec += charbonnier_loss(output, target)
            inputs = torch.cat([inputs, output], dim=1)[:, -self.in_frames :]

        loss_dict = dict(loss_rec=loss_rec / future_frames)
        return loss_dict

    def forward_step(self, inputs, times, graph, step: int):
        edge_index, edge_attr, node_static, grid_shape = graph
        bsz, _, _, height, width = inputs.shape
        if grid_shape is not None:
            height = int(grid_shape[0].item())
            width = int(grid_shape[1].item())

        x = inputs[:, -self.in_frames :]
        x = x.reshape(bsz, self.in_frames * self.in_chans, height, width)
        x = x.permute(0, 2, 3, 1).reshape(bsz, height * width, -1)

        if node_static is not None:
            node_static = node_static.unsqueeze(0).expand(bsz, -1, -1)
        h = self.encoder(x, node_static)
        if self.time_embed:
            timestamp = self.get_timestamp(inputs, times, step)
            time_token = self.step_embed(timestamp["step"]) + self.hour_embed(timestamp["hour"]) + self.doy_embed(timestamp["doy"])
            h = h + self.time_mlp(time_token).unsqueeze(1)
        for layer in self.pre_layers:
            h = layer(h, edge_index, edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        for layer in self.post_layers:
            h = layer(h, edge_index, edge_attr)
        out = self.decoder(h)
        out = out.reshape(bsz, height, width, self.out_chans).permute(0, 3, 1, 2)
        out = out.unsqueeze(1)
        if self.out_frames > 1:
            out = out.repeat(1, self.out_frames, 1, 1, 1)
        out = out + inputs[:, -1:].to(out)
        return out

    @torch.no_grad()
    def test_cascade(self, data):
        inputs = data["input"]
        times = data["times"]
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames
        assert sum(self.step_range) <= total_step

        chw = list(inputs.shape[-3:])
        outputs = inputs.new_zeros(inputs.shape[0], len(steps), *chw)

        graph = self._extract_graph(data)
        for num_step in self.step_range:
            curr_step = 0
            for step in range(num_step):
                self.reset_input(inputs)
                output = self.forward_step(inputs, times, graph, step=step + self.start_frames)
                inputs = torch.cat([inputs, output], dim=1)[:, -self.in_frames :]
                if step + 1 in steps:
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                    curr_step += 1

        return outputs
