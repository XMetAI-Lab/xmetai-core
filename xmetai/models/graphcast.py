"""GraphCast-style predictor built on BasePredictor."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from xmetai.builder import MODELS
from xmetai.data import (
    build_coarse_mapping,
    build_edge_attr,
    build_grid_edges,
    build_node_static,
    downsample_lat_lon,
)
from xmetai.layers import GraphCastDecoder, GraphCastEncoder, GraphCastLayer, TimestepEmbed
from xmetai.loss import charbonnier_loss

from .base_predictor import BasePredictor

__all__ = ["GraphCastPredictor"]


@MODELS.register()
class GraphCastPredictor(BasePredictor):
    def __init__(
        self,
        *args: Any,
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
        use_multiscale: bool = False,
        coarse_factors: list[int] | None = None,
        num_coarse_layers: int | list[int] = 2,
        **kwargs: Any,
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
        self.use_multiscale = use_multiscale
        self.coarse_factors = coarse_factors
        self.num_coarse_layers = num_coarse_layers
        self._graph_cache: dict[Any, Any] = {}
        self._multiscale_cache: dict[Any, Any] = {}

        in_dim = self.in_chans * self.in_frames
        self.encoder = GraphCastEncoder(in_dim, embed_dim, static_dim=static_dim)
        self.pre_layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_pre_layers)])
        self.layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_layers)])
        self.post_layers = nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(num_post_layers)])
        if self.coarse_factors is None:
            self.coarse_factors = [2]

        if isinstance(self.num_coarse_layers, int):
            layer_counts = [self.num_coarse_layers for _ in self.coarse_factors]
        else:
            if len(self.num_coarse_layers) != len(self.coarse_factors):
                raise ValueError("num_coarse_layers must match coarse_factors length")
            layer_counts = self.num_coarse_layers

        self.coarse_layers = nn.ModuleList([nn.ModuleList([GraphCastLayer(embed_dim, edge_dim, dropout=dropout) for _ in range(n)]) for n in layer_counts])
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

    def _extract_graph(self, data, inputs):
        edge_index = data.get("edge_index")
        edge_attr = data.get("edge_attr")
        node_static = data.get("node_static")
        grid_shape = data.get("grid_shape")

        if edge_index is None or edge_attr is None:
            height = inputs.shape[-2]
            width = inputs.shape[-1]
            base_graph = self._build_graph_from_shape(height, width, inputs.device)
            ms_graph = None
            if self.use_multiscale:
                ms_graph = [self._build_multiscale_from_shape(height, width, factor, inputs.device) for factor in self.coarse_factors]
            return (base_graph[0], base_graph[1], base_graph[2], base_graph[3], ms_graph)

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
        ms_graph = None
        if self.use_multiscale:
            height = inputs.shape[-2]
            width = inputs.shape[-1]
            if grid_shape is not None:
                height = int(grid_shape[0].item())
                width = int(grid_shape[1].item())
            factors = self.coarse_factors or [2]
            ms_graph = [self._build_multiscale_from_shape(height, width, factor, inputs.device) for factor in factors]
        return edge_index, edge_attr, node_static, grid_shape, ms_graph

    def _build_graph_from_shape(self, height: int, width: int, device: Any) -> Any:
        cache_key = (height, width, self.edge_dim, self.static_dim, self.neighbors)
        if cache_key in self._graph_cache:
            edge_index, edge_attr, node_static, grid_shape = self._graph_cache[cache_key]
            return (
                edge_index.to(device=device),
                edge_attr.to(device=device, dtype=self.dtype),
                node_static.to(device=device, dtype=self.dtype),
                grid_shape.to(device=device),
            )

        if hasattr(self, "lat") and hasattr(self, "lon"):
            lat_value = getattr(self, "lat")
            lon_value = getattr(self, "lon")
            lat_tensor = torch.as_tensor(lat_value).detach().cpu()
            lon_tensor = torch.as_tensor(lon_value).detach().cpu()
            lat = np.asarray(lat_tensor.to(torch.float32).numpy(), dtype=np.float32)
            lon = np.asarray(lon_tensor.to(torch.float32).numpy(), dtype=np.float32)
        else:
            lat = np.linspace(90, -90, height, dtype=np.float32)
            lon = np.linspace(0, 360, width, dtype=np.float32, endpoint=False)

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        edge_index = build_grid_edges(height, width, neighbors=self.neighbors, periodic_lon=True)
        include_trig = self.edge_dim == 7
        edge_attr = build_edge_attr(lat_rad, lon_rad, edge_index, include_trig=include_trig)
        node_static = build_node_static(lat_rad, lon_rad)

        if node_static.size(-1) < self.static_dim:
            pad = self.static_dim - node_static.size(-1)
            node_static = torch.cat([node_static, torch.zeros(node_static.size(0), pad)], dim=-1)

        grid_shape = torch.tensor([height, width], dtype=torch.long)
        self._graph_cache[cache_key] = (edge_index, edge_attr, node_static, grid_shape)
        return (
            edge_index.to(device=device),
            edge_attr.to(device=device, dtype=self.dtype),
            node_static.to(device=device, dtype=self.dtype),
            grid_shape.to(device=device),
        )

    def _build_multiscale_from_shape(self, height: int, width: int, factor: int, device: Any) -> Any:
        cache_key = (height, width, self.edge_dim, factor, self.neighbors)
        if cache_key in self._multiscale_cache:
            cached = self._multiscale_cache[cache_key]
            return {
                "edge_index": cached["edge_index"].to(device=device),
                "edge_attr": cached["edge_attr"].to(device=device, dtype=self.dtype),
                "fine_to_coarse": cached["fine_to_coarse"].to(device=device),
                "coarse_counts": cached["coarse_counts"].to(device=device, dtype=self.dtype),
                "coarse_shape": cached["coarse_shape"].to(device=device),
            }

        if hasattr(self, "lat") and hasattr(self, "lon"):
            lat_value = getattr(self, "lat")
            lon_value = getattr(self, "lon")
            lat_tensor = torch.as_tensor(lat_value).detach().cpu()
            lon_tensor = torch.as_tensor(lon_value).detach().cpu()
            lat = np.asarray(lat_tensor.to(torch.float32).numpy(), dtype=np.float32)
            lon = np.asarray(lon_tensor.to(torch.float32).numpy(), dtype=np.float32)
        else:
            lat = np.linspace(90, -90, height, dtype=np.float32)
            lon = np.linspace(0, 360, width, dtype=np.float32, endpoint=False)

        fine_to_coarse, coarse_counts, coarse_h, coarse_w = build_coarse_mapping(height, width, factor)
        coarse_lat, coarse_lon = downsample_lat_lon(lat, lon, factor)
        coarse_lat_rad = np.deg2rad(coarse_lat)
        coarse_lon_rad = np.deg2rad(coarse_lon)

        coarse_edge_index = build_grid_edges(coarse_h, coarse_w, neighbors=self.neighbors, periodic_lon=True)
        include_trig = self.edge_dim == 7
        coarse_edge_attr = build_edge_attr(coarse_lat_rad, coarse_lon_rad, coarse_edge_index, include_trig=include_trig)
        coarse_shape = torch.tensor([coarse_h, coarse_w], dtype=torch.long)

        ms_graph = {
            "edge_index": coarse_edge_index,
            "edge_attr": coarse_edge_attr,
            "fine_to_coarse": fine_to_coarse,
            "coarse_counts": coarse_counts,
            "coarse_shape": coarse_shape,
        }
        self._multiscale_cache[cache_key] = ms_graph
        return {
            "edge_index": coarse_edge_index.to(device=device),
            "edge_attr": coarse_edge_attr.to(device=device, dtype=self.dtype),
            "fine_to_coarse": fine_to_coarse.to(device=device),
            "coarse_counts": coarse_counts.to(device=device, dtype=self.dtype),
            "coarse_shape": coarse_shape.to(device=device),
        }

    def process_data(self, data):
        batch = super().process_data(data)
        if "edge_index" in data:
            batch["edge_index"] = data["edge_index"].to(device=self.device, non_blocking=True)
        if "edge_attr" in data:
            batch["edge_attr"] = data["edge_attr"].to(dtype=self.dtype, device=self.device, non_blocking=True)
        if "node_static" in data:
            batch["node_static"] = data["node_static"].to(dtype=self.dtype, device=self.device, non_blocking=True)
        if "grid_shape" in data:
            batch["grid_shape"] = data["grid_shape"].to(device=self.device, non_blocking=True)
        return batch

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        inputs = data["input"]
        times = data["times"]
        targets = data["targets"]
        graph = self._extract_graph(data, inputs)

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames
        loss_rec = 0
        pinn_acc: Dict[str, torch.Tensor] = {}

        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k : k + self.out_frames]
            self.reset_input(inputs)
            output = self.forward_step(inputs, times, graph, step=t)
            loss_rec += charbonnier_loss(output, target, weight=getattr(self, "weight", None))
            if self.pinn_enable:
                pinn_losses = self.compute_pinn_losses(output, target)
                for k, v in pinn_losses.items():
                    pinn_acc[k] = pinn_acc.get(k, 0) + v
            inputs = torch.cat([inputs, output], dim=1)[:, -self.in_frames :]

        loss_dict = dict(loss_rec=loss_rec / future_frames)
        if len(pinn_acc) > 0:
            for k, v in pinn_acc.items():
                loss_dict[k] = v / future_frames
        return loss_dict

    def forward_step(self, inputs: Any, times: Any, graph: Any, step: int, timestamp: dict | None = None) -> Any:
        if len(graph) == 4:
            edge_index, edge_attr, node_static, grid_shape = graph
            ms_graph = None
        else:
            edge_index, edge_attr, node_static, grid_shape, ms_graph = graph
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
            if timestamp is None:
                timestamp = self.get_timestamp(inputs, times, step)
            time_token = self.step_embed(timestamp["step"]) + self.hour_embed(timestamp["hour"]) + self.doy_embed(timestamp["doy"])
            h = h + self.time_mlp(time_token).unsqueeze(1)
        for layer in self.pre_layers:
            h = layer(h, edge_index, edge_attr)
        if ms_graph is not None and len(self.coarse_layers) > 0:
            for level, coarse_layers in enumerate(list(self.coarse_layers)):
                if level >= len(ms_graph):
                    break
                level_graph = ms_graph[level]
                fine_to_coarse = level_graph["fine_to_coarse"]
                coarse_counts = level_graph["coarse_counts"]
                coarse_edge_index = level_graph["edge_index"]
                coarse_edge_attr = level_graph["edge_attr"]

                num_coarse = coarse_counts.size(0)
                idx = fine_to_coarse.unsqueeze(0).unsqueeze(-1).expand(h.size(0), -1, h.size(-1))
                coarse_nodes = h.new_zeros(h.size(0), num_coarse, h.size(-1))
                coarse_nodes.scatter_add_(1, idx, h)
                coarse_nodes = coarse_nodes / coarse_counts.unsqueeze(0).unsqueeze(-1)

                for layer in coarse_layers:  # type: ignore[attr-defined]
                    coarse_nodes = layer(coarse_nodes, coarse_edge_index, coarse_edge_attr)

                h = h + coarse_nodes[:, fine_to_coarse]
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

        graph = self._extract_graph(data, inputs)
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

    @torch.no_grad()
    def export_onnx(self, inputs, step, hour, doy):
        normed_input = self.normalize(inputs)
        normed_input = normed_input.to(dtype=self.dtype)
        step = step.to(dtype=self.dtype)
        hour = hour.to(dtype=self.dtype)
        doy = doy.to(dtype=self.dtype)

        self.reset_input(normed_input)
        batch_size = inputs.shape[0]
        timestamp = {
            "step": step.repeat(batch_size),
            "hour": hour.repeat(batch_size),
            "doy": doy.repeat(batch_size),
        }

        height = normed_input.shape[-2]
        width = normed_input.shape[-1]
        graph = self._build_graph_from_shape(height, width, normed_input.device)
        if self.use_multiscale:
            factors = self.coarse_factors or [2]
            ms_graph = [self._build_multiscale_from_shape(height, width, factor, normed_input.device) for factor in factors]
            graph = (*graph, ms_graph)
        output = self.forward_step(normed_input, None, graph, step=0, timestamp=timestamp)
        output = self.inv_normalize(output)
        return output
