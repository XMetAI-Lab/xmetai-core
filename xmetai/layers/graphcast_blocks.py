"""
Lightweight GraphCast-style blocks for message passing on a structured grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GraphCastLayer", "GraphCastEncoder", "GraphCastDecoder"]


class GraphCastLayer(nn.Module):
    """Single message-passing layer with edge + node updates.

    Args:
        hidden_dim: Node hidden dimension.
        edge_dim: Edge feature dimension.
        dropout: Dropout probability applied after updates.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Run message passing.

        Args:
            x: Node features (B, N, H).
            edge_index: Edge indices (2, E).
            edge_attr: Edge features (E, A).
        """

        if edge_index.dim() != 2:
            raise ValueError("edge_index must be (2, E)")
        if edge_attr.dim() != 2:
            raise ValueError("edge_attr must be (E, A)")

        src, dst = edge_index
        bsz, num_nodes, hidden_dim = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError("Hidden dim mismatch in GraphCastLayer")

        edge_attr = edge_attr.to(dtype=x.dtype, device=x.device)
        edge_feat = self.edge_proj(edge_attr)
        edge_feat = self.edge_norm(edge_feat)

        src_feat = x[:, src]  # (B, E, H)
        dst_feat = x[:, dst]
        edge_feat = edge_feat.unsqueeze(0).expand(bsz, -1, -1)
        msg = self.edge_mlp(torch.cat([src_feat, dst_feat, edge_feat], dim=-1))

        agg = x.new_zeros(bsz, num_nodes, hidden_dim, dtype=msg.dtype)
        deg = x.new_zeros(num_nodes, dtype=msg.dtype)

        ones = torch.ones_like(dst, dtype=deg.dtype)
        deg = deg.scatter_add(0, dst, ones)
        deg = deg.clamp(min=1.0)

        idx = dst.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, hidden_dim)
        agg = agg.scatter_add(1, idx, msg)
        agg = agg / deg.unsqueeze(0).unsqueeze(-1)

        update = self.node_mlp(torch.cat([x, agg], dim=-1))
        update = self.node_norm(update)
        return x + self.dropout(update)


class GraphCastEncoder(nn.Module):
    """Project grid input to node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, static_dim: int = 0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.static_dim = static_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.static_proj = nn.Linear(static_dim, hidden_dim) if static_dim > 0 else None
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, node_static: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(x)
        if self.static_proj is not None and node_static is not None:
            h = h + self.static_proj(node_static)
        h = self.norm(h)
        return h + self.mlp(h)


class GraphCastDecoder(nn.Module):
    """Project node embeddings back to output channels."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))
