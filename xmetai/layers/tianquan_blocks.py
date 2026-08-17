#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Reusable building blocks from the TianQuan S2S weather forecasting model.

These components implement the Perceiver cross-attention architecture (Jaegle et al., 2021)
and the Perceiver Resampler from Flamingo (Alayrac et al., 2022). They are general-purpose
modules for compressing a variable-length context sequence into a fixed set of latent vectors
via cross-attention — a pattern useful beyond weather modeling (e.g., multi-modal fusion,
multi-resolution feature aggregation).

References
----------
- Perceiver: `Perceiver: General Perception with Iterative Attention`_ (Jaegle et al., 2021)
- Flamingo: `Flamingo: a Visual Language Model for Few-Shot Learning`_ (Alayrac et al., 2022)

.. _Perceiver\: General Perception with Iterative Attention:
    https://arxiv.org/abs/2103.03206
.. _Flamingo\: a Visual Language Model for Few-Shot Learning:
    https://arxiv.org/abs/2204.14198
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.layers.mlp import Mlp

__all__ = [
    "PerceiverAttention",
    "PerceiverResampler",
]


class PerceiverAttention(nn.Module):
    """Cross-attention module from the Perceiver architecture.

    Unlike standard cross-attention where query/key/value all come from separate
    projections of the same or different inputs, Perceiver attention uses a **latent**
    (learned or computed query) to attend to a **context** (key/value). The latent
    sequence is typically much shorter than the context sequence, creating an
    information bottleneck that forces the model to extract the most salient features.

    This is the core building block of the Perceiver family of models. It uses
    :func:`torch.nn.functional.scaled_dot_product_attention` (Flash Attention) for
    efficient computation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent (query) features.
    context_dim : int
        Dimensionality of the context (key/value) features.
    head_dim : int, default 64
        Dimensionality of each attention head.
    num_heads : int, default 8
        Number of attention heads. Total inner dimension is ``head_dim * num_heads``.

    Shapes
    ------
    - Input ``latents``: :math:`(B, L_{\\text{latent}}, D_{\\text{latent}})`
    - Input ``x`` (context): :math:`(B, L_{\\text{context}}, D_{\\text{context}})`
    - Output: :math:`(B, L_{\\text{latent}}, D_{\\text{latent}})`

    Examples
    --------
    >>> # Compress 1024 context tokens into 64 latent tokens
    >>> attn = PerceiverAttention(latent_dim=512, context_dim=768, head_dim=64, num_heads=8)
    >>> latents = torch.randn(2, 64, 512)    # (B, L_latent, D_latent)
    >>> context = torch.randn(2, 1024, 768)   # (B, L_context, D_context)
    >>> output = attn(latents, context)       # (2, 64, 512)
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        head_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(latent_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, latent_dim, bias=False)

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Attend latent queries to context key/value pairs.

        Parameters
        ----------
        latents : torch.Tensor
            Latent query features of shape ``(B, L_latent, D_latent)``.
        x : torch.Tensor
            Context features of shape ``(B, L_context, D_context)``.

        Returns
        -------
        torch.Tensor
            Updated latent features of shape ``(B, L_latent, D_latent)``.
        """
        h = self.num_heads

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=h), (q, k, v))

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B H L1 D -> B L1 (H D)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Multi-layer Perceiver Resampler from the Flamingo paper.

    Stacks multiple :class:`PerceiverAttention` layers with feed-forward networks
    and layer normalisation to iteratively refine a fixed-size set of latent vectors
    by attending to a (potentially much larger) context.

    Uses **post-residual-normalisation** (like Swin V2) rather than the pre-norm
    used in the original Perceiver paper. This empirically yields better convergence
    in weather forecasting applications.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent (query) features.
    context_dim : int
        Dimensionality of the context (key/value) features.
    depth : int, default 1
        Number of PerceiverAttention + FFN layers.
    head_dim : int, default 64
        Dimensionality of each attention head.
    num_heads : int, default 16
        Number of attention heads per layer.
    mlp_ratio : float, default 4.0
        Expansion ratio for the feed-forward hidden dimension.
    drop : float, default 0.0
        Dropout rate applied after each FFN layer.
    residual_latent : bool, default True
        If ``True``, add a residual connection from the input latents to the
        attention output. Set to ``False`` when the semantics of the query and
        output are fundamentally different (e.g., modality gap).
    ln_eps : float, default 1e-5
        Epsilon for the internal :class:`~torch.nn.LayerNorm` layers.

    Shapes
    ------
    - Input ``latents``: :math:`(B, L_{\\text{latent}}, D_{\\text{latent}})`
    - Input ``x`` (context): :math:`(B, L_{\\text{context}}, D_{\\text{context}})`
    - Output: :math:`(B, L_{\\text{latent}}, D_{\\text{latent}})`

    Examples
    --------
    >>> # 4-layer resampler: compress a long context into a handful of learned queries
    >>> resampler = PerceiverResampler(
    ...     latent_dim=512, context_dim=1024, depth=4, num_heads=16
    ... )
    >>> queries = torch.randn(2, 32, 512)     # (B, 32 latent tokens)
    >>> context = torch.randn(2, 2048, 1024)   # (B, 2048 context tokens)
    >>> output = resampler(queries, context)   # (2, 32, 512)

    Notes
    -----
    The ``residual_latent=False`` mode follows the HuggingFace Perceiver
    implementation's recommendation for cases where the query space differs
    semantically from the output space.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        residual_latent: bool = True,
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.residual_latent = residual_latent
        self.layers = nn.ModuleList([])
        mlp_hidden_dim = int(latent_dim * mlp_ratio)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            head_dim=head_dim,
                            num_heads=num_heads,
                        ),
                        Mlp(
                            in_features=latent_dim,
                            hidden_features=mlp_hidden_dim,
                            drop=drop,
                        ),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                        nn.LayerNorm(latent_dim, eps=ln_eps),
                    ]
                )
            )

    def forward(self, latents: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Iteratively refine latent vectors through cross-attention layers.

        Parameters
        ----------
        latents : torch.Tensor
            Initial latent vectors of shape ``(B, L_latent, D_latent)``.
        x : torch.Tensor
            Context features of shape ``(B, L_context, D_context)``.

        Returns
        -------
        torch.Tensor
            Refined latent vectors of shape ``(B, L_latent, D_latent)``.
        """
        for attn, ff, ln1, ln2 in self.layers:
            attn_out = ln1(attn(latents, x))
            latents = attn_out + latents if self.residual_latent else attn_out
            latents = ln2(ff(latents)) + latents
        return latents
