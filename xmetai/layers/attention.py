import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import einsum

from .helpers import default, exists, zero_module

if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
    from torch.backends.cuda import SDPBackend, sdp_kernel

    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,
            "enable_flash": False,
            "enable_mem_efficient": False,
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,
            "enable_flash": True,
            "enable_mem_efficient": False,
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,
            "enable_flash": False,
            "enable_mem_efficient": True,
        },
        None: {
            "enable_math": True,
            "enable_flash": True,
            "enable_mem_efficient": True},
    }
else:
    from contextlib import nullcontext

    SDP_IS_AVAILABLE = False
    sdp_kernel = nullcontext
    BACKEND_MAP = {}


__all__ = ["AttentionBlock", "TwoAttentionBlock", "TransformerBlock"]


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)
        inner_dim = mult * ((inner_dim + mult - 1) // mult)
        dim_out = default(dim_out, dim)

        self.fc1 = nn.Linear(dim, inner_dim * 2)
        self.act = GEGLU()
        self.fc2 = nn.Linear(inner_dim, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class NewCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.backend = backend

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(
                t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # new
        with sdp_kernel(**BACKEND_MAP[self.backend]):
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )  # scale is dim_head ** -0.5 per default

        del q, k, v

        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
        precision="fp32",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.precision = precision

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(
                t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if self.precision == "fp32":
            with torch.autocast(enabled=False, device_type="cuda"):
                q, k = q.float(), k.float()
                sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        else:
            sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        # del q, k

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1).to(v)

        out = einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        n_heads=8,
        dropout=0.0,
        context_dim=None,
        sdp_backend=None,
    ):
        super().__init__()

        self.attn = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            backend=sdp_backend,
        )
        self.ffn = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn(self.norm1(x), context=context) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class TwoAttentionBlock(AttentionBlock):
    def __init__(
        self,
        dim,
        dim_head,
        n_heads=8,
        dropout=0.0,
        context_dim=None,
        sdp_backend=None,
    ):
        super().__init__(
            dim,
            dim_head,
            n_heads,
            dropout,
            context_dim=None,
            sdp_backend=sdp_backend)
        self.attn2 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            backend=sdp_backend,
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ffn(self.norm3(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads=8,
        depth=1,
        dropout=0.0,
        proj_type="linear",
        context_dim=None,
        sdp_backend=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proj_type = proj_type

        d_head = in_channels // n_heads
        inner_dim = n_heads * d_head

        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )

        if proj_type == "linear":
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )

        self.transformer_blocks = nn.ModuleList(
            [
                AttentionBlock(
                    inner_dim,
                    d_head,
                    n_heads=n_heads,
                    dropout=dropout,
                    context_dim=context_dim,
                    sdp_backend=sdp_backend,
                )
                for _ in range(depth)
            ]
        )

        if proj_type == "linear":
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        else:
            self.proj_out = zero_module(
                nn.Conv2d(
                    inner_dim,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))

    def forward(self, x, context=None):
        h, w = x.shape[-2:]
        x_in = x
        x = self.norm(x)

        if self.proj_type == "linear":
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()

        for _, block in enumerate(self.transformer_blocks):
            x = block(x, context=context)

        if self.proj_type == "linear":
            x = self.proj_out(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        else:
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            x = self.proj_out(x)

        return x + x_in
