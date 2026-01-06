import torch
from einops.einops import rearrange, repeat
from torch import nn as nn

from .helpers import to_2tuple

__all__ = ['PatchEmbed', 'CubeEmbed', 'FieldEmbed', 'TimestepEmbed', 'GeoEmbed']


def sincos_embedding(x, embed_dim, max_period=10000, is_periodic=False):
    """
    Args:
        x: Input tensor
        embed_dim: Embedding dimension (must be even)
        max_period: Maximum period for non-periodic embeddings
        is_periodic: Whether the input is periodic (like time)
        period: The period length (e.g., 24 for hours, 365 for days)
    """
    omega = torch.arange(embed_dim//2, dtype=x.dtype, device=x.device)

    if is_periodic:
        # For periodic variables, use direct angular conversion
        x = 2 * torch.pi * x
    else:
        # Original implementation for non-periodic variables
        omega /= embed_dim / 2.
        omega = 1. / max_period ** omega  # (D/2,)

    out = torch.einsum('m,d->md', x.reshape(-1), omega)  # (M, D/2)
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        input_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
            self,
            input_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=None):
        super().__init__()
        input_size = to_2tuple(input_size)
        patch_size = to_2tuple(patch_size)
        input_size = [
            input_size[0] //
            patch_size[0],
            input_size[1] //
            patch_size[1]]

        self.input_size = input_size
        self.patch_size = patch_size
        self.input_size = input_size

        self.num_patch = input_size[0] * input_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.input_size[0] and W == self.input_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patch_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class CubeEmbed(nn.Module):
    def __init__(self,
                 in_chans: int,
                 out_chans: int,
                 in_frames=1,
                 patch_size: int = 2,
                 stride_size: int = -1,
                 const_chans: int = 0,
                 norm_layer=None,
                 flatten=False,
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.pad = None

        if stride_size < 1:
            stride_size = patch_size
        else:
            stride_size = to_2tuple(stride_size)
            self.pad = nn.ZeroPad2d((1, 1, 1, 0))  # manually pad input before conv.
        self.proj = nn.Conv2d(
            in_chans * in_frames, out_chans,
            kernel_size=patch_size,
            stride=stride_size,
            padding=0
        )
        if const_chans > 0:
            self.const_proj = nn.Conv2d(
                const_chans, out_chans,
                kernel_size=patch_size,
                stride=stride_size,
                padding=0
            )
        self.norm = norm_layer(out_chans) if norm_layer else nn.Identity()

    def forward(self, x, const=None):
        if x.ndim == 5:
            x = rearrange(x, 'n t c h w -> n (t c) h w')
        if self.pad is not None:
            x = self.pad(x)
        x = self.proj(x)
        if const is not None:
            if const.ndim == 3:
                const = const.unsqueeze(0)  # add batch dimension
            x += self.const_proj(const)
        if self.flatten:
            x = rearrange(x, 'n c h w -> n (h w) c')
        x = self.norm(x)
        return x


class FieldEmbed(nn.Module):
    def __init__(self,
                 in_chans,
                 pl_chans,
                 const_chans,
                 embed_dim,
                 in_frames=1,
                 patch_size=4,
                 norm_layer=None,
                 flatten=False,
                 proj_type="conv2d",
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.pl_chans = pl_chans
        self.flatten = flatten
        self.proj_type = proj_type

        if proj_type == "conv2d":
            conv_class = nn.Conv2d
            kernel_size = patch_size
        else:
            conv_class = nn.Conv3d
            kernel_size = (in_frames, *patch_size)
            in_frames = 1

        self.pl_proj = conv_class(
            pl_chans * in_frames, embed_dim // 2,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0
        )

        self.sf_proj = conv_class(
            (in_chans - pl_chans) * in_frames + const_chans, embed_dim // 2,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0
        )

        self.out_proj = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=1, padding=0)

    def forward(self, x, const=None):
        xp = x[:, :, :self.pl_chans]
        xs = x[:, :, self.pl_chans:]

        if self.proj_type == "conv2d":
            xp, xs = map(
                lambda x: rearrange(
                    x, 'n t c h w -> n (t c) h w'), (xp, xs))
        elif self.proj_type == "conv3d":
            xp, xs = map(
                lambda x: rearrange(
                    x, 'n t c h w -> n c t h w'), (xp, xs))

        if const is not None:
            const = repeat(const, 'c h w -> n c h w', n=x.shape[0])
            xs = torch.cat([xs, const], dim=1)

        xp = self.pl_proj(xp)
        xs = self.sf_proj(xs)

        embed = torch.cat([xp, xs], dim=1)
        embed = self.out_proj(embed.squeeze(dim=2))

        if self.flatten:
            embed = rearrange(embed, 'n c h w -> n (h w) c')

        return embed


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency = frequency

    def forward(self, x):
        embed = sincos_embedding(x.float(), self.frequency)
        embed = self.mlp(embed.to(x))
        return embed


class GeoEmbed(nn.Module):
    def __init__(self, hidden_size, frequency=36, patch_size=4, flatten=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.frequency = frequency
        self.flatten = flatten

        self.proj = nn.Conv2d(
            frequency, hidden_size,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
            padding=0
        )

    def forward(self, z):
        ih, iw = z.shape[-2:]
        lat = torch.linspace(90, -90, ih, device=z.device) / ih
        lon = torch.linspace(0, 360, iw, device=z.device) / iw
        lon, lat = torch.meshgrid([lon, lat], indexing='xy')
        embed = torch.stack([lon, lat, z], dim=-1).flatten()
        embed = sincos_embedding(embed, self.frequency // 3)
        embed = rearrange(embed, '(h w c) d -> 1 (c d) h w', h=ih, w=iw)
        embed = self.proj(embed.to(z))

        if self.flatten:
            embed = rearrange(embed, 'n c h w -> n (h w) c')

        return embed
