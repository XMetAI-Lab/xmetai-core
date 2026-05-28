"""
Earth-aware Convolution Layers

Implements convolution and transposed convolution layers that respect Earth's topology:
- Longitude (horizontal): circular padding (Earth wraps around)
- Latitude (vertical): reflect padding (poles are symmetric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["earth_pad", "EarthConv2d", "EarthConvTranspose2d", "EarthDeconv2d"]


def earth_pad(x, pad_lat, pad_lon):
    """
    Apply Earth-aware padding to input tensor.

    Args:
        x: Input tensor of shape (B, C, H, W) where H is latitude, W is longitude
        pad_lat: Padding size for latitude (vertical)
        pad_lon: Padding size for longitude (horizontal)

    Returns:
        Padded tensor
    """
    # 1️⃣ 经度方向 circular pad (左右循环，因为地球是圆的)
    x = torch.cat([x[..., -pad_lon:], x, x[..., :pad_lon]], dim=-1)

    # 2️⃣ 纬度方向 reflect pad (上下反射，模拟南北极)
    x = F.pad(x, (0, 0, pad_lat, pad_lat), mode="reflect")  # (left, right, top, bottom) for last 2 dims

    return x


class EarthConv2d(nn.Module):
    """
    2D Convolution with Earth-aware padding.

    Applies circular padding in longitude direction and reflect padding in latitude direction
    before performing standard 2D convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",  # This is ignored, kept for API compatibility
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding size (will be applied as earth_pad)
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: If True, adds a learnable bias
            padding_mode: Ignored (kept for compatibility with nn.Conv2d)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        # Calculate padding for latitude and longitude
        if isinstance(padding, int):
            self.pad_lat = padding
            self.pad_lon = padding
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            self.pad_lat = padding[0]  # height/latitude padding
            self.pad_lon = padding[1]  # width/longitude padding
        else:
            self.pad_lat = 0
            self.pad_lon = 0

        # Create the actual conv layer with no padding (we handle it manually)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,  # We handle padding manually
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Convolved tensor
        """
        # Apply Earth-aware padding
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = earth_pad(x, self.pad_lat, self.pad_lon)

        # Apply standard convolution
        return self.conv(x)


class EarthConvTranspose2d(nn.Module):
    """
    2D Transposed Convolution (deconvolution) with Earth-aware padding.

    Applies circular padding in longitude direction and reflect padding in latitude direction
    before performing standard 2D transposed convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",  # This is ignored, kept for API compatibility
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding size (will be applied as earth_pad)
            output_padding: Additional size added to output shape
            dilation: Spacing between kernel elements
            groups: Number of blocked connections
            bias: If True, adds a learnable bias
            padding_mode: Ignored (kept for compatibility with nn.ConvTranspose2d)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups

        # Calculate padding for latitude and longitude
        if isinstance(padding, int):
            self.pad_lat = padding
            self.pad_lon = padding
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            self.pad_lat = padding[0]  # height/latitude padding
            self.pad_lon = padding[1]  # width/longitude padding
        else:
            self.pad_lat = 0
            self.pad_lon = 0

        # Create the actual conv transpose layer with no padding (we handle it manually)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,  # We handle padding manually
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Transposed convolved tensor
        """
        # Apply Earth-aware padding
        if self.pad_lat > 0 or self.pad_lon > 0:
            x = earth_pad(x, self.pad_lat, self.pad_lon)

        # Apply standard transposed convolution
        return self.conv_transpose(x)


# Alias for convenience
EarthDeconv2d = EarthConvTranspose2d
