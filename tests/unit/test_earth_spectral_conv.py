"""Unit tests for EarthSpectralConv2d."""

import pytest
import torch

from xmetai.layers.earth_spectral_conv import EarthSpectralConv2d


class TestEarthSpectralConv2d:
    def test_forward_shape(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(4, 8, img_size=(nlat, nlon))
        x = torch.randn(2, 4, nlat, nlon)

        output = layer(x)

        assert output.shape == (2, 8, nlat, nlon)

    def test_channel_change(self, torch_seed):
        nlat, nlon = 12, 24
        layer = EarthSpectralConv2d(3, 5, img_size=(nlat, nlon))
        x = torch.randn(1, 3, nlat, nlon)

        output = layer(x)

        assert output.shape == (1, 5, nlat, nlon)

    def test_square_img_size_int(self, torch_seed):
        size = 16
        layer = EarthSpectralConv2d(2, 2, img_size=size)
        x = torch.randn(1, 2, size, size)

        output = layer(x)

        assert output.shape == (1, 2, size, size)

    def test_output_size_change(self, torch_seed):
        layer = EarthSpectralConv2d(4, 4, img_size=(16, 32), out_size=(32, 64))
        x = torch.randn(1, 4, 16, 32)

        output = layer(x)

        assert output.shape == (1, 4, 32, 64)

    def test_groups(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(8, 16, img_size=(nlat, nlon), groups=4)
        x = torch.randn(2, 8, nlat, nlon)

        output = layer(x)

        assert output.shape == (2, 16, nlat, nlon)

    def test_diagonal_operator(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(4, 4, img_size=(nlat, nlon), operator_type="diagonal")
        x = torch.randn(1, 4, nlat, nlon)

        output = layer(x)

        assert output.shape == (1, 4, nlat, nlon)

    def test_block_diagonal_operator(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(4, 4, img_size=(nlat, nlon), operator_type="block-diagonal")
        x = torch.randn(1, 4, nlat, nlon)

        output = layer(x)

        assert output.shape == (1, 4, nlat, nlon)

    def test_bias(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(4, 4, img_size=(nlat, nlon), bias=True)
        x = torch.randn(1, 4, nlat, nlon)

        output = layer(x)

        assert output.shape == (1, 4, nlat, nlon)
        assert layer.bias.shape == (4, 1, 1)

    def test_gradient_flow(self, torch_seed):
        nlat, nlon = 16, 32
        layer = EarthSpectralConv2d(4, 8, img_size=(nlat, nlon))
        x = torch.randn(2, 4, nlat, nlon, requires_grad=True)

        output = layer(x)
        output.sum().backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        assert layer.weight.grad is not None
        assert layer.weight.grad.abs().sum() > 0

    def test_invalid_groups_raises(self):
        with pytest.raises(ValueError, match="divisible by groups"):
            EarthSpectralConv2d(7, 8, img_size=(16, 32), groups=4)

    def test_invalid_input_size_raises(self, torch_seed):
        layer = EarthSpectralConv2d(4, 8, img_size=(16, 32))
        x = torch.randn(1, 4, 8, 16)

        with pytest.raises(ValueError, match="Expected input spatial size"):
            layer(x)

    def test_invalid_operator_type_raises(self):
        with pytest.raises(ValueError, match="Unknown operator_type"):
            EarthSpectralConv2d(4, 8, img_size=(16, 32), operator_type="invalid")
