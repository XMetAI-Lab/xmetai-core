import torch
from torch import nn

from xmetai.models.pred_head import PLSFHead


def test_resize_preserves_constant_fields_and_zero_innovations():
    head = PLSFHead(4, 3, 2, (8, 8), [4, 8], upsample_mode="resize")
    x = torch.ones(1, 4, 4, 8)
    y = head(x, need_rearrange=False)
    assert y.shape == (1, 5, 32, 64)
    torch.testing.assert_close(y, y[..., :1, :1].expand_as(y))
    torch.testing.assert_close(head(torch.zeros_like(x), need_rearrange=False), torch.zeros_like(y))


def test_resize_is_longitude_periodic_and_differentiable():
    torch.manual_seed(42)
    head = PLSFHead(4, 3, 2, (8, 8), [4, 8], upsample_mode="resize")
    x = torch.randn(1, 4, 4, 8, requires_grad=True)
    y = head(x, need_rearrange=False)
    torch.testing.assert_close(head(x.roll(1, -1), need_rearrange=False), y.roll(8, -1))
    torch.testing.assert_close(head(-x, need_rearrange=False), -y)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


def test_existing_transpose_mode_and_token_output_contract():
    original = PLSFHead(4, 3, 2, (8, 8), [4, 8])
    assert isinstance(original.pl_head, nn.ConvTranspose2d)
    assert isinstance(original.sf_head, nn.ConvTranspose2d)
    resized = PLSFHead(4, 3, 2, (8, 8), [4, 8], upsample_mode="resize")
    x = torch.randn(1, 32, 4)
    assert original(x).shape == resized(x).shape == (1, 32, 5 * 64)
