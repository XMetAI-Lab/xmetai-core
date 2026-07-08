import pytest
import torch
import torch.nn as nn

from xmetai.layers.activation import ACTIVATIONS, GEGLU, SquaredReLU, SwiGLU, get_activation


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_get_activation_instantiates_builtin_names(name):
    module = get_activation(name)
    assert isinstance(module, nn.Module)


def test_get_activation_accepts_module_class():
    module = get_activation(nn.GELU)
    assert isinstance(module, nn.GELU)


def test_get_activation_accepts_callable_factory():
    module = get_activation(lambda: nn.SiLU())
    assert isinstance(module, nn.SiLU)


def test_get_activation_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown activation"):
        get_activation("not_a_real_activation")


def test_get_activation_rejects_invalid_factory_return_type():
    with pytest.raises(TypeError, match="must return nn.Module"):
        get_activation(lambda: 1)


@pytest.mark.parametrize("activation_cls", [GEGLU, SwiGLU])
def test_gated_activations_preserve_last_dim(activation_cls):
    x = torch.randn(2, 8, 16)
    gated = torch.cat([x, x], dim=-1)
    output = activation_cls()(gated)
    assert output.shape == x.shape


def test_squared_relu_is_non_negative():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    output = SquaredReLU()(x)
    assert torch.all(output >= 0)
    assert output[3].item() == pytest.approx(1.0)
    assert output[4].item() == pytest.approx(4.0)


def test_swiglu_differs_from_geglu_for_same_input():
    x = torch.randn(4, 6)
    gated = torch.cat([x, x], dim=-1)
    swiglu_out = SwiGLU()(gated)
    geglu_out = GEGLU()(gated)
    assert not torch.allclose(swiglu_out, geglu_out)
