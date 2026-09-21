"""AC policy resolution and FSDP2 boundaries, without requiring CUDA."""

import copy

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper

from xmetai.cli.train import build_model
from xmetai.config import LazyConfig
from xmetai.core.defaults import apply_model_activation_checkpointing, create_fsdp2_model
from xmetai.layers.gated_deltanet2 import GatedDeltaNet2HybridLayer
from xmetai.models.meta import Meta
from xmetai.recipes import recipe_path


class PolicyModel(Meta):
    def __init__(self):
        super().__init__(device="cpu")
        self.layers = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.4), torch.nn.Linear(4, 1))

    def parallelize(self):
        return (torch.nn.Sequential,), (torch.nn.Linear,)

    def forward(self, x):
        return self.layers(x)


def policy(mods, enabled=True):
    return OmegaConf.create(dict(mods=mods, enable=enabled, preserve_rng_state=True), flags={"allow_objects": True})


@pytest.mark.parametrize("override", [None, [], [torch.nn.Linear]])
def test_build_model_respects_overrides_without_mutating_config(override):
    train = LazyConfig.load(recipe_path("train")).train
    train.device_type = "cpu"
    train.use_ddp = train.use_fsdp = train.use_fsdp2 = False
    train.ac.mods = override
    before = copy.deepcopy(train)
    cfg = OmegaConf.create(dict(train=train, model={"_target_": PolicyModel}), flags={"allow_objects": True})
    model = build_model(cfg)
    assert cfg.train == before
    wrappers = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
    expected = 1 if override is None else len(override) * 2
    assert len(wrappers) == expected
    if override is None:
        assert isinstance(model.layers, CheckpointWrapper)
    elif override:
        assert isinstance(model.layers[0], CheckpointWrapper)


def test_ac_preserves_gradients_rng_and_state_dict():
    model = PolicyModel()
    reference = copy.deepcopy(model)
    apply_model_activation_checkpointing(model, policy([torch.nn.Sequential]))
    x = torch.randn(2, 4)
    results = []
    for candidate in (reference, model):
        torch.manual_seed(42)
        output = candidate(x)
        output.square().sum().backward()
        results.append((output.detach(), [p.grad.clone() for p in candidate.parameters()], torch.rand(3)))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert model.state_dict().keys() == reference.state_dict().keys()
    reference.load_state_dict(model.state_dict())


def test_ac_disabled_and_invalid_policy():
    model = PolicyModel()
    apply_model_activation_checkpointing(model, policy([torch.nn.Linear], enabled=False))
    assert not any(isinstance(m, CheckpointWrapper) for m in model.modules())
    with pytest.raises(TypeError, match="Module classes"):
        apply_model_activation_checkpointing(model, policy(["Linear"]))


@pytest.mark.parametrize("shard_outer", [False, True])
def test_fsdp2_uses_selected_units_then_root(monkeypatch, shard_outer):
    import torch.distributed.device_mesh as mesh_module
    import torch.distributed.fsdp as fsdp_module

    model = PolicyModel()
    apply_model_activation_checkpointing(model, policy([torch.nn.Sequential]))
    expected = [model.layers[2], model.layers[0]] + ([model.layers] if shard_outer else []) + [model]
    calls = []
    monkeypatch.setattr(mesh_module, "init_device_mesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_module, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
    monkeypatch.setattr("xmetai.core.defaults.comm.get_world_size", lambda: 1)
    create_fsdp2_model(
        model,
        mods=(torch.nn.Linear, torch.nn.Sequential) if shard_outer else (torch.nn.Linear,),
        precision=OmegaConf.create(dict(param_dtype="bf16", reduce_dtype="fp32")),
    )
    assert [m for m, _ in calls] == expected
    assert calls[-1][1]["reshard_after_forward"] is False


class EmbeddingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, time_embedding, **kwargs):
        return x + time_embedding * self.weight


def test_hybrid_time_embedding_dispatch_survives_wrapping():
    # Exercise the real HybridLayer forward with light blocks, without CUDA GDN kernels.
    layer = GatedDeltaNet2HybridLayer.__new__(GatedDeltaNet2HybridLayer)
    torch.nn.Module.__init__(layer)
    layer.with_embedding = True
    layer.adaLN_modulation = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.adaLN_modulation.weight.copy_(2 * torch.eye(4))
    layer.blocks = torch.nn.ModuleList([EmbeddingBlock(), EmbeddingBlock()])
    reference = copy.deepcopy(layer)
    layer.blocks[0] = checkpoint_wrapper(layer.blocks[0])
    layer.blocks[1] = checkpoint_wrapper(layer.blocks[1])
    x, embedding = torch.randn(2, 4), torch.randn(2, 4)
    torch.testing.assert_close(layer(x, embedding), x + 3 * embedding)
    for candidate in (reference, layer):
        candidate(x, embedding).square().sum().backward()
    for p, q in zip(reference.parameters(), layer.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AC + FSDP2 integration")
def test_ac_fsdp2_forward_backward_matches_unsharded_model(tmp_path):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    if dist.is_initialized():
        pytest.skip("This test owns its process group")
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        model = PolicyModel().cuda()
        reference = copy.deepcopy(model)
        apply_model_activation_checkpointing(model, policy([torch.nn.Sequential, torch.nn.Linear]))
        create_fsdp2_model(model, mods=(torch.nn.Linear,), precision=OmegaConf.create(dict(param_dtype="fp32", reduce_dtype="fp32")))
        x = torch.randn(2, 4, device="cuda")
        for candidate in (reference, model):
            torch.manual_seed(42)
            candidate(x).square().sum().backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            grad = actual.grad.full_tensor() if isinstance(actual.grad, DTensor) else actual.grad
            torch.testing.assert_close(grad, expected.grad)
    finally:
        dist.destroy_process_group()


def test_checkpoint_mode_is_declared_by_layer(monkeypatch):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

    import xmetai.core.defaults as defaults

    model = PolicyModel()
    model.layers[0].activation_checkpoint_reentrant = True
    modes = []
    original_wrapper = defaults.checkpoint_wrapper

    def record_wrapper(module, **kwargs):
        modes.append(kwargs["checkpoint_impl"])
        return original_wrapper(module, **kwargs)

    monkeypatch.setattr(defaults, "checkpoint_wrapper", record_wrapper)
    apply_model_activation_checkpointing(model, policy([torch.nn.Linear]))
    assert modes == [CheckpointImpl.REENTRANT, CheckpointImpl.NO_REENTRANT]


@pytest.mark.parametrize("mamba_type", ["mamba", "mamba_met", "mamba2", "mamba2_met", "mamba3", "mamba3_met"])
def test_mamba_dependency_failure_is_at_construction(monkeypatch, mamba_type):
    import sys

    from xmetai.layers import MambaVisionLayer

    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    with pytest.raises(ImportError, match="mamba-ssm"):
        MambaVisionLayer(dim=8, depth=1, num_heads=2, mamba_type=mamba_type)


def test_attention_only_mamba_layer_needs_no_mamba_backend(monkeypatch):
    import sys

    from xmetai.layers import MambaVisionLayer

    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    layer = MambaVisionLayer(dim=8, depth=1, num_heads=2, transformer_blocks=[0])
    x = torch.randn(2, 4, 8, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert not layer.activation_checkpoint_reentrant
    mamba3 = MambaVisionLayer(dim=8, depth=1, num_heads=2, transformer_blocks=[0], mamba_type="mamba3")
    assert mamba3.activation_checkpoint_reentrant


def test_gdn_dependency_failure_is_at_construction(monkeypatch):
    import sys

    from xmetai.layers import GatedDeltaNet2Layer

    monkeypatch.setitem(sys.modules, "fla", None)
    with pytest.raises(ImportError, match="flash-linear-attention"):
        GatedDeltaNet2Layer(dim=8, depth=1, num_heads=2)


def test_mamba_scan_binding_and_onnx_registration(monkeypatch):
    import sys
    from types import ModuleType

    from xmetai.layers.mamba_vision import MambaVisionMixer, symbolic_selective_scan_fn

    backend = ModuleType("mamba_ssm.ops.selective_scan_interface")
    backend.SelectiveScanFn = type("SelectiveScanFn", (), {})
    calls = []

    def scan(x, *args, **kwargs):
        calls.append(x.shape)
        return x

    backend.selective_scan_fn = scan
    monkeypatch.setitem(sys.modules, "mamba_ssm", ModuleType("mamba_ssm"))
    monkeypatch.setitem(sys.modules, "mamba_ssm.ops", ModuleType("mamba_ssm.ops"))
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    model = MambaVisionMixer(d_model=8)
    assert model.selective_scan_fn is scan
    assert backend.SelectiveScanFn.symbolic is symbolic_selective_scan_fn
    x = torch.randn(2, 4, 8, requires_grad=True)
    model(x).sum().backward()
    assert len(calls) == 1
    assert x.grad is not None


@pytest.mark.parametrize("enabled", [False, True])
def test_parallelize_runs_independently_of_common_ac_switch(monkeypatch, enabled):
    calls = []

    def parallelize(self):
        calls.append(self)
        return None, None

    monkeypatch.setattr(PolicyModel, "parallelize", parallelize)
    train = LazyConfig.load(recipe_path("train")).train
    train.device_type = "cpu"
    train.use_ddp = train.use_fsdp = train.use_fsdp2 = False
    train.ac.enable = enabled
    cfg = OmegaConf.create(dict(train=train, model={"_target_": PolicyModel}), flags={"allow_objects": True})
    model = build_model(cfg)
    assert calls == [model]


@pytest.mark.parametrize("model_defaults", [None, (), (torch.nn.Sequential,)])
@pytest.mark.parametrize("override", [None, [], [torch.nn.Linear]])
def test_parallel_defaults_and_explicit_overrides(monkeypatch, model_defaults, override):
    monkeypatch.setattr(PolicyModel, "parallelize", lambda self: (model_defaults, model_defaults))
    train = LazyConfig.load(recipe_path("train")).train
    train.device_type = "cpu"
    train.use_ddp = train.use_fsdp = False
    train.use_fsdp2 = True
    train.ac.default_mods = train.fsdp.default_mods = (torch.nn.Linear,)
    train.ac.mods = train.fsdp.mods = override
    captured = []

    def capture_fsdp(model, **kwargs):
        captured.append(tuple(kwargs["mods"]))
        return model

    monkeypatch.setattr("xmetai.cli.train.create_fsdp2_model", capture_fsdp)
    before = copy.deepcopy(train)
    cfg = OmegaConf.create(dict(train=train, model={"_target_": PolicyModel}), flags={"allow_objects": True})
    model = build_model(cfg)
    expected = tuple(override) if override is not None else ((torch.nn.Linear,) if model_defaults is None else model_defaults)
    assert captured == [expected]
    wrappers = [module for module in model.modules() if isinstance(module, CheckpointWrapper)]
    assert len(wrappers) == (2 if expected == (torch.nn.Linear,) else len(expected))
    assert cfg.train == before


def test_base_model_inherits_recipe_defaults():
    assert Meta(device="cpu").parallelize() == (None, None)
