import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from xmetai.core.defaults import create_fsdp2_model
from xmetai.core.trainer import SimpleTrainer
from xmetai.models.ema import EMA


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 16),
        )
        # Keep one parameter owned by the FSDP2 root module. The root uses
        # reshard_after_forward=False, so this covers EMA's explicit reshard().
        self.scale = torch.nn.Parameter(torch.ones(16))

    def forward(self, inputs):
        return self.layers(inputs) * self.scale


class DTensorModel(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)


def _local_tensor(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
        if hasattr(tensor, "wait"):
            tensor = tensor.wait()
    return tensor


def _clone_parameters(model):
    return {name: _local_tensor(param).detach().clone() for name, param in model.named_parameters()}


def _assert_parameters_equal(model, expected, atol=0):
    for name, param in model.named_parameters():
        torch.testing.assert_close(_local_tensor(param), expected[name], rtol=0, atol=atol)


def _assert_ema_uses_local_tensors(ema, model):
    if any(isinstance(module, FSDP) for module in model.modules()):
        context = FSDP.summon_full_params(model)
    else:
        context = contextlib.nullcontext()

    with context:
        model_params = dict(model.named_parameters())

    assert set(ema.ema_params).issubset(model_params)
    assert all(not isinstance(param, DTensor) for param in ema.ema_params.values())
    for name, ema_param in ema.ema_params.items():
        assert ema_param.shape == _local_tensor(model_params[name]).shape
        assert ema_param.device == _local_tensor(model_params[name]).device


def _validate_backend_roundtrip(model, device, expect_model_dtensor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model, decay=0.9)
    _assert_ema_uses_local_tensors(ema, model)

    inputs = torch.randn(4, 16, device=device)
    targets = torch.randn(4, 16, device=device)

    if expect_model_dtensor:
        # FSDP2 exposes the root-owned scale as an unsharded Tensor after
        # forward. update_params() must reshard it before touching EMA.
        with torch.no_grad():
            model(inputs)
        assert any(not isinstance(param, DTensor) for param in model.parameters())
        ema.update_params(model)
        assert all(isinstance(param, DTensor) for param in model.parameters())
        _assert_ema_uses_local_tensors(ema, model)

    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema(model)
    after_step = _clone_parameters(model)

    ema.reverse(model)
    ema_parameters = _clone_parameters(model)
    ema.restore(model)

    _assert_parameters_equal(model, after_step)
    assert any(not torch.equal(ema_parameters[name], after_step[name]) for name in ema_parameters)

    ema_local_bytes = sum(param.numel() * param.element_size() for param in ema.ema_params.values())
    model_local_bytes = sum(_local_tensor(param).numel() * _local_tensor(param).element_size() for param in model.parameters())
    assert ema_local_bytes == model_local_bytes


def _init_single_rank_process_group(tmp_path):
    init_file = tmp_path / "single_rank.dist_init"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )


def _build_gpu_backend(backend, device):
    model = TinyModel().to(device)
    if backend == "plain":
        return model
    if backend == "ddp":
        return DDP(model, device_ids=[device.index])
    if backend == "fsdp1":
        return FSDP(model, device_id=device, use_orig_params=True)
    if backend == "fsdp2":
        precision = SimpleNamespace(param_dtype="fp32", reduce_dtype="fp32")
        return create_fsdp2_model(
            model,
            precision=precision,
            cpu_offload=False,
            mods=(torch.nn.Linear,),
        )
    raise ValueError(f"Unknown backend: {backend}")


def _dtensor_ema_worker(rank, world_size, init_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        mesh = init_device_mesh("cpu", mesh_shape=(world_size,))
        global_weight = torch.arange(world_size * 8, dtype=torch.float32).reshape(world_size * 2, 4)
        sharded_weight = distribute_tensor(global_weight, mesh, placements=[Shard(0)])
        model = DTensorModel(sharded_weight)
        ema = EMA(model, decay=0.5)

        ema_weight = ema.ema_params["weight"]
        assert not isinstance(ema_weight, DTensor)
        assert ema_weight.shape == model.weight.to_local().shape
        assert ema_weight.numel() * world_size == model.weight.numel()

        initial = model.weight.to_local().detach().clone()
        with torch.no_grad():
            model.weight.add_(2.0)
        updated = model.weight.to_local().detach().clone()

        ema(model)
        first_decay = min(0.5, 2.0 / 11.0)
        expected_ema = initial.clone().mul_(first_decay).add_(updated, alpha=(1.0 - first_decay))

        ema.reverse(model)
        torch.testing.assert_close(model.weight.to_local(), expected_ema, rtol=0, atol=0)
        ema.restore(model)
        torch.testing.assert_close(model.weight.to_local(), updated, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_ema_rejects_invalid_decay():
    model = torch.nn.Linear(4, 3)
    with pytest.raises(ValueError, match="Decay must be between 0 and 1"):
        EMA(model, decay=-0.1)
    with pytest.raises(ValueError, match="Decay must be between 0 and 1"):
        EMA(model, decay=1.1)


def test_ema_only_tracks_trainable_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    model[0].requires_grad_(False)

    ema = EMA(model)

    assert set(ema.ema_params) == {"1.weight", "1.bias"}


def test_ema_update_reverse_restore():
    model = torch.nn.Linear(4, 3)
    initial = _clone_parameters(model)
    ema = EMA(model, decay=0.5)

    with torch.no_grad():
        for param in model.parameters():
            param.add_(2.0)
    updated = _clone_parameters(model)

    ema(model)

    first_decay = min(0.5, 2.0 / 11.0)
    expected_ema = {name: tensor.clone() for name, tensor in initial.items()}
    for name in expected_ema:
        expected_ema[name].mul_(first_decay).add_(updated[name], alpha=(1.0 - first_decay))

    ema.reverse(model)
    _assert_parameters_equal(model, expected_ema, atol=1e-7)

    ema.restore(model)
    _assert_parameters_equal(model, updated)


def test_ema_update_params_copies_current_parameters():
    model = torch.nn.Linear(4, 3)
    ema = EMA(model)

    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-3.0, 3.0)
    current = _clone_parameters(model)

    ema.update_params(model)

    assert ema.num_updates.item() == 0
    for name, ema_param in ema.ema_params.items():
        assert torch.equal(ema_param, current[name])


def test_ema_keeps_existing_state_dict_semantics():
    model = torch.nn.Linear(4, 3)
    ema = EMA(model)

    assert set(ema.state_dict()) == {"decay", "num_updates"}


def test_simple_trainer_updates_ema_after_optimizer_step():
    class OneBatchLoader:
        def __iter__(self):
            return iter([{"inputs": torch.ones(1, 1), "targets": torch.zeros(1, 1)}])

    class TrainModel(torch.nn.Module):
        device = "cpu"
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(self.linear.weight, 1.0)

        def forward(self, batch):
            pred = self.linear(batch["inputs"])
            return {"loss": torch.nn.functional.mse_loss(pred, batch["targets"])}

    model = TrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleTrainer(model, OneBatchLoader(), optimizer, use_ema=True)
    trainer.ema.decay.fill_(0.0)

    trainer.train(0, 1)

    torch.testing.assert_close(trainer.ema.ema_params["linear.weight"], model.linear.weight.detach())


def test_ema_two_rank_cpu_dtensor(tmp_path):
    world_size = 2
    init_file = tmp_path / "dtensor_ema_init"
    mp.spawn(
        _dtensor_ema_worker,
        args=(world_size, str(init_file)),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["plain", "ddp", "fsdp1", "fsdp2"])
def test_ema_gpu_backends(backend, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA is required for GPU EMA backend tests")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is required for GPU EMA backend tests")
    if dist.is_initialized():
        pytest.skip("This test manages its own single-rank process group")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    _init_single_rank_process_group(tmp_path)
    try:
        torch.manual_seed(1234)
        model = _build_gpu_backend(backend, device)
        _validate_backend_roundtrip(model, device, expect_model_dtensor=backend == "fsdp2")
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()
