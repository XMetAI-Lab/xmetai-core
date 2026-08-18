from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from xmetai.core.checkpoint import FSDPCheckpointer
from xmetai.core.defaults import create_fsdp2_model


class DummyCheckpointable:
    def state_dict(self):
        return {"value": torch.ones(1)}


def test_fsdp_checkpointer_saves_checkpointables_by_default(tmp_path):
    model = torch.nn.Linear(2, 1)
    checkpointer = FSDPCheckpointer(
        model,
        save_dir=str(tmp_path),
        trainer=DummyCheckpointable(),
    )

    checkpointer.save("model", iteration=7)

    checkpoint = torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model", "trainer", "iteration"}
    assert checkpoint["iteration"] == 7
    torch.testing.assert_close(checkpoint["trainer"]["value"], torch.ones(1))
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())


@pytest.mark.gpu
def test_fsdp2_checkpointer_saves_full_cpu_model_state(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA is required for FSDP2 checkpoint tests")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is required for FSDP2 checkpoint tests")
    if dist.is_initialized():
        pytest.skip("This test manages its own single-rank process group")

    init_file = tmp_path / "single_rank.dist_init"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        torch.cuda.set_device(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.GELU(),
            torch.nn.Linear(8, 2),
        ).cuda()
        precision = SimpleNamespace(param_dtype="fp32", reduce_dtype="fp32")
        model = create_fsdp2_model(model, precision=precision, cpu_offload=False, mods=(torch.nn.Linear,))
        checkpointer = FSDPCheckpointer(model, save_dir=str(tmp_path))

        checkpointer.save("fsdp2_model")

        checkpoint = torch.load(tmp_path / "fsdp2_model.pth", map_location="cpu", weights_only=False)
        assert set(checkpoint) == {"model"}
        assert checkpoint["model"]
        assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()


@pytest.mark.gpu
def test_fsdp1_checkpointer_saves_full_cpu_model_state(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA is required for FSDP checkpoint tests")
    if not dist.is_nccl_available():
        pytest.skip("NCCL backend is required for FSDP checkpoint tests")
    if dist.is_initialized():
        pytest.skip("This test manages its own single-rank process group")

    init_file = tmp_path / "single_rank_fsdp1.dist_init"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    try:
        torch.cuda.set_device(0)
        model = FSDP(torch.nn.Linear(4, 2).cuda(), device_id=torch.device("cuda", 0), use_orig_params=True)
        checkpointer = FSDPCheckpointer(model, save_dir=str(tmp_path))

        checkpointer.save("fsdp1_model")

        checkpoint = torch.load(tmp_path / "fsdp1_model.pth", map_location="cpu", weights_only=False)
        assert set(checkpoint) == {"model"}
        assert checkpoint["model"]
        assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())
    finally:
        dist.destroy_process_group()
        torch.cuda.empty_cache()
