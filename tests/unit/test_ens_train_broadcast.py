import numpy as np
import pytest
import torch

from xmetai.core.trainer import _broadcast_batch_from_rank0, _to_device_tensor


def test_to_device_tensor_from_numpy():
    value = np.ones((2, 3), dtype=np.float32)
    tensor = _to_device_tensor(value, torch.device("cpu"))
    assert torch.equal(tensor, torch.ones(2, 3, dtype=torch.float32))

def test_broadcast_batch_from_rank0_single_process():
    batch = {
        "inputs": torch.randn(1, 2, 4, 8, 8),
        "targets": torch.randn(1, 1, 4, 8, 8),
        "times": torch.tensor([1]),
    }
    synced = _broadcast_batch_from_rank0(batch)
    assert set(synced.keys()) == set(batch.keys())
    for key in batch:
        assert torch.equal(synced[key].cpu(), batch[key])


def test_broadcast_batch_from_rank0_requires_batch_without_dist():
    with pytest.raises(ValueError, match="src rank must provide batch"):
        _broadcast_batch_from_rank0(None)


def _broadcast_batch_from_rank0_worker(rank, world_size, init_file):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        expected_inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expected_times = torch.tensor([99])
        raw = {"inputs": expected_inputs, "times": expected_times} if rank == 0 else None
        synced = _broadcast_batch_from_rank0(raw, src=0)
        assert set(synced.keys()) == {"inputs", "times"}
        torch.testing.assert_close(synced["inputs"].cpu(), expected_inputs)
        torch.testing.assert_close(synced["times"].cpu(), expected_times)
    finally:
        dist.destroy_process_group()


def test_broadcast_batch_from_rank0_two_rank_gloo(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if dist.is_initialized():
        pytest.skip("This test manages its own process group")

    world_size = 2
    init_file = tmp_path / "ens_train_broadcast_init"
    mp.spawn(
        _broadcast_batch_from_rank0_worker,
        args=(world_size, str(init_file)),
        nprocs=world_size,
        join=True,
    )
