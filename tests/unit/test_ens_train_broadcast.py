import numpy as np
import pytest
import torch

from xmetai.core.trainer import _broadcast_batch_from_rank0, _to_device_tensor


def test_to_device_tensor_from_numpy():
    value = np.ones((2, 3), dtype=np.float32)
    tensor = _to_device_tensor(value, torch.device("cpu"))
    assert torch.equal(tensor, torch.ones(2, 3))


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
