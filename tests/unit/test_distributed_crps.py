from datetime import timedelta
from importlib import import_module

import pytest
import torch
import torch.distributed as dist


def _check_channel_sharded_crps(rank, init_file):
    crps = import_module("xmetai.loss.crps_loss")
    rng = torch.Generator().manual_seed(91)
    fields = torch.randn(2, 1, 1, 5, 5, 8, generator=rng, dtype=torch.float64)
    target = torch.randn(1, 1, 1, 5, 8, generator=rng, dtype=torch.float64)
    weight = torch.rand(1, 1, 5, 5, 1, generator=rng, dtype=torch.float64)

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        for scales in ([], [1.25, 0.5]):
            output = fields[rank].clone().requires_grad_()
            actual = crps.multiscale_afcrps_loss(output, target, weight, filter_scales=scales)
            reference_fields = fields.clone().requires_grad_()
            reference = crps._score_members(reference_fields, target, weight, 0.95, scales)
            reference_grad = torch.autograd.grad(reference, reference_fields)[0][rank] * 2
            torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(torch.autograd.grad(actual, output)[0], reference_grad, rtol=1e-11, atol=1e-12)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_channel_sharded_crps_matches_full_ensemble_with_uneven_channels(tmp_path):
    torch.multiprocessing.spawn(_check_channel_sharded_crps, args=(str(tmp_path / "gloo"),), nprocs=2)
