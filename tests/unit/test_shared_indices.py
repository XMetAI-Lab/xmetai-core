from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf

from xmetai.core.trainer import ExpandTrainer
from xmetai.data.loader import build_train_loader
from xmetai.data.sampler import TrainingSampler
from xmetai.data.weather.multi_zarr_dataset import MultiZarrDataset


def test_shared_indices_reproduce_rank_zero_batches(monkeypatch):
    monkeypatch.setattr("xmetai.data.loader.get_world_size", lambda: 4)
    monkeypatch.setattr("xmetai.data.sampler.comm.get_world_size", lambda: 4)
    monkeypatch.setattr("xmetai.data.sampler.comm.shared_random_seed", lambda: 42)
    dataset = [{"times": index} for index in range(13)]

    def read_batches(rank, shared_indices):
        monkeypatch.setattr("xmetai.data.sampler.comm.get_rank", lambda: rank)
        loader = build_train_loader(dataset, total_batch_size=8, pin_memory=False, shared_indices=shared_indices)
        assert type(loader.sampler) is TrainingSampler
        batches = iter(loader)
        return [next(batches)["times"].tolist() for _ in range(8)]

    reference = read_batches(rank=0, shared_indices=False)
    assert read_batches(rank=3, shared_indices=True) == reference
    assert read_batches(rank=3, shared_indices=False) != reference


@pytest.mark.parametrize("num_workers", [0, 2])
def test_shared_indices_read_runtime_forecast_frames_after_resume(monkeypatch, num_workers):
    monkeypatch.setattr("xmetai.data.loader.get_world_size", lambda: 4)
    monkeypatch.setattr("xmetai.data.sampler.comm.get_world_size", lambda: 4)
    monkeypatch.setattr("xmetai.data.sampler.comm.shared_random_seed", lambda: 42)
    times = pd.date_range("2024-01-01", periods=40, freq="6h")
    source = xr.Dataset(
        {"data": (("time", "channel", "lat", "lon"), np.arange(160, dtype=np.float32).reshape(40, 1, 2, 2))},
        coords={"time": times, "channel": ["z500"], "lat": [1.0, 0.0], "lon": [0.0, 1.0]},
    )
    monkeypatch.setattr("xmetai.data.weather.multi_zarr_dataset.xr.open_zarr", lambda *args, **kwargs: source)
    dataset = MultiZarrDataset(["unused"], hist_frames=2, fcst_frames=4, years=("2024-01-01", "2024-01-10"), return_valid_mask=True)
    assert dataset[0]["targets"].shape[0] == 4
    assert dataset.prepare_data(0, forecast_frames=8)["targets"].shape[0] == 4
    dataset.runtime_train_fcst_frames.value = 2
    prepare_data = MultiZarrDataset.prepare_data

    def update_horizon_during_read(sample_dataset, index, *, forecast_frames=None):
        sample_dataset.runtime_train_fcst_frames.value = 3
        return prepare_data(sample_dataset, index, forecast_frames=forecast_frames)

    with monkeypatch.context() as patch:
        patch.setattr(MultiZarrDataset, "prepare_data", update_horizon_during_read)
        assert [sample["targets"].shape[0] for sample in dataset.__getitems__([0, 1])] == [2, 2]

    class LocalModel(torch.nn.Module):
        device = "cpu"
        dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.batches = []

        def forward(self, batch):
            frames = batch["future_frames"]
            assert batch["targets"].shape[0] == 2
            assert batch["targets"].shape[1] >= frames
            assert batch["target_valid_mask"].shape == batch["targets"].shape
            self.batches.append((batch["times"].tolist(), frames))
            return {"loss": self.scale * batch["targets"][:, :frames].mean()}

    def train_rank(rank, sampler=None, workers=num_workers):
        monkeypatch.setattr("xmetai.data.sampler.comm.get_rank", lambda: rank)
        shared_indices = sampler is None
        loader = build_train_loader(dataset, total_batch_size=8, num_workers=workers, pin_memory=False, shared_indices=shared_indices, sampler=sampler)
        if shared_indices:
            assert type(loader.sampler) is TrainingSampler
        else:
            assert loader.sampler is sampler
        model = LocalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        teacher_forcing = OmegaConf.create(dict(teach_iters=5, teach_start=2, teach_steps=4))
        trainer = ExpandTrainer(model, loader, optimizer, use_ema=False, teacher_forcing=teacher_forcing, ens_train=True)
        trainer.train(start_iter=1, max_iter=15)
        return model.batches

    first = train_rank(0)
    other = train_rank(3)
    assert other == first
    scheduled = [2] * 4 + [3] * 5 + [4] * 5
    assert [frames for _, frames in first] == scheduled
    if num_workers == 0:
        assert [frames for _, frames in train_rank(0, torch.utils.data.SequentialSampler(dataset))] == scheduled


def _run_ranks_with_mismatched_prefetch(rank, init_file):
    import torch.distributed as dist

    class PrefetchedLoader:
        dataset = SimpleNamespace(runtime_train_fcst_frames=SimpleNamespace(value=4), max_train_fcst_frames=4)

        def __iter__(self):
            assert self.dataset.runtime_train_fcst_frames.value == 3
            available = (3, 3, 3) if rank == 0 else (2, 3, 3)
            for frames in available:
                yield {"targets": torch.ones(1, frames)}

    class LocalModel(torch.nn.Module):
        device = "cpu"
        dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.frames = []

        def forward(self, batch):
            self.frames.append(batch["future_frames"])
            return {"loss": self.scale * batch["targets"][:, : batch["future_frames"]].mean()}

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        model = LocalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        teacher_forcing = dict(teach_iters=20, teach_start=2, teach_steps=4)
        trainer = ExpandTrainer(model, PrefetchedLoader(), optimizer, use_ema=False, teacher_forcing=teacher_forcing)
        trainer.train(start_iter=19, max_iter=22)
        assert model.frames == [2, 3, 3]
    finally:
        dist.destroy_process_group()


def test_rollout_uses_same_ready_horizon_across_ranks(tmp_path):
    torch.multiprocessing.spawn(_run_ranks_with_mismatched_prefetch, args=(str(tmp_path / "gloo"),), nprocs=2)
