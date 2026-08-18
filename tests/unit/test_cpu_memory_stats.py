import torch

from xmetai.core.hooks import CPUMemoryStats
from xmetai.core.trainer import SimpleTrainer
from xmetai.utils.memory import model_cpu_memory_bytes, object_cpu_memory_bytes, optimizer_cpu_memory_bytes


def test_object_cpu_memory_bytes_counts_tensors_and_numpy():
    tensor = torch.zeros(2, 3, dtype=torch.float32)
    stats = object_cpu_memory_bytes({"a": tensor, "b": [tensor]})
    assert stats["tensor_cpu_bytes"] == tensor.untyped_storage().nbytes()
    assert stats["total_cpu_bytes"] == stats["tensor_cpu_bytes"]


def test_model_and_optimizer_cpu_memory_bytes():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    data = {"inputs": torch.ones(3, 4), "targets": torch.zeros(3, 2)}
    loss = torch.nn.functional.mse_loss(model(data["inputs"]), data["targets"])
    loss.backward()
    optimizer.step()

    model_stats = model_cpu_memory_bytes(model)
    optimizer_bytes = optimizer_cpu_memory_bytes(optimizer)

    assert model_stats["model_param_cpu_bytes"] > 0
    assert model_stats["model_grad_cpu_bytes"] > 0
    assert optimizer_bytes > 0


def test_cpu_memory_stats_hook_writes_storage_metrics():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [{"inputs": torch.ones(3, 4), "targets": torch.zeros(3, 2)}]

    class Loader:
        def __iter__(self):
            return iter(data)

    class WrappedModel(torch.nn.Module):
        device = "cpu"
        dtype = torch.float32

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, batch):
            pred = self.module(batch["inputs"])
            return {"loss": torch.nn.functional.mse_loss(pred, batch["targets"])}

    trainer = SimpleTrainer(WrappedModel(model), Loader(), optimizer, use_ema=False)
    trainer.register_hooks([CPUMemoryStats(period=1)])
    trainer.train(0, 1)

    assert trainer.storage.history("cpu_memory_data_mb").latest() > 0
    assert trainer.storage.history("cpu_memory_model_mb").latest() > 0
    assert trainer.storage.history("cpu_memory_untracked_or_fragmented_mb").latest() >= 0
