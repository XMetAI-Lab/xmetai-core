import torch

from xmetai.core.evaluator import inference_context


def test_eval_cached_tensors_remain_usable_by_training_backward():
    model = torch.nn.Linear(2, 2).train()
    with inference_context(model):
        cached = torch.ones(1, 2)
        assert not torch.is_grad_enabled()
    assert model.training
    model(cached).sum().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
