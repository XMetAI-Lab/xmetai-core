from abc import ABCMeta

import torch
import torch.nn as nn

__all__ = ["Meta"]


class Meta(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for model.
    """

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    def __init__(self, precision="fp32", device="cuda", export_kwargs={}):
        super().__init__()
        self.device = device
        self.dtype = self.dtype_dict[precision]
        self.export_kwargs = export_kwargs
        self.epoch = 0

    def params(self):
        return sum(map(lambda x: x.numel(), self.parameters()))

    def freeze(self, names=set()):
        for name in names:
            m = getattr(self, f"{name}")
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode=mode)
