import logging
from abc import ABCMeta
from typing import Collection, Optional

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

    def __init__(self, precision: str = "fp32", device: str = "cuda", export_kwargs: Optional[dict] = None):
        super().__init__()
        self.device = device
        self.dtype = self.dtype_dict[precision]
        self.export_kwargs = export_kwargs if export_kwargs is not None else {}
        self.epoch = 0
        self.logger = logging.getLogger(__name__)
        self._frozen_names: set[str] = set()

    def parallelize(self) -> tuple[Optional[tuple[type[nn.Module], ...]], Optional[tuple[type[nn.Module], ...]]]:
        """Return model AC/FSDP defaults; None inherits the recipe, () selects no modules.

        Explicit training mods override these defaults. Overrides may also install
        model-specific checkpointing independently of the common AC switch.
        """
        return None, None

    def params(self) -> int:
        return sum(map(lambda x: x.numel(), self.parameters()))

    def freeze(self, names: Optional[Collection[str]] = None) -> None:
        if names is None:
            return

        names = tuple(names)
        self._frozen_names.update(names)

        for name in names:
            try:
                m = getattr(self, f"{name}")
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
            except AttributeError:
                for n, p in self.named_parameters():
                    if name in n:
                        p.requires_grad = False
                for module_name, module in self.named_modules():
                    if module_name == name or module_name.startswith(name + "."):
                        module.eval()
            except Exception:
                self.logger.exception(f"Unexpected error while freezing component {name}")
                raise

    def train(self, mode: bool = True) -> "Meta":
        super().train(mode=mode)
        if mode and len(self._frozen_names) > 0:
            self.freeze(self._frozen_names)
        return self
