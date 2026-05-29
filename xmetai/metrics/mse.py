import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from xmetai.core import DatasetEvaluator
from xmetai.layers import to_1tuple
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = [
    "MSE",
]


@METRICS.register()
class MSE(DatasetEvaluator):
    def __init__(
        self,
        test_names: Sequence[str],
        test_times: Sequence[int],
        top_percent: float = 0.0,
    ) -> None:
        super().__init__()
        self.test_names = test_names
        self.test_times = to_1tuple(test_times)
        self.top_percent = float(top_percent)
        if not 0.0 <= self.top_percent < 1.0:
            raise ValueError("top_percent must be in [0, 1).")

    def get_key(self, name: str, t: int) -> str:
        return "{} @ {:03d}h".format(name, t)

    def get_top_key(self, name: str, t: int) -> str:
        return "{} top{:g}% @ {:03d}h".format(name, self.top_percent * 100.0, t)

    def reset(self) -> None:
        self.rmse: dict[str, list[float]] = {}
        self.top_rmse: dict[str, list[float]] = {}

        for t in self.test_times:
            for name in self.test_names:
                key = self.get_key(name, t)
                self.rmse[key] = []
                if self.top_percent > 0.0:
                    self.top_rmse[self.get_top_key(name, t)] = []

    def _process_output(self, output: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | float, prefix: str = "") -> None:
        metric_prefix = f"{prefix}/" if prefix else ""
        for i, t in enumerate(self.test_times):
            for j, name in enumerate(self.test_names):
                output_j = output[:, i, j]
                target_j = target[:, i, j]

                err = F.mse_loss(target_j, output_j, reduction="none")
                rmse = (err * weight).mean().sqrt()
                key = metric_prefix + self.get_key(name, t)
                self.rmse.setdefault(key, [])
                self.rmse[key].append(rmse.item())
                if self.top_percent > 0.0:
                    flat_target = target_j.abs().flatten(1)
                    top_k = max(1, int(flat_target.shape[1] * self.top_percent))
                    threshold = flat_target.topk(top_k, dim=1).values[:, -1].view(-1, 1, 1)
                    extreme_mask = target_j.abs() >= threshold
                    extreme_weight = weight * extreme_mask
                    denom = extreme_weight.sum().clamp_min(1.0)
                    top_rmse = (err * extreme_weight).sum().div(denom).sqrt()
                    top_key = metric_prefix + self.get_top_key(name, t)
                    self.top_rmse.setdefault(top_key, [])
                    self.top_rmse[top_key].append(top_rmse.item())

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        target = outputs["target"].cpu()

        if "weight" in outputs:
            weight = outputs["weight"].cpu()
        else:
            weight = 1

        if "output_by_mode" in outputs:
            for mode, output in outputs["output_by_mode"].items():
                self._process_output(output.cpu(), target, weight, prefix=mode)
        else:
            self._process_output(outputs["output"].cpu(), target, weight)

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()

        for key in list(self.rmse):
            rmse = comm.gather(self.rmse[key], dst=0)
            self.rmse[key] = list(itertools.chain(*rmse))
            if len(self.rmse[key]) == 0:
                del self.rmse[key]
        if self.top_percent > 0.0:
            for key in list(self.top_rmse):
                top_rmse = comm.gather(self.top_rmse[key], dst=0)
                self.top_rmse[key] = list(itertools.chain(*top_rmse))
                if len(self.top_rmse[key]) == 0:
                    del self.top_rmse[key]

        if not comm.is_main_process():
            return None

        flatten_results = {"Times": self.test_times}

        for name in self.test_names:
            value_keys = [key for key in self.rmse if key.endswith(self.get_key(name, self.test_times[0]))]
            if len(value_keys) == 0:
                continue
            for first_key in value_keys:
                prefix = first_key[: -len(self.get_key(name, self.test_times[0]))].rstrip("/")
                column_name = f"{prefix}/{name}" if prefix else name
                flatten_results[column_name] = []
                for t in self.test_times:
                    key = f"{prefix}/{self.get_key(name, t)}" if prefix else self.get_key(name, t)
                    flatten_results[column_name].append(float(np.mean(self.rmse[key])))
            if self.top_percent > 0.0:
                top_value_keys = [key for key in self.top_rmse if key.endswith(self.get_top_key(name, self.test_times[0]))]
                if len(top_value_keys) == 0:
                    continue
                for first_key in top_value_keys:
                    prefix = first_key[: -len(self.get_top_key(name, self.test_times[0]))].rstrip("/")
                    column_name = f"{prefix}/{name}_top{self.top_percent * 100.0:g}%" if prefix else f"{name}_top{self.top_percent * 100.0:g}%"
                    flatten_results[column_name] = []
                    for t in self.test_times:
                        key = f"{prefix}/{self.get_top_key(name, t)}" if prefix else self.get_top_key(name, t)
                        flatten_results[column_name].append(float(np.mean(self.top_rmse[key])))

        return {"RMSE": flatten_results}
