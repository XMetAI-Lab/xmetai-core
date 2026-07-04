from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from xmetai.core import DatasetEvaluator
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = [
    "ClimatePSScore",
    "PS",
]


@METRICS.register()
class ClimatePSScore(DatasetEvaluator):
    """Climate business PS score per 《月、季气候趋势预测业务产品检验方法》.

    Formula::

        PS = (2·N0 + 2·N1 + 4·N2) / (N + N0 + 2·N1 + 4·N2) × 100

    Anomaly level classification (inputs must be anomaly percentage in fractional form,
    e.g. 0.20 represents 20%):

    - Normal:  ``|anom| < threshold_l1``
    - Level-1: ``threshold_l1 ≤ |anom| < threshold_l2``
    - Level-2: ``|anom| ≥ threshold_l2``

    Counts:

    - **N0**: grid points where predicted and observed anomaly share the same sign (trend correct).
    - **N1**: N0 subset where both observed and predicted are ≥ level-1 anomaly.
    - **N2**: N1 subset where both observed and predicted are level-2 anomaly.
    - **N**: total grid points per sample.

    Counts are accumulated across all processed batches before computing PS, so the score
    is consistent regardless of how the data is split into batches.

    Args:
        test_names: Variable / channel names to evaluate (length must be ≤ C).
        test_times: Lead-time labels (integers) aligned with the T dimension of the input.
        threshold_l1: Level-1 anomaly threshold in fractional form (default ``0.20`` = 20 %).
        threshold_l2: Level-2 anomaly threshold in fractional form (default ``0.50`` = 50 %).
        output_key: Key for predicted anomaly tensor in the ``outputs`` dict.
        target_key: Key for observed anomaly tensor in the ``outputs`` dict.
    """

    def __init__(
        self,
        test_names: Sequence[str],
        test_times: Sequence[int],
        threshold_l1: float = 0.20,
        threshold_l2: float = 0.50,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.test_names = list(test_names)
        self.test_times = tuple(int(t) for t in test_times)
        self.threshold_l1 = float(threshold_l1)
        self.threshold_l2 = float(threshold_l2)
        self.output_key = output_key
        self.target_key = target_key

        if len(self.test_names) == 0:
            raise ValueError("test_names must not be empty")
        if len(self.test_times) == 0:
            raise ValueError("test_times must not be empty")
        if not (0.0 < self.threshold_l1 < self.threshold_l2):
            raise ValueError(f"thresholds must satisfy 0 < threshold_l1 < threshold_l2, " f"got threshold_l1={threshold_l1}, threshold_l2={threshold_l2}")

    def reset(self) -> None:
        self._counts: dict[str, dict[int, dict[str, float]]] = {
            name: {t: {"N0": 0.0, "N1": 0.0, "N2": 0.0, "N": 0.0} for t in self.test_times} for name in self.test_names
        }

    @staticmethod
    def _classify_level(anom_abs: torch.Tensor, l1: float, l2: float) -> torch.Tensor:
        """Return an integer level tensor: 0 = normal, 1 = level-1, 2 = level-2."""
        level = torch.zeros_like(anom_abs, dtype=torch.int32)
        level[anom_abs >= l1] = 1
        level[anom_abs >= l2] = 2
        return level

    @classmethod
    def _accumulate_counts(
        cls,
        pred: torch.Tensor,
        obs: torch.Tensor,
        l1: float,
        l2: float,
    ) -> tuple[float, float, float, float]:
        """Compute ``(N0, N1, N2, N)`` for a pair of flat 1-D tensors."""
        pred_level = cls._classify_level(pred.abs(), l1, l2)
        obs_level = cls._classify_level(obs.abs(), l1, l2)
        same_sign = pred.sign() == obs.sign()
        n0 = float(same_sign.sum().item())
        n1 = float(((obs_level >= 1) & (pred_level >= 1) & same_sign).sum().item())
        n2 = float(((obs_level == 2) & (pred_level == 2) & same_sign).sum().item())
        n = float(pred.numel())
        return n0, n1, n2, n

    @staticmethod
    def _ps_formula(n0: float, n1: float, n2: float, n: float) -> float:
        """Compute PS from accumulated counts.  Returns 0 when ``n == 0``."""
        if n <= 0.0:
            return 0.0
        numerator = 2.0 * n0 + 2.0 * n1 + 4.0 * n2
        denominator = n + n0 + 2.0 * n1 + 4.0 * n2
        return float(numerator / denominator * 100.0)

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        output = outputs[self.output_key]
        target = outputs[self.target_key]

        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"ClimatePSScore expects output/target as (B, T, C, H, W), " f"got {tuple(output.shape)} and {tuple(target.shape)}")
        if output.shape != target.shape:
            raise ValueError(f"output and target shapes must match, " f"got {tuple(output.shape)} and {tuple(target.shape)}")
        if output.shape[1] != len(self.test_times):
            raise ValueError(f"output has {output.shape[1]} lead times but test_times has {len(self.test_times)} entries")
        if output.shape[2] < len(self.test_names):
            raise ValueError(f"output has {output.shape[2]} channels but test_names has {len(self.test_names)} entries")

        output = output.detach().float()
        target = target.detach().float()
        if output.is_cuda:
            output = output.cpu()
            target = target.cpu()

        for t_idx, t_val in enumerate(self.test_times):
            for c_idx, name in enumerate(self.test_names):
                pred_flat = output[:, t_idx, c_idx].reshape(-1)
                obs_flat = target[:, t_idx, c_idx].reshape(-1)
                n0, n1, n2, n = self._accumulate_counts(pred_flat, obs_flat, self.threshold_l1, self.threshold_l2)
                self._counts[name][t_val]["N0"] += n0
                self._counts[name][t_val]["N1"] += n1
                self._counts[name][t_val]["N2"] += n2
                self._counts[name][t_val]["N"] += n

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered = comm.gather(self._counts, dst=0)
        if not comm.is_main_process():
            return None

        merged: dict[str, dict[int, dict[str, float]]] = {
            name: {t: {"N0": 0.0, "N1": 0.0, "N2": 0.0, "N": 0.0} for t in self.test_times} for name in self.test_names
        }
        for item in gathered:
            for name in self.test_names:
                for t_val in self.test_times:
                    for key in ("N0", "N1", "N2", "N"):
                        merged[name][t_val][key] += item[name][t_val][key]

        lead_labels: list[str] = [str(t) for t in self.test_times] + ["Overall"]
        results: dict[str, list[float] | list[str]] = {"Lead\\PS": lead_labels}

        for name in self.test_names:
            lead_ps: list[float] = []
            n0_total = n1_total = n2_total = n_total = 0.0
            for t_val in self.test_times:
                c = merged[name][t_val]
                lead_ps.append(self._ps_formula(c["N0"], c["N1"], c["N2"], c["N"]))
                n0_total += c["N0"]
                n1_total += c["N1"]
                n2_total += c["N2"]
                n_total += c["N"]
            overall_ps = self._ps_formula(n0_total, n1_total, n2_total, n_total)
            results[name] = lead_ps + [overall_ps]

        return {"PS": results}


@METRICS.register()
class PS(ClimatePSScore):
    pass
