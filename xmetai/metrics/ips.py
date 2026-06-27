from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from xmetai.core import DatasetEvaluator
from xmetai.layers import to_1tuple
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = [
    "IntegratedPatternScore",
    "IPS",
]


@METRICS.register()
class IntegratedPatternScore(DatasetEvaluator):
    """Integrated Pattern Score for pentad anomaly forecasts.

    The competition definition evaluates lead pentads 3-12 with:
    IPS_j = ((((PCC_j + 1) / 2) + AS_j) / 2) * 100.
    """

    def __init__(
        self,
        test_names: Sequence[str],
        test_times: Sequence[int],
        start_pentad: int = 3,
        end_pentad: int = 12,
        lead_unit: str = "auto",
        pentad_size: int = 5,
        use_spatial_weight: bool = False,
        output_key: str = "output",
        target_key: str = "target",
    ) -> None:
        super().__init__()
        self.test_names = list(test_names)
        self.test_times = tuple(int(t) for t in to_1tuple(test_times))
        self.start_pentad = int(start_pentad)
        self.end_pentad = int(end_pentad)
        self.lead_unit = lead_unit
        self.pentad_size = int(pentad_size)
        self.use_spatial_weight = bool(use_spatial_weight)
        self.output_key = output_key
        self.target_key = target_key
        if len(self.test_times) == 0:
            raise ValueError("test_times must not be empty")
        if self.start_pentad > self.end_pentad:
            raise ValueError("start_pentad must be <= end_pentad")
        if self.lead_unit not in {"auto", "hour", "day", "pentad"}:
            raise ValueError("lead_unit must be one of: auto, hour, day, pentad")
        if self.pentad_size <= 0:
            raise ValueError("pentad_size must be positive")

    def reset(self) -> None:
        self.pentads = tuple(range(self.start_pentad, self.end_pentad + 1))
        self.pentad_indices = self._build_pentad_indices()
        self.scores: dict[str, dict[int, dict[str, list[float]]]] = {
            name: {pentad: {"pcc": [], "as": [], "ips": []} for pentad in self.pentads} for name in self.test_names
        }

    def _effective_lead_unit(self) -> str:
        if self.lead_unit != "auto":
            return self.lead_unit
        # Heuristic: hour-based leads are typically expressed in hours and often day-aligned (multiples of 24).
        if all(lead % 24 == 0 for lead in self.test_times) and max(self.test_times) >= 24:
            return "hour"
        return "day" if max(self.test_times) > self.end_pentad else "pentad"

    def _build_pentad_indices(self) -> dict[int, list[int]]:
        lead_unit = self._effective_lead_unit()
        indices: dict[int, list[int]] = {pentad: [] for pentad in self.pentads}
        for idx, lead in enumerate(self.test_times):
            if lead_unit == "pentad":
                pentad = lead
            else:
                lead_day = (lead + 23) // 24 if lead_unit == "hour" else lead
                pentad = (lead_day - 1) // self.pentad_size + 1
            if pentad in indices:
                indices[pentad].append(idx)
        return indices

    @staticmethod
    def _spatial_weight(weight: Any, reference: torch.Tensor) -> torch.Tensor | None:
        if weight is None:
            return None
        weight_tensor = torch.as_tensor(weight, dtype=reference.dtype, device=reference.device).squeeze()
        h, w = reference.shape[-2:]
        if weight_tensor.dim() == 0:
            weight_tensor = weight_tensor.expand(h, w)
        elif weight_tensor.dim() == 1:
            weight_tensor = weight_tensor.unsqueeze(1).expand(h, w)
        elif weight_tensor.dim() == 2:
            weight_tensor = weight_tensor.expand(h, w)
        else:
            raise ValueError(f"weight must squeeze to scalar, (H,), or (H, W), got {tuple(weight_tensor.shape)}")
        return weight_tensor.contiguous()

    @staticmethod
    def _weighted_mean(values: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if weight is None:
            return values[valid].mean()
        used_weight = weight[valid]
        return (values[valid] * used_weight).sum() / used_weight.sum().clamp_min(1.0e-8)

    @classmethod
    def _pattern_scores(cls, pred: torch.Tensor, true: torch.Tensor, weight: torch.Tensor | None) -> tuple[float, float, float]:
        valid = torch.isfinite(pred) & torch.isfinite(true)
        if weight is not None:
            valid = valid & torch.isfinite(weight) & (weight > 0)
        if not valid.any():
            return 0.0, 0.0, 0.0

        pred_valid = pred[valid]
        true_valid = true[valid]
        if torch.allclose(pred_valid, true_valid):
            pcc = pred.new_tensor(1.0)
        else:
            pred_mean = cls._weighted_mean(pred, valid, weight)
            true_mean = cls._weighted_mean(true, valid, weight)
            pred_anom = pred - pred_mean
            true_anom = true - true_mean
            pred_var = cls._weighted_mean(pred_anom * pred_anom, valid, weight)
            true_var = cls._weighted_mean(true_anom * true_anom, valid, weight)
            denom = torch.sqrt(pred_var * true_var)
            if denom <= 1.0e-8:
                pcc = pred.new_tensor(0.0)
            else:
                cov = cls._weighted_mean(pred_anom * true_anom, valid, weight)
                pcc = (cov / denom).clamp(-1.0, 1.0)

        sign_match = ((pred >= 0) & (true >= 0)) | ((pred < 0) & (true < 0))
        as_score = cls._weighted_mean(sign_match.to(pred.dtype), valid, weight)
        ips = ((((pcc + 1.0) / 2.0) + as_score) / 2.0) * 100.0
        return float(pcc.item()), float(as_score.item()), float(ips.item())

    @staticmethod
    def _lead_weight(pentad: int) -> float:
        if 3 <= pentad <= 4:
            return 5.0
        if 5 <= pentad <= 6:
            return 4.0
        if 7 <= pentad <= 8:
            return 3.0
        if 9 <= pentad <= 10:
            return 2.0
        if 11 <= pentad <= 12:
            return 1.0
        return 1.0

    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        output = outputs[self.output_key]
        target = outputs[self.target_key]
        if output.ndim != 5 or target.ndim != 5:
            raise ValueError(f"IPS expects output/target as (B, T, C, H, W), got {tuple(output.shape)} and {tuple(target.shape)}")
        if output.shape != target.shape:
            raise ValueError(f"output and target shapes must match, got {tuple(output.shape)} and {tuple(target.shape)}")
        if output.shape[1] != len(self.test_times):
            raise ValueError(f"output has {output.shape[1]} lead times but test_times has {len(self.test_times)} entries")
        if output.shape[2] < len(self.test_names):
            raise ValueError(f"output has {output.shape[2]} channels but test_names has {len(self.test_names)} entries")

        output = output.detach()
        target = target.detach()
        if output.is_cuda:
            output = output.cpu()
            target = target.cpu()
        spatial_weight = self._spatial_weight(outputs.get("weight"), output) if self.use_spatial_weight else None

        for pentad, indices in self.pentad_indices.items():
            if len(indices) == 0:
                continue
            pred_pentad = output[:, indices].mean(dim=1)
            true_pentad = target[:, indices].mean(dim=1)
            for channel_idx, name in enumerate(self.test_names):
                for batch_idx in range(pred_pentad.shape[0]):
                    pcc, as_score, ips = self._pattern_scores(
                        pred_pentad[batch_idx, channel_idx],
                        true_pentad[batch_idx, channel_idx],
                        spatial_weight,
                    )
                    self.scores[name][pentad]["pcc"].append(pcc)
                    self.scores[name][pentad]["as"].append(as_score)
                    self.scores[name][pentad]["ips"].append(ips)

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered = comm.gather(self.scores, dst=0)
        if not comm.is_main_process():
            return None

        merged: dict[str, dict[int, dict[str, list[float]]]] = {
            name: {pentad: {"pcc": [], "as": [], "ips": []} for pentad in self.pentads} for name in self.test_names
        }
        for item in gathered:
            for name in self.test_names:
                for pentad in self.pentads:
                    for key in ("pcc", "as", "ips"):
                        merged[name][pentad][key].extend(item[name][pentad][key])

        results: dict[str, list[float] | list[str]] = {"Pentad\\IPS": [f"P{pentad}" for pentad in self.pentads] + ["Overall"]}
        for name in self.test_names:
            lead_ips = []
            lead_pcc = []
            lead_as = []
            weighted_ips_sum = 0.0
            weight_sum = 0.0
            for pentad in self.pentads:
                pcc_values = merged[name][pentad]["pcc"]
                as_values = merged[name][pentad]["as"]
                ips_values = merged[name][pentad]["ips"]
                avg_pcc = float(np.mean(pcc_values)) if pcc_values else 0.0
                avg_as = float(np.mean(as_values)) if as_values else 0.0
                avg_ips = float(np.mean(ips_values)) if ips_values else 0.0
                lead_pcc.append(avg_pcc)
                lead_as.append(avg_as)
                lead_ips.append(avg_ips)
                if ips_values:
                    lead_weight = self._lead_weight(pentad)
                    weighted_ips_sum += lead_weight * avg_ips
                    weight_sum += lead_weight

            overall_ips = weighted_ips_sum / weight_sum if weight_sum > 0 else 0.0
            results[f"{name}_PCC"] = lead_pcc + [0.0]
            results[f"{name}_AS"] = lead_as + [0.0]
            results[name] = lead_ips + [float(overall_ips)]

        return {"IPS": results}


@METRICS.register()
class IPS(IntegratedPatternScore):
    pass
