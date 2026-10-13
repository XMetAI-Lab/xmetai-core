"""Area-weighted amplitude and spherical spectral diagnostics for global fields."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from xmetai.core import DatasetEvaluator
from xmetai.loss import SphericalBandPowerLoss
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = ["ForecastFidelity"]


@METRICS.register()
class ForecastFidelity(DatasetEvaluator):
    """Physical-unit bias, spatial RMS amplitude ratio, and band power ratios.

    Accumulates sufficient statistics across cases and ranks before dividing.
    Spatial means are removed separately for each case; this is not a temporal
    anomaly metric. Zero target energy produces NaN (undefined), not a ratio of
    one. Inputs must cover the complete global grid, without spatial masks.
    """

    def __init__(self, test_names: Sequence[str], test_times: Sequence[int], image_size: tuple[int, int]) -> None:
        self.test_names = tuple(test_names)
        self.test_times = tuple(test_times)
        self.spectrum = SphericalBandPowerLoss(image_size)
        self.reset()

    def reset(self) -> None:
        self.statistics: dict[tuple[str, int], NDArray[np.float64]] = {}

    @torch.no_grad()
    def process(self, inputs: Any, outputs: dict[str, Any]) -> None:
        target = outputs["target"].detach().float()
        frame_start = int(outputs.get("frame_start", 0))
        modes = outputs["output_by_mode"] if "output_by_mode" in outputs else {"": outputs["output"]}
        self.spectrum.to(target.device)
        weights = self.spectrum.latitude_weights
        for mode, prediction in modes.items():
            prediction = prediction.detach().float()
            if prediction.shape != target.shape or target.ndim != 5 or target.shape[2] != len(self.test_names):
                raise ValueError("ForecastFidelity expects matching B,T,C,H,W tensors with C=len(test_names)")
            if frame_start < 0 or frame_start + target.shape[1] > len(self.test_times):
                raise ValueError("ForecastFidelity frames exceed configured test_times")
            # Keep only scalar sufficient statistics, not full-resolution fields.
            pred_mean = (prediction.mean(-1) * weights).sum(-1)
            true_mean = (target.mean(-1) * weights).sum(-1)
            pred_var = ((prediction - pred_mean[..., None, None]).square().mean(-1) * weights).sum(-1)
            true_var = ((target - true_mean[..., None, None]).square().mean(-1) * weights).sum(-1)
            pred_bands = self.spectrum.band_power(prediction)
            true_bands = self.spectrum.band_power(target)
            stats = (
                torch.cat(
                    [
                        torch.ones_like(pred_mean)[..., None],
                        (pred_mean - true_mean)[..., None],
                        pred_var[..., None],
                        true_var[..., None],
                        pred_bands,
                        true_bands,
                    ],
                    dim=-1,
                )
                .double()
                .sum(0)
                .cpu()
                .numpy()
            )
            for index, values in enumerate(stats):
                key = (mode, frame_start + index)
                if key not in self.statistics:
                    self.statistics[key] = np.zeros_like(values)
                self.statistics[key] += values

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered = comm.gather(self.statistics, dst=0)
        if not comm.is_main_process():
            return None
        totals = {}
        for rank in gathered:
            for key, values in rank.items():
                if key not in totals:
                    totals[key] = np.zeros_like(values)
                totals[key] += values
        results = {}
        nbands = len(self.spectrum.bands)
        names = ["Bias", "SpatialAmplitudeRatio"] + [f"BandPowerRatio_{low}_{high}" for low, high in self.spectrum.bands]
        for mode in sorted({key[0] for key in totals}):
            frames = sorted(index for prefix, index in totals if prefix == mode)
            values = np.stack([totals[mode, index] for index in frames])

            def ratio(numerator, denominator):
                return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)

            metrics = [values[..., 1] / values[..., 0], np.sqrt(ratio(values[..., 2], values[..., 3]))]
            metrics.extend(ratio(values[..., 4 + i], values[..., 4 + nbands + i]) for i in range(nbands))
            for name, metric in zip(names, metrics):
                key = f"{mode}/{name}" if mode else name
                results[key] = {"Times": [self.test_times[index] for index in frames]}
                results[key].update({channel: metric[:, i].tolist() for i, channel in enumerate(self.test_names)})
        return results
