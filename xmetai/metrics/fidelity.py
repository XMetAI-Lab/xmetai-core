"""Area-weighted amplitude and spherical spectral diagnostics for global fields."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from xmetai.core import DatasetEvaluator
from xmetai.registries import METRICS
from xmetai.utils import comm

__all__ = ["ForecastFidelity"]


class _SphericalBandPower(nn.Module):
    BANDS = ((1, 4), (5, 20), (21, 40), (41, 64), (65, 128))

    def __init__(self, image_size: tuple[int, int]) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.bands = self.BANDS
        nlat, nlon = self.image_size
        self.lmax = self.bands[-1][1]
        n = nlat - 1
        if n % 2 or 2 * self.lmax >= n or self.lmax >= nlon // 2:
            raise ValueError("Need an odd latitude count with 2*lmax < H-1 and lmax < W//2")
        theta = torch.arange(nlat, dtype=torch.float64) * (math.pi / n)
        weights = torch.zeros(nlat, dtype=torch.float64)
        weights[[0, -1]] = 1 / (n * n - 1)
        interior = torch.ones(n - 1, dtype=torch.float64)
        for k in range(1, n // 2):
            interior -= 2 * torch.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
        interior -= torch.cos(n * theta[1:-1]) / (n * n - 1)
        weights[1:-1] = 2 * interior / n
        basis = torch.zeros(self.lmax + 1, self.lmax + 1, nlat, dtype=torch.float64)
        basis[0, 0] = 1 / math.sqrt(4 * math.pi)
        x, sint = theta.cos(), theta.sin()
        for m in range(self.lmax + 1):
            if m:
                basis[m, m] = -math.sqrt((2 * m + 1) / (2 * m)) * sint * basis[m - 1, m - 1]
            if m < self.lmax:
                basis[m + 1, m] = math.sqrt(2 * m + 3) * x * basis[m, m]
            for degree in range(m + 2, self.lmax + 1):
                a = math.sqrt((4 * degree * degree - 1) / (degree * degree - m * m))
                b = math.sqrt(((degree - 1) ** 2 - m * m) / (4 * (degree - 1) ** 2 - 1))
                basis[degree, m] = a * (x * basis[degree - 1, m] - b * basis[degree - 2, m])
        multiplicity = torch.full((self.lmax + 1,), 2.0)
        multiplicity[0] = 1.0
        self.register_buffer("latitude_weights", (weights / 2).float(), persistent=False)
        self.register_buffer("weighted_basis", (basis * weights).float(), persistent=False)
        self.register_buffer("multiplicity", multiplicity, persistent=False)

    def band_power(self, field: torch.Tensor) -> torch.Tensor:
        leading_shape = field.shape[:-2]
        field = field.float().reshape(-1, *self.image_size)
        mean = (field.mean(dim=-1) * self.latitude_weights).sum(dim=-1)
        field = field - mean[:, None, None]
        fft = torch.fft.rfft(field, dim=-1, norm="forward")[..., : self.lmax + 1]
        real = torch.einsum("lmh,nhm->nlm", self.weighted_basis, fft.real)
        imag = torch.einsum("lmh,nhm->nlm", self.weighted_basis, fft.imag)
        power = ((real.square() + imag.square()) * self.multiplicity).sum(dim=-1) * (2 * math.pi) ** 2
        bands = torch.stack([power[:, low : high + 1].sum(-1) for low, high in self.bands], dim=-1)
        return bands.reshape(*leading_shape, len(self.bands))


@METRICS.register()
class ForecastFidelity(DatasetEvaluator):
    """Physical-unit bias, spatial RMS amplitude ratio, and band power ratios."""

    BANDS = _SphericalBandPower.BANDS

    def __init__(self, test_names: Sequence[str], test_times: Sequence[int], image_size: tuple[int, int]) -> None:
        self.test_names = tuple(test_names)
        self.test_times = tuple(test_times)
        _SphericalBandPower.BANDS = self.BANDS
        self.spectrum = _SphericalBandPower(image_size)
        self.reset()

    def reset(self) -> None:
        self.statistics: dict[tuple[str, int], NDArray[np.float64]] = {}

    def _zonal_band_power(self, field: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=field.device.type, enabled=False):
            field = field.float() - field.float().mean(-1, keepdim=True)
            power = torch.fft.rfft(field, dim=-1, norm="forward").abs().square()
            latitude = torch.linspace(-torch.pi / 2, torch.pi / 2, field.shape[-2], device=field.device)
            weights = latitude.cos().clamp_min(0)
            power = (power * weights[:, None]).sum(-2) / weights.sum()
            return torch.stack([power[..., low : high + 1].sum(-1) for low, high in self.spectrum.bands], -1)

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
            pred_mean = (prediction.mean(-1) * weights).sum(-1)
            true_mean = (target.mean(-1) * weights).sum(-1)
            pred_var = ((prediction - pred_mean[..., None, None]).square().mean(-1) * weights).sum(-1)
            true_var = ((target - true_mean[..., None, None]).square().mean(-1) * weights).sum(-1)
            stats = torch.cat([
                torch.ones_like(pred_mean)[..., None],
                (pred_mean - true_mean)[..., None], pred_var[..., None], true_var[..., None],
                self.spectrum.band_power(prediction), self.spectrum.band_power(target),
                self._zonal_band_power(prediction), self._zonal_band_power(target),
            ], dim=-1).double().sum(0).cpu().numpy()
            for index, values in enumerate(stats):
                key = (mode, frame_start + index)
                self.statistics[key] = self.statistics.get(key, np.zeros_like(values)) + values

    def evaluate(self) -> dict[str, Any] | None:
        comm.synchronize()
        gathered = comm.gather(self.statistics, dst=0)
        if not comm.is_main_process():
            return None
        totals = {}
        for rank in gathered:
            for key, values in rank.items():
                totals[key] = totals.get(key, np.zeros_like(values)) + values
        results = {}
        nbands = len(self.spectrum.bands)
        names = ["Bias", "SpatialAmplitudeRatio"] + [f"BandPowerRatio_{low}_{high}" for low, high in self.spectrum.bands]
        for mode in sorted({key[0] for key in totals}):
            frames = sorted(index for prefix, index in totals if prefix == mode)
            values = np.stack([totals[mode, index] for index in frames])
            ratio = lambda numerator, denominator: np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)
            metrics = [values[..., 1] / values[..., 0], np.sqrt(ratio(values[..., 2], values[..., 3]))]
            metrics.extend(ratio(values[..., 4 + i], values[..., 4 + nbands + i]) for i in range(nbands))
            metric_names = list(names)
            for i, (low, high) in enumerate(self.spectrum.bands):
                pred, obs = values[..., 4 + 2 * nbands + i], values[..., 4 + 3 * nbands + i]
                metric_names.extend([f"ZonalPowerPred_{low}_{high}", f"ZonalPowerObs_{low}_{high}", f"ZonalPowerRatio_{low}_{high}"])
                metrics.extend([pred / values[..., 0], obs / values[..., 0], ratio(pred, obs)])
            for name, metric in zip(metric_names, metrics):
                key = f"{mode}/{name}" if mode else name
                results[key] = {"Times": [self.test_times[index] for index in frames]}
                results[key].update({channel: metric[:, i].tolist() for i, channel in enumerate(self.test_names)})
        return results