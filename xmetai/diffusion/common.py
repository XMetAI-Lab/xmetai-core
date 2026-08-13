"""Shared utilities for diffusion schedulers.

Ported from diffusers (Apache 2.0), stripped of framework dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

__all__ = ["SchedulerOutput", "betas_for_alpha_bar", "rescale_zero_terminal_snr", "threshold_sample"]


@dataclass
class SchedulerOutput:
    """Output of a scheduler's step function."""

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor | None = None


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
    alpha_transform_type: Literal["cosine", "exp", "laplace"] = "cosine",
) -> torch.Tensor:
    """Create a beta schedule that discretizes the given alpha_t_bar function.

    Ported from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar.
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "laplace":

        def alpha_bar_fn(t):
            lmb = -0.5 * math.copysign(1, 0.5 - t) * math.log(1 - 2 * math.fabs(0.5 - t) + 1e-6)
            snr = math.exp(lmb)
            return math.sqrt(snr / (1 + snr))

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescales betas to have zero terminal SNR.

    Based on https://huggingface.co/papers/2305.08891 (Algorithm 1).
    Ported from diffusers.schedulers.scheduling_ddpm.rescale_zero_terminal_snr.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt -= alphas_bar_sqrt_T
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    alphas_bar = alphas_bar_sqrt**2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def threshold_sample(
    sample: torch.Tensor,
    dynamic_thresholding_ratio: float = 0.995,
    sample_max_value: float = 1.0,
) -> torch.Tensor:
    """Apply dynamic thresholding to the predicted sample.

    Ported from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample.
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape

    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()

    sample = sample.reshape(batch_size, channels * int(np.prod(remaining_dims)))
    abs_sample = sample.abs()

    s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(s, min=1, max=sample_max_value)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s

    sample = sample.reshape(batch_size, channels, *remaining_dims)
    sample = sample.to(dtype)
    return sample
