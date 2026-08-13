"""DDIM (Denoising Diffusion Implicit Models) scheduler.

Faithful port of diffusers.schedulers.scheduling_ddim.DDIMScheduler,
stripped of diffusers framework dependencies.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ..common import SchedulerOutput, betas_for_alpha_bar, rescale_zero_terminal_snr, threshold_sample

__all__ = ["DDIMScheduler"]


class DDIMScheduler:
    """DDIM scheduler with non-Markovian guidance for faster sampling.

    Supports: linear, scaled_linear, squaredcos_cap_v2 beta schedules.
    Supports: epsilon, sample, v_prediction prediction types.
    The `eta` parameter in step() controls stochasticity: 0=deterministic (DDIM), 1=DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: Literal["leading", "trailing", "linspace"] = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.clip_sample_range = clip_sample_range
        self.sample_max_value = sample_max_value
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        return sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for a given diffusion step."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """Set the discrete timesteps for inference."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:" f" {self.num_train_timesteps}.")

        self.num_inference_steps = num_inference_steps
        assert self.num_inference_steps is not None

        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
        elif self.timestep_spacing == "leading":
            step_ratio: float = self.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"{self.timestep_spacing} is not supported.")

        self.timesteps = torch.from_numpy(timesteps).to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
    ) -> SchedulerOutput:
        """Predict the sample from the previous timestep.

        Args:
            model_output: Direct output from the diffusion model.
            timestep: Current discrete timestep.
            sample: Current sample from the diffusion process.
            eta: Weight of noise (0=DDIM deterministic, 1=DDPM stochastic).
            use_clipped_model_output: If True, re-derive epsilon from clipped x_0.
            generator: Random number generator for reproducibility.
            variance_noise: Pre-generated noise (alternative to generator).
        """
        if self.num_inference_steps is None:
            raise ValueError("Run 'set_timesteps' after creating the scheduler")

        # 1. get previous step
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample and epsilon
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            pred_epsilon = alpha_prod_t**0.5 * model_output + beta_prod_t**0.5 * sample
        else:
            raise ValueError(f"prediction_type must be epsilon, sample, or v_prediction, got {self.prediction_type}")

        # 4. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = threshold_sample(pred_original_sample, self.dynamic_thresholding_ratio, self.sample_max_value)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 5. compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance**0.5

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

        # 6. compute "direction pointing to x_t"
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon

        # 7. compute x_{t-1}
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is None:
                variance_noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample + std_dev_t * variance_noise

        return SchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor | torch.LongTensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to original samples."""
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor | torch.LongTensor) -> torch.Tensor:
        """Compute the velocity prediction from sample and noise."""
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample

    def __len__(self) -> int:
        return self.num_train_timesteps
