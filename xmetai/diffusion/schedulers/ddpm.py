"""DDPM (Denoising Diffusion Probabilistic Models) scheduler.

Faithful port of diffusers.schedulers.scheduling_ddpm.DDPMScheduler,
stripped of diffusers framework dependencies (ConfigMixin, SchedulerMixin, etc.).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ..common import SchedulerOutput, betas_for_alpha_bar, rescale_zero_terminal_snr, threshold_sample

__all__ = ["DDPMScheduler"]


class DDPMScheduler:
    """DDPM noise scheduler with reverse diffusion sampling.

    Supports: linear, scaled_linear, squaredcos_cap_v2, laplace, sigmoid beta schedules.
    Supports: fixed_small, fixed_small_log, fixed_large, fixed_large_log, learned, learned_range variance types.
    Supports: epsilon, sample, v_prediction prediction types.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"] = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        variance_type: Literal["fixed_small", "fixed_small_log", "fixed_large", "fixed_large_log", "learned", "learned_range"] = "fixed_small",
        clip_sample: bool = True,
        prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: Literal["linspace", "leading", "trailing"] = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        # Store config
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.clip_sample_range = clip_sample_range
        self.sample_max_value = sample_max_value
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        # Compute betas
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "laplace":
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="laplace")
        elif beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.init_noise_sigma = 1.0

        self.custom_timesteps = False
        self.num_inference_steps: int | None = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
    ) -> None:
        """Set the discrete timesteps for inference."""
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`timesteps` must be in descending order.")
            if timesteps[0] >= self.num_train_timesteps:
                raise ValueError(f"`timesteps` must start before `num_train_timesteps`: {self.num_train_timesteps}.")
            timesteps_arr = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            assert num_inference_steps is not None
            if num_inference_steps > self.num_train_timesteps:
                raise ValueError(f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:" f" {self.num_train_timesteps}.")
            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False
            assert self.num_inference_steps is not None

            if self.timestep_spacing == "linspace":
                timesteps_arr = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            elif self.timestep_spacing == "leading":
                step_ratio: float = self.num_train_timesteps // self.num_inference_steps
                timesteps_arr = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
                timesteps_arr += self.steps_offset
            elif self.timestep_spacing == "trailing":
                step_ratio = self.num_train_timesteps / self.num_inference_steps
                timesteps_arr = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
                timesteps_arr -= 1
            else:
                raise ValueError(f"{self.timestep_spacing} is not supported.")

        self.timesteps = torch.from_numpy(timesteps_arr).to(device)

    def _get_variance(
        self,
        t: int,
        predicted_variance: torch.Tensor | None = None,
        variance_type: str | None = None,
    ) -> torch.Tensor:
        """Compute variance for a given timestep."""
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        vtype = variance_type if variance_type is not None else self.variance_type

        if vtype == "fixed_small":
            pass
        elif vtype == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif vtype == "fixed_large":
            variance = current_beta_t
        elif vtype == "fixed_large_log":
            variance = torch.log(current_beta_t)
        elif vtype == "learned":
            return predicted_variance
        elif vtype == "learned_range":
            assert predicted_variance is not None
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput:
        """Predict the sample from the previous timestep by reversing the SDE."""
        t = timestep
        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
        else:
            raise ValueError(f"prediction_type must be epsilon, sample, or v_prediction, got {self.prediction_type}")

        # 3. Clip or threshold "predicted x_0"
        if self.thresholding:
            pred_original_sample = threshold_sample(pred_original_sample, self.dynamic_thresholding_ratio, self.sample_max_value)
        elif self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-self.clip_sample_range, self.clip_sample_range)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = alpha_prod_t_prev**0.5 * current_beta_t / beta_prod_t
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = self._get_variance(t, predicted_variance=predicted_variance) ** 0.5 * variance_noise

            pred_prev_sample = pred_prev_sample + variance

        return SchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

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

    def previous_timestep(self, timestep: int) -> int | torch.Tensor:
        """Compute the previous timestep in the diffusion chain."""
        if self.custom_timesteps or self.num_inference_steps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            prev_t = timestep - 1
        return prev_t

    def __len__(self) -> int:
        return self.num_train_timesteps
