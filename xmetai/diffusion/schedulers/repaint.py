"""RePaint scheduler for DDPM inpainting inside a given mask.

Faithful port of diffusers.schedulers.scheduling_repaint.RePaintScheduler,
stripped of diffusers framework dependencies.

Reference: RePaint: Inpainting using Denoising Diffusion Probabilistic Models
(https://huggingface.co/papers/2201.09865)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from ..common import SchedulerOutput, betas_for_alpha_bar

__all__ = ["RePaintScheduler"]


class RePaintScheduler:
    """RePaint scheduler for DDPM-based inpainting.

    Implements the time-jump strategy (forward/backward in diffusion timesteps)
    to harmonize inpainted regions with known context.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"] = "linear",
        eta: float = 0.0,
        trained_betas: np.ndarray | list[float] | None = None,
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.eta = eta

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.final_alpha_cumprod = torch.tensor(1.0)

        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        device: str | torch.device | None = None,
    ) -> None:
        """Set timesteps with RePaint's forward-backward jump strategy.

        Args:
            num_inference_steps: Number of diffusion steps for inference.
            jump_length: Steps taken forward before going backward per jump ("j" in paper).
            jump_n_sample: Number of forward jumps per time sample.
            device: Device for timesteps tensor.
        """
        num_inference_steps = min(self.num_train_timesteps, num_inference_steps)
        self.num_inference_steps = num_inference_steps

        timesteps = []
        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)
            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        assert self.num_inference_steps is not None
        timesteps = np.array(timesteps) * (self.num_train_timesteps // self.num_inference_steps)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, t: int) -> torch.Tensor:
        assert self.num_inference_steps is not None
        prev_timestep = t - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        original_image: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput:
        """Predict the sample from the previous timestep with inpainting.

        Args:
            model_output: Direct output from the diffusion model (predicted noise).
            timestep: Current discrete timestep.
            sample: Current noisy sample.
            original_image: The original image to inpaint on.
            mask: Mask where 0.0 indicates the region to inpaint.
            generator: Random number generator.
        """
        assert self.num_inference_steps is not None
        t = timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # 3. Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Add noise (DDIM-style with eta)
        device = model_output.device
        noise = torch.randn(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

        variance = 0
        if t > 0 and self.eta > 0:
            variance = std_dev_t * noise

        # 5. compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * model_output

        # 6. compute x_{t-1} for unknown part
        prev_unknown_part = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction + variance

        # 7. compute x_{t-1} for known part (re-noised original)
        prev_known_part = alpha_prod_t_prev**0.5 * original_image + (1 - alpha_prod_t_prev) ** 0.5 * noise

        # 8. combine known and unknown via mask
        pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part

        return SchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def undo_step(self, sample: torch.Tensor, timestep: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Undo a diffusion step — re-noise the sample forward."""
        assert self.num_inference_steps is not None
        n = self.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample

    def __len__(self) -> int:
        return self.num_train_timesteps
