"""DPM-Solver Multistep scheduler (DDPM++ 2M).

Faithful port of diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler,
stripped of diffusers framework dependencies.

Supports: dpmsolver, dpmsolver++, sde-dpmsolver, sde-dpmsolver++ algorithms.
Supports: solver orders 1/2/3, midpoint/heun solver types.
Supports: Karras, exponential, LU, flow, and beta sigma schedules.

References:
- https://huggingface.co/papers/2206.00927 (DPM-Solver)
- https://huggingface.co/papers/2211.01095 (DPM-Solver++)
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch

from ..common import SchedulerOutput, betas_for_alpha_bar, rescale_zero_terminal_snr, threshold_sample

__all__ = ["DPMSolverMultistepScheduler"]


class DPMSolverMultistepScheduler:
    """Fast dedicated high-order solver for diffusion ODEs.

    The default configuration (algorithm_type="dpmsolver++", solver_order=2)
    corresponds to DDPM++ 2M.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: np.ndarray | list[float] | None = None,
        solver_order: int = 2,
        prediction_type: Literal["epsilon", "sample", "v_prediction", "flow_prediction"] = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: Literal["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"] = "dpmsolver++",
        solver_type: Literal["midpoint", "heun"] = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: bool = False,
        use_exponential_sigmas: bool = False,
        use_beta_sigmas: bool = False,
        use_lu_lambdas: bool = False,
        use_flow_sigmas: bool = False,
        flow_shift: float = 1.0,
        final_sigmas_type: Literal["zero", "sigma_min"] = "zero",
        lambda_min_clipped: float = -float("inf"),
        variance_type: Literal["learned", "learned_range"] | None = None,
        timestep_spacing: Literal["linspace", "leading", "trailing"] = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.euler_at_final = euler_at_final
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas
        self.use_lu_lambdas = use_lu_lambdas
        self.use_flow_sigmas = use_flow_sigmas
        self.flow_shift = flow_shift
        self.final_sigmas_type = final_sigmas_type
        self.lambda_min_clipped = lambda_min_clipped
        self.variance_type = variance_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset

        if sum([use_beta_sigmas, use_exponential_sigmas, use_karras_sigmas]) > 1:
            raise ValueError("Only one of use_beta_sigmas, use_exponential_sigmas, use_karras_sigmas can be used.")

        # Compute betas
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

        if rescale_betas_zero_snr:
            self.alphas_cumprod[-1] = 2**-24

        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.init_noise_sigma = 1.0

        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")
        if solver_type not in ["midpoint", "heun"]:
            raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")
        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            raise ValueError(f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}.")

        self.num_inference_steps: int | None = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.sigmas = self.sigmas.to("cpu")

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
    ) -> None:
        """Set the discrete timesteps for inference."""
        if num_inference_steps is None and timesteps is None:
            raise ValueError("Must pass exactly one of `num_inference_steps` or `timesteps`.")
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `timesteps`.")
        if timesteps is not None and self.use_karras_sigmas:
            raise ValueError("Cannot use `timesteps` with `use_karras_sigmas = True`")
        if timesteps is not None and self.use_lu_lambdas:
            raise ValueError("Cannot use `timesteps` with `use_lu_lambdas = True`")
        if timesteps is not None and self.use_exponential_sigmas:
            raise ValueError("Cannot set `timesteps` with `use_exponential_sigmas = True`.")
        if timesteps is not None and self.use_beta_sigmas:
            raise ValueError("Cannot set `timesteps` with `use_beta_sigmas = True`.")

        if timesteps is not None:
            ts_arr = np.array(timesteps).astype(np.int64)
        else:
            assert num_inference_steps is not None
            clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.lambda_min_clipped)
            last_timestep = ((self.num_train_timesteps - clipped_idx).numpy()).item()

            if self.timestep_spacing == "linspace":
                ts_arr = np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            elif self.timestep_spacing == "leading":
                step_ratio = last_timestep // (num_inference_steps + 1)
                ts_arr = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
                ts_arr += self.steps_offset
            elif self.timestep_spacing == "trailing":
                step_ratio = self.num_train_timesteps / num_inference_steps
                ts_arr = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
                ts_arr -= 1
            else:
                raise ValueError(f"{self.timestep_spacing} is not supported.")

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)

        if self.use_karras_sigmas:
            assert num_inference_steps is not None
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            ts_arr = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            if self.beta_schedule != "squaredcos_cap_v2":
                ts_arr = ts_arr.round()
        elif self.use_lu_lambdas:
            assert num_inference_steps is not None
            lambdas = np.flip(log_sigmas.copy())
            lambdas = self._convert_to_lu(in_lambdas=lambdas, num_inference_steps=num_inference_steps)
            sigmas = np.exp(lambdas)
            ts_arr = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
            if self.beta_schedule != "squaredcos_cap_v2":
                ts_arr = ts_arr.round()
        elif self.use_exponential_sigmas:
            assert num_inference_steps is not None
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            ts_arr = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.use_beta_sigmas:
            assert num_inference_steps is not None
            sigmas = np.flip(sigmas).copy()
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            ts_arr = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.use_flow_sigmas:
            assert num_inference_steps is not None
            alphas = np.linspace(1, 1 / self.num_train_timesteps, num_inference_steps + 1)
            flow_sigmas = 1.0 - alphas
            sigmas = np.flip(self.flow_shift * flow_sigmas / (1 + (self.flow_shift - 1) * flow_sigmas))[:-1].copy()
            ts_arr = (sigmas * self.num_train_timesteps).copy()
        else:
            sigmas = np.interp(ts_arr, np.arange(0, len(sigmas)), sigmas)

        if self.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {self.final_sigmas_type}")

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(ts_arr).to(device=device, dtype=torch.int64)
        self.num_inference_steps = len(ts_arr)
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _sigma_to_t(self, sigma: np.ndarray, log_sigmas: np.ndarray) -> np.ndarray:
        log_sigma = np.log(np.maximum(sigma, 1e-10))
        dists = log_sigma - log_sigmas[:, np.newaxis]
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.reshape(sigma.shape)

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_flow_sigmas:
            alpha_t = 1 - sigma
            sigma_t = sigma
        else:
            alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
            sigma_t = sigma * alpha_t
        return alpha_t, sigma_t

    def _convert_to_karras(self, in_sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        rho = 7.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_lu(self, in_lambdas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        lambda_min = in_lambdas[-1].item()
        lambda_max = in_lambdas[0].item()
        rho = 1.0
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = lambda_min ** (1 / rho)
        max_inv_rho = lambda_max ** (1 / rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

    def _convert_to_exponential(self, in_sigmas: np.ndarray, num_inference_steps: int) -> np.ndarray:
        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))

    def _convert_to_beta(self, in_sigmas: np.ndarray, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6) -> np.ndarray:
        try:
            from scipy.stats import beta as beta_dist
        except ImportError:
            raise ImportError("scipy is required for use_beta_sigmas")

        sigma_min = in_sigmas[-1].item()
        sigma_max = in_sigmas[0].item()
        ppfs = [beta_dist.ppf(t, alpha, beta) for t in 1 - np.linspace(0, 1, num_inference_steps)]
        return np.array([sigma_min + (ppf * (sigma_max - sigma_min)) for ppf in ppfs])

    def convert_model_output(self, model_output: torch.Tensor, *, sample: torch.Tensor) -> torch.Tensor:
        """Convert model output to the format needed by the DPM-Solver algorithm."""
        # DPM-Solver++ (data prediction)
        if self.algorithm_type in ["dpmsolver++", "sde-dpmsolver++"]:
            if self.prediction_type == "epsilon":
                if self.variance_type in ["learned", "learned_range"]:
                    model_output = model_output[:, :3]
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.prediction_type == "sample":
                x0_pred = model_output
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = alpha_t * sample - sigma_t * model_output
            elif self.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

            if self.thresholding:
                x0_pred = threshold_sample(x0_pred, self.dynamic_thresholding_ratio, self.sample_max_value)
            return x0_pred

        # DPM-Solver (noise prediction)
        elif self.algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            if self.prediction_type == "epsilon":
                if self.variance_type in ["learned", "learned_range"]:
                    epsilon = model_output[:, :3]
                else:
                    epsilon = model_output
            elif self.prediction_type == "sample":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = (sample - alpha_t * model_output) / sigma_t
            elif self.prediction_type == "v_prediction":
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                epsilon = alpha_t * model_output + sigma_t * sample
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

            if self.thresholding:
                sigma = self.sigmas[self.step_index]
                alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
                x0_pred = (sample - sigma_t * epsilon) / alpha_t
                x0_pred = threshold_sample(x0_pred, self.dynamic_thresholding_ratio, self.sample_max_value)
                epsilon = (sample - alpha_t * x0_pred) / sigma_t
            return epsilon

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *,
        sample: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step for first-order DPM-Solver (equivalent to DDIM)."""
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        h = lambda_t - lambda_s
        if self.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.algorithm_type == "dpmsolver":
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            x_t = (alpha_t / alpha_s) * sample - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: list[torch.Tensor],
        *,
        sample: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step for second-order multistep DPM-Solver."""
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)

        if self.algorithm_type == "dpmsolver++":
            if self.solver_type == "midpoint":
                x_t = (sigma_t / sigma_s0) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * D0 - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
            elif self.solver_type == "heun":
                x_t = (sigma_t / sigma_s0) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * D0 + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
        elif self.algorithm_type == "dpmsolver":
            if self.solver_type == "midpoint":
                x_t = (alpha_t / alpha_s0) * sample - (sigma_t * (torch.exp(h) - 1.0)) * D0 - 0.5 * (sigma_t * (torch.exp(h) - 1.0)) * D1
            elif self.solver_type == "heun":
                x_t = (alpha_t / alpha_s0) * sample - (sigma_t * (torch.exp(h) - 1.0)) * D0 - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                    + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
        elif self.algorithm_type == "sde-dpmsolver":
            assert noise is not None
            if self.solver_type == "midpoint":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - (sigma_t * (torch.exp(h) - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            elif self.solver_type == "heun":
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * D0
                    - 2.0 * (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: list[torch.Tensor],
        *,
        sample: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One step for third-order multistep DPM-Solver."""
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)

        if self.algorithm_type == "dpmsolver++":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2
            )
        elif self.algorithm_type == "dpmsolver":
            x_t = (
                (alpha_t / alpha_s0) * sample
                - (sigma_t * (torch.exp(h) - 1.0)) * D0
                - (sigma_t * ((torch.exp(h) - 1.0) / h - 1.0)) * D1
                - (sigma_t * ((torch.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2
            )
        elif self.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1.0 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h) - 2.0 * h) / (2.0 * h) ** 2 - 0.5)) * D2
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        return x_t

    def index_for_timestep(
        self,
        timestep: int | torch.Tensor,
        schedule_timesteps: torch.Tensor | None = None,
    ) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        index_candidates = (schedule_timesteps == timestep).nonzero()
        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()
        return step_index

    def _init_step_index(self, timestep: int | torch.Tensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        variance_noise: torch.Tensor | None = None,
    ) -> SchedulerOutput:
        """Predict the sample from the previous timestep using multistep DPM-Solver."""
        if self.num_inference_steps is None:
            raise ValueError("Run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.euler_at_final or (self.lower_order_final and len(self.timesteps) < 15) or self.final_sigmas_type == "zero"
        )
        lower_order_second = (self.step_index == len(self.timesteps) - 2) and self.lower_order_final and len(self.timesteps) < 15

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        sample = sample.to(torch.float32)
        if self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
            noise = torch.randn(model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32)
        elif self.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
            assert variance_noise is not None
            noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
        else:
            noise = None

        if self.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample, noise=noise)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        prev_sample = prev_sample.to(model_output.dtype)
        assert self._step_index is not None
        self._step_index += 1

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor | torch.LongTensor,
    ) -> torch.Tensor:
        """Add noise to original samples according to the noise schedule."""
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self) -> int:
        return self.num_train_timesteps
