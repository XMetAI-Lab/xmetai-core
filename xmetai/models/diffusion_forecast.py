"""
Detectron2 Diffusion Meta-Architecture + Custom Trainer
Author: douzsh
Date: 2025-09-01

Design goals:
- Minimal external deps: pure PyTorch + Detectron2.
- Inputs follow Detectron2's convention: a batch is a list[dict].
  Required keys per sample:
    - "input": Tensor of shape (T_in, C, H, W)  (conditioning frames)
    - "target": Tensor of shape (T_out, C, H, W) (we train on the first future frame by default)
  Optional: "times" (any metadata you need; not used here)
- Conditioning: uses the *last* conditioning frame (T_in[-1]) concatenated to the noised target as model input.
- Training objective: DDPM epsilon-prediction (can switch to x0 prediction via cfg).
- Inference: DDIM sampler for speed; autoregressive over out_frames if >1.
"""

from typing import Any, List, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from xmetai.layers import DarknetConvUNet2D
from xmetai.registries import MODELS
from xmetai.utils import ensure_torchvision_nms_schema

from .base_predictor import BasePredictor

__all__ = ["DiffusionForecast"]


def pad_to_even(x: Tensor, multiple: int = 4) -> Tensor:
    H, W = x.shape[-2:]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h))


@torch.jit.script
def _extract_timestep_values(values: Tensor, timesteps: Tensor, sample: Tensor) -> Tensor:
    coeff = values.index_select(0, timesteps.to(torch.long)).to(sample)
    shape: List[int] = [sample.shape[0]]
    for _ in range(sample.dim() - 1):
        shape.append(1)
    return coeff.reshape(shape)


@torch.jit.script
def _jit_add_noise(original_samples: Tensor, noise: Tensor, timesteps: Tensor, sqrt_alphas_cumprod: Tensor, sqrt_one_minus_alphas_cumprod: Tensor) -> Tensor:
    sqrt_alpha = _extract_timestep_values(sqrt_alphas_cumprod, timesteps, original_samples)
    sqrt_one_minus_alpha = _extract_timestep_values(sqrt_one_minus_alphas_cumprod, timesteps, original_samples)
    return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise


@torch.jit.script
def _jit_ddim_step(
    model_output: Tensor,
    timestep: Tensor,
    prev_timestep: Tensor,
    sample: Tensor,
    noise: Tensor,
    alphas_cumprod: Tensor,
    final_alpha_cumprod: Tensor,
    eta: float,
    predict_type: str,
    clip_sample: bool,
) -> Tensor:
    alpha_prod_t = alphas_cumprod[timestep.to(torch.long)].to(sample)
    if prev_timestep.item() >= 0:
        alpha_prod_t_prev = alphas_cumprod[prev_timestep.to(torch.long)].to(sample)
    else:
        alpha_prod_t_prev = final_alpha_cumprod.to(sample)

    beta_prod_t = 1.0 - alpha_prod_t
    if predict_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
    elif predict_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.clamp_min(1e-12).sqrt()
    else:
        pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        pred_epsilon = alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample

    if clip_sample:
        pred_original_sample = pred_original_sample.clamp(-1.0, 1.0)

    variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
    std_dev_t = eta * variance.clamp_min(0.0).sqrt()
    pred_sample_direction = (1.0 - alpha_prod_t_prev - std_dev_t * std_dev_t).clamp_min(0.0).sqrt() * pred_epsilon
    prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
    if eta > 0.0:
        prev_sample = prev_sample + std_dev_t * noise
    return prev_sample


def _weighted_charbonnier_loss(output: Tensor, target: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    loss = torch.sqrt((output - target).float().pow(2) + eps)
    if weight is None:
        return loss.mean()

    weight = weight.to(loss)
    loss = loss * weight
    denom = weight.expand_as(loss).sum().clamp_min(torch.finfo(loss.dtype).eps)
    return loss.sum() / denom


@MODELS.register()
class DiffusionForecast(BasePredictor):
    """
    Conditional diffusion forecasting meta-arch.
    Training: predict noise eps from noised target x_t, conditioned on last input frame.
    Inference: DDIM sampling; if OUT_FRAMES>1, autoregressively generate next frames.
    """

    def __init__(
        self,
        base_channels: int,
        ch_mults: Tuple[int, ...],
        attn_res: Tuple[int, ...],
        num_heads: int,
        dropout: float = 0.0,
        layers_per_block: int = 3,
        backbone: str = "diffusers_unet",
        use_attention: bool = True,
        num_steps: int = 1000,
        predict_type: str = "epsilon",
        sample_steps: int = 100,
        ddim_eta: float = 0.0,  # for DDIM sampling
        clip_x0: bool = True,
        scheduler_type: str = "ddim",  # "ddpm" or "ddim"
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        image_size: Tuple[int, int] = (120, 240),
        predict_residual: bool = True,
        zero_init_output: bool = False,
        sample_init: str = "noise",
        sample_timestep_max: int | None = None,
        train_timestep_max: int | None = None,
        const_chans: int = 0,
        use_timestamp: bool = False,
        train_chans: List[int] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.logger.name = __name__
        if predict_type not in {"epsilon", "sample", "v_prediction"}:
            raise ValueError(f"Unknown predict_type {predict_type}")
        if scheduler_type != "ddim":
            raise ValueError("DiffusionForecast JIT scheduler currently supports scheduler_type='ddim' only.")
        if sample_init not in {"noise", "zeros"}:
            raise ValueError(f"Unknown sample_init {sample_init}")
        if backbone not in {"diffusers_unet", "darknet"}:
            raise ValueError(f"Unknown backbone {backbone}")

        self.predict_type = predict_type
        self.scheduler_type = scheduler_type
        self.sample_init = sample_init
        self.num_steps = num_steps
        self.sample_steps = sample_steps
        self.clip_x0 = clip_x0
        self.ddim_eta = ddim_eta
        self.predict_residual = predict_residual
        self.sample_timestep_max = num_steps - 1 if sample_timestep_max is None else sample_timestep_max
        self.train_timestep_max = num_steps - 1 if train_timestep_max is None else train_timestep_max
        if not 0 <= self.sample_timestep_max < num_steps:
            raise ValueError(f"sample_timestep_max must be in [0, {num_steps - 1}], got {self.sample_timestep_max}")
        if not 0 <= self.train_timestep_max < num_steps:
            raise ValueError(f"train_timestep_max must be in [0, {num_steps - 1}], got {self.train_timestep_max}")
        self.const_chans = const_chans
        self.use_timestamp = use_timestamp
        self.train_chans = list(train_chans) if train_chans is not None else []
        self.model_out_chans = len(self.train_chans) if self.train_chans else self.out_chans
        self.pad_multiple = 2 ** max(1, len(ch_mults) - 1)

        # Cond + target are concatenated along channel dim
        cond_chans = self.in_chans * self.in_frames + const_chans + (3 if use_timestamp else 0)
        model_in_ch = cond_chans + self.model_out_chans  # input conditioning + noised target

        t_emb_dim = base_channels * 4
        if backbone == "darknet":
            self.unet = DarknetConvUNet2D(
                in_channels=model_in_ch,
                out_channels=self.model_out_chans,
                base_channels=base_channels,
                ch_mults=ch_mults,
                layers_per_block=layers_per_block,
                time_dim=t_emb_dim,
                dropout=dropout,
            )
        else:
            ensure_torchvision_nms_schema()
            from diffusers import UNet2DModel

            block_count = len(ch_mults)
            if use_attention:
                down_block_types = tuple(["DownBlock2D"] * max(0, block_count - 1) + ["AttnDownBlock2D"])
                up_block_types = tuple(["AttnUpBlock2D"] + ["UpBlock2D"] * max(0, block_count - 1))
            else:
                down_block_types = tuple(["DownBlock2D"] * block_count)
                up_block_types = tuple(["UpBlock2D"] * block_count)
            self.unet = UNet2DModel(
                sample_size=image_size,
                in_channels=model_in_ch,
                out_channels=self.model_out_chans,
                layers_per_block=layers_per_block,
                block_out_channels=np.array(ch_mults) * base_channels,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
                time_embedding_dim=t_emb_dim,
                attention_head_dim=num_heads,
                dropout=dropout,
            )
        if zero_init_output:
            torch.nn.init.zeros_(self.unet.conv_out.weight)
            torch.nn.init.zeros_(self.unet.conv_out.bias)

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.register_buffer("final_alpha_cumprod", torch.tensor(1.0, dtype=torch.float32), persistent=False)
        self.set_sample_steps(sample_steps)

    def set_sample_steps(self, sample_steps: int) -> None:
        if sample_steps <= 0 or sample_steps > self.num_steps:
            raise ValueError(f"sample_steps must be in [1, {self.num_steps}], got {sample_steps}")
        self.sample_steps = sample_steps
        if sample_steps == 1:
            timesteps = torch.tensor([self.sample_timestep_max], dtype=torch.long)
            prev_timesteps = torch.tensor([-1], dtype=torch.long)
        else:
            timesteps = torch.linspace(float(self.sample_timestep_max), 0.0, sample_steps).round().to(torch.long)
            prev_timesteps = torch.cat([timesteps[1:], timesteps.new_tensor([-1])])
        self.register_buffer("sample_timesteps", timesteps, persistent=False)
        self.register_buffer("sample_prev_timesteps", prev_timesteps, persistent=False)

    # -------------- I/O helpers --------------
    def preprocess_batch(self, batched_inputs: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert list[dict] to tensors.
        Returns:
            cond: (B, C, H, W) last conditioning frame
            target: (B, C, H, W) first target frame (train step uses one frame)
        """
        conds = batched_inputs["input"]  # list of (B, in_frames, C, H, W)
        targets = batched_inputs["targets"]  # list of (B, out_frames, C, H, W)
        if self.train_chans:
            targets = targets[:, :, self.train_chans]
            residual_base = conds[:, -1:, self.train_chans]
        else:
            residual_base = conds[:, -1:]
        if self.predict_residual:
            targets = targets - residual_base  # (B, out_frames, C, H, W)
        conds = einops.rearrange(conds, "B T C H W -> B (T C) H W")
        targets = einops.rearrange(targets, "B T C H W -> B (T C) H W")
        return conds, targets

    def condition_extra(self, data: dict[str, Any], input: Tensor, step: int = 0) -> Tensor | None:
        extras = []
        if self.const_chans > 0:
            extras.append(self.condition_const(data, input))
        if self.use_timestamp:
            extras.append(self.condition_timestamp(data, input, step))
        if not extras:
            return None
        return torch.cat(extras, dim=1)

    def condition_const(self, data: dict[str, Any], input: Tensor) -> Tensor:
        const = data["const"].to(input)
        if const.ndim == 3:
            const = const.unsqueeze(0).expand(input.shape[0], -1, -1, -1)
        return const[:, : self.const_chans]

    def condition_timestamp(self, data: dict[str, Any], input: Tensor, step: int) -> Tensor:
        _, _, _, height, width = input.shape
        timestamp = self.get_timestamp(input, data["times"], step)
        return torch.cat([timestamp[k].to(input).reshape(input.shape[0], 1, 1, 1).expand(-1, 1, height, width) for k in ("step", "hour", "doy")], dim=1)

    def build_model_input(self, cond: Tensor, target_input: Tensor, extra: Tensor | None = None) -> Tensor:
        if extra is None:
            return torch.cat([cond, target_input], dim=1)
        return torch.cat([cond, extra, target_input], dim=1)

    def reconstruction_weight(self, target: Tensor) -> Tensor | None:
        if "channel_weights" in self._buffers:
            channel_weight = self.channel_weights[self.train_chans] if self.train_chans else self.channel_weights[: self.out_chans]
            channel_weight = channel_weight.repeat(self.out_frames, 1, 1)
            return (channel_weight * self.weight).reshape(1, self.out_frames * self.model_out_chans, *self.weight.shape[-2:]).to(target)
        if "weight" in self._buffers:
            return self.weight.reshape(1, 1, *self.weight.shape[-2:]).to(target)
        return None

    # -------------- forward --------------
    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            return self.test(data)  # type: ignore[return-value]

        data = self.process_data(data)
        cond, target = self.preprocess_batch(data)
        extra = self.condition_extra(data, data["input"], step=self.start_frames)
        batch_size = target.shape[0]

        timesteps = torch.randint(
            0,
            self.train_timestep_max + 1,
            (batch_size,),
            device=target.device,
        )
        noise = torch.randn_like(target)
        noisy_target = _jit_add_noise(target, noise, timesteps, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod)

        model_in = self.build_model_input(cond, noisy_target, extra)
        _, _, H, W = model_in.shape
        model_in = pad_to_even(model_in, self.pad_multiple)
        self.logger.debug(f"model input shape: {model_in.shape}, min: {model_in.min().item()}, max: {model_in.max().item()}")
        pred = self.unet(model_in, timesteps, return_dict=False)[0][:, :, :H, :W]  # type: ignore
        self.logger.debug(f"model output shape: {pred.shape}, min: {pred.min().item()}, max: {pred.max().item()}")

        if self.predict_type == "epsilon":
            loss = _weighted_charbonnier_loss(pred, noise, weight=self.reconstruction_weight(target))
        elif self.predict_type == "sample":
            loss = _weighted_charbonnier_loss(pred, target, weight=self.reconstruction_weight(target))
        elif self.predict_type == "v_prediction":
            velocity = (
                _extract_timestep_values(self.sqrt_alphas_cumprod, timesteps, target) * noise
                - _extract_timestep_values(self.sqrt_one_minus_alphas_cumprod, timesteps, target) * target
            )
            loss = _weighted_charbonnier_loss(pred, velocity, weight=self.reconstruction_weight(target))
        else:
            raise ValueError(f"Unknown predict_type {self.predict_type}")

        return {"loss_rec": loss}

    # -------------- inference --------------
    @torch.no_grad()
    def test_cascade(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        input = data["input"]
        const_extra = self.condition_const(data, input) if self.const_chans > 0 else None
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames
        assert sum(self.step_range) <= total_step

        chw = list(input.shape[-3:])
        sample_chw = [self.model_out_chans, *chw[-2:]]
        outputs = input.new_zeros(input.shape[0], len(steps), *chw)  # batch_size, saved_steps, c,h,w
        self.logger.debug(f"output shape is {outputs.shape}")

        for num_step in self.step_range:
            curr_step = 0
            for step in range(num_step):
                self.reset_input(input)
                last_frame = einops.rearrange(input, "B T C H W -> B (T C) H W")
                if self.use_timestamp:
                    timestamp_extra = self.condition_timestamp(data, input, step=self.start_frames + step)
                    extra = timestamp_extra if const_extra is None else torch.cat([const_extra, timestamp_extra], dim=1)
                else:
                    extra = const_extra
                if self.sample_init == "noise":
                    x_t = torch.randn(input.shape[0], *sample_chw, device=input.device)
                else:
                    x_t = torch.zeros(input.shape[0], *sample_chw, device=input.device, dtype=input.dtype)
                for t, prev_t in zip(self.sample_timesteps, self.sample_prev_timesteps):
                    model_in = self.build_model_input(last_frame, x_t, extra)
                    _, _, H, W = model_in.shape
                    model_in = pad_to_even(model_in, self.pad_multiple)
                    noise_pred = self.unet(model_in, t, return_dict=False)[0][:, :, :H, :W]  # type: ignore

                    ddim_noise = torch.randn_like(x_t) if self.ddim_eta > 0.0 else torch.zeros_like(x_t)
                    x_t = _jit_ddim_step(
                        noise_pred,
                        t,
                        prev_t,
                        x_t,
                        ddim_noise,
                        self.alphas_cumprod,
                        self.final_alpha_cumprod,
                        self.ddim_eta,
                        self.predict_type,
                        self.clip_x0,
                    )

                output = self.merge_prediction(input, x_t.detach())
                input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                # save for output
                if step + 1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                    curr_step += 1

        return outputs

    def merge_prediction(self, input: Tensor, pred: Tensor) -> Tensor:
        if not self.train_chans:
            output = pred.unsqueeze(1)
            if self.predict_residual:
                output = output + input[:, -1:]
            return output

        output = input[:, -1:].clone()
        pred = pred.unsqueeze(1).to(output)
        if self.predict_residual:
            output[:, :, self.train_chans] = output[:, :, self.train_chans] + pred
        else:
            output[:, :, self.train_chans] = pred
        return output
