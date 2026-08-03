"""Shared conditional generative forecasts with interchangeable weather backbones."""

from typing import Any, List, Mapping, cast

import einops
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from xmetai.layers import DarknetConvUNet2D
from xmetai.registries import MODELS

from .base_predictor import BasePredictor
from .utransformer import UTransformer

__all__ = ["ConditionalGenerativeForecast", "DiffusionForecast", "FlowMatchingForecast", "JiTForecast", "ODEGenerativeForecast"]


def _weighted_mean(loss: Tensor, weight: Tensor | None) -> Tensor:
    if weight is None:
        return loss.mean()
    weight = weight.to(loss)
    return (loss * weight).sum() / weight.expand_as(loss).sum().clamp_min(torch.finfo(loss.dtype).eps)


def weighted_mse_loss(output: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """Return a spatially/channel weighted float32 mean squared error."""
    return _weighted_mean((output - target).float().square(), weight)


def weighted_charbonnier_loss(output: Tensor, target: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """Return a spatially/channel weighted Charbonnier loss."""
    return _weighted_mean(torch.sqrt((output - target).float().square() + eps), weight)


def pad_to_multiple(x: Tensor, multiple: int | tuple[int, int]) -> Tensor:
    """Pad the bottom and right edges to a rectangular backbone multiple."""
    multiple = (multiple, multiple) if isinstance(multiple, int) else multiple
    height, width = x.shape[-2:]
    pad_h = (multiple[0] - height % multiple[0]) % multiple[0]
    pad_w = (multiple[1] - width % multiple[1]) % multiple[1]
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))


class TimeConditionedUTransformer(UTransformer):
    """Expose UTransformer through the same time-conditioned 2D interface as DarknetUNet."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        kwargs.setdefault("in_frames", 1)
        kwargs.setdefault("out_frames", 1)
        kwargs.setdefault("const_chans", 0)
        kwargs.setdefault("head_type", "2fc")
        kwargs.setdefault("embed_type", "step")
        if kwargs["in_frames"] != 1 or kwargs["out_frames"] != 1 or kwargs["const_chans"] != 0:
            raise ValueError("The generative UTransformer backbone uses in_frames=1, out_frames=1, and const_chans=0 after channel flattening")
        if kwargs["head_type"] not in {"1fc", "2fc"}:
            raise ValueError("The generative UTransformer backbone requires head_type='1fc' or '2fc'")
        if kwargs["embed_type"] == "all":
            raise ValueError("The generative UTransformer backbone requires a scalar-time embed_type such as 'step'")
        super().__init__(in_chans=in_channels, out_chans=out_channels, **kwargs)

    def forward(self, x: Tensor, timesteps: Tensor | None = None, return_dict: bool = False) -> tuple[Tensor] | dict[str, Tensor]:
        if timesteps is None:
            raise ValueError("TimeConditionedUTransformer requires timesteps")
        batch_size = x.shape[0]
        source = x
        resized = self.resize_input(x.unsqueeze(1)).unsqueeze(1)
        h = self.patch_embed(resized, None)
        time = timesteps.reshape(-1).to(x)
        if time.shape[0] == 1 and batch_size > 1:
            time = time.expand(batch_size)
        if time.shape[0] != batch_size:
            raise ValueError(f"timestep batch size {time.shape[0]} does not match input batch size {batch_size}")
        embed = self.get_embed(time)

        for index, block in enumerate(self.layers):
            h = self._modules[f"drop{index}"](h, self.drop_mcmc)
            if "rotary" in self.attn_type:
                h = block(h, time_embedding=embed, freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
            else:
                h = block(h, time_embedding=embed)

        if self.head is None:
            raise RuntimeError("TimeConditionedUTransformer requires an output head")
        output = self.resize_output(self.head(h), source)
        return {"sample": output} if return_dict else (output,)


def build_generative_backbone(backbone: str, in_channels: int, out_channels: int, backbone_kwargs: Mapping[str, Any]) -> tuple[nn.Module, tuple[int, int]]:
    """Build one of the supported backbones and return its required input multiple."""
    name = backbone.lower().replace("-", "_")
    options = dict(backbone_kwargs)
    if name in {"darknet", "darknet_unet", "darknetunet"}:
        name = "darknet_unet"
        ch_mults = tuple(options.pop("ch_mults"))
        if len(ch_mults) < 2:
            raise ValueError("DarknetUNet expects at least two channel levels")
        base_channels = int(options.pop("base_channels"))
        options.setdefault("layers_per_block", 3)
        options.setdefault("time_dim", base_channels * 4)
        options.setdefault("dropout", 0.0)
        network = DarknetConvUNet2D(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, ch_mults=ch_mults, **options)
        multiple = 2 ** (len(ch_mults) - 1)
        return network, (multiple, multiple)
    if name in {"utransformer", "u_transformer"}:
        return TimeConditionedUTransformer(in_channels=in_channels, out_channels=out_channels, **options), (1, 1)
    raise ValueError(f"Unsupported generative backbone {backbone!r}; expected 'darknet_unet' or 'utransformer'")


class ConditionalGenerativeForecast(BasePredictor):
    """Shared data, conditioning, backbone, and autoregressive rollout contract."""

    backbone_module = "backbone"

    def __init__(
        self,
        backbone: str,
        backbone_kwargs: Mapping[str, Any],
        sample_steps: int = 50,
        sample_init: str = "noise",
        noise_scale: float = 1.0,
        predict_residual: bool = True,
        const_chans: int = 0,
        use_timestamp: bool = False,
        train_chans: list[int] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}")
        if sample_init not in {"noise", "zeros"}:
            raise ValueError(f"sample_init must be 'noise' or 'zeros', got {sample_init}")
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale}")
        if self.members <= 0:
            raise ValueError(f"members must be positive, got {self.members}")

        self.sample_steps = sample_steps
        self.sample_init = sample_init
        self.noise_scale = noise_scale
        self.predict_residual = predict_residual
        self.const_chans = const_chans
        self.use_timestamp = use_timestamp
        self.train_chans = list(train_chans) if train_chans is not None else []
        self.model_out_chans = len(self.train_chans) if self.train_chans else self.out_chans
        self.condition_chans = self.in_chans * self.in_frames + self.const_chans + (3 if self.use_timestamp else 0)
        self.model_input_chans = self.condition_chans + self.model_out_chans
        network, self.backbone_multiple = build_generative_backbone(backbone, self.model_input_chans, self.model_out_chans, backbone_kwargs)
        self.add_module(self.backbone_module, network)
        self.backbone_name = backbone.lower().replace("-", "_")

    @property
    def generative_backbone(self) -> nn.Module:
        return self._modules[self.backbone_module]

    def predict_backbone(self, cond: Tensor, state: Tensor, time: Tensor, extra: Tensor | None) -> Tensor:
        """Run the selected backbone with identical conditioning and spatial handling."""
        model_input = torch.cat([cond, state], dim=1) if extra is None else torch.cat([cond, extra, state], dim=1)
        height, width = model_input.shape[-2:]
        model_input = pad_to_multiple(model_input, self.backbone_multiple)
        prediction = self.generative_backbone(model_input, time.reshape(-1).to(model_input), return_dict=False)[0]
        return cast(Tensor, prediction[:, :, :height, :width])

    def prepare_condition(self, input: Tensor) -> Tensor:
        return einops.rearrange(input, "B T C H W -> B (T C) H W")

    def prepare_target(self, input: Tensor, targets: Tensor) -> Tensor:
        if self.train_chans:
            targets = targets[:, :, self.train_chans]
            residual_base = input[:, -1:, self.train_chans]
        else:
            residual_base = input[:, -1:]
        if self.predict_residual:
            targets = targets - residual_base
        return einops.rearrange(targets, "B T C H W -> B (T C) H W")

    def preprocess_batch(self, batched_inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        return self.prepare_condition(batched_inputs["input"]), self.prepare_target(batched_inputs["input"], batched_inputs["targets"])

    def condition_extra(self, data: dict[str, Any], input: Tensor, step: int = 0) -> Tensor | None:
        extras = []
        if self.const_chans > 0:
            extras.append(self.condition_const(data, input))
        if self.use_timestamp:
            extras.append(self.condition_timestamp(data, input, step))
        return torch.cat(extras, dim=1) if extras else None

    def condition_const(self, data: dict[str, Any], input: Tensor) -> Tensor:
        const = data["const"].to(input)
        if const.ndim == 3:
            const = const.unsqueeze(0).expand(input.shape[0], -1, -1, -1)
        return const[:, : self.const_chans]

    def condition_timestamp(self, data: dict[str, Any], input: Tensor, step: int) -> Tensor:
        height, width = input.shape[-2:]
        timestamp = self.get_timestamp(input, data["times"], step)
        return torch.cat([timestamp[key].to(input).reshape(input.shape[0], 1, 1, 1).expand(-1, 1, height, width) for key in ("step", "hour", "doy")], dim=1)

    def reconstruction_weight(self, target: Tensor) -> Tensor | None:
        if "channel_weights" in self._buffers:
            channel_weight = self.channel_weights[self.train_chans] if self.train_chans else self.channel_weights[: self.out_chans]
            channel_weight = channel_weight.repeat(self.out_frames, 1, 1)
            return (channel_weight * self.weight).reshape(1, self.out_frames * self.model_out_chans, *self.weight.shape[-2:]).to(target)
        if "weight" in self._buffers:
            return self.weight.reshape(1, 1, *self.weight.shape[-2:]).to(target)
        return None

    def merge_prediction(self, input: Tensor, pred: Tensor) -> Tensor:
        if not self.train_chans:
            output = pred.unsqueeze(1)
            return output + input[:, -1:] if self.predict_residual else output

        output = input[:, -1:].clone()
        pred = pred.unsqueeze(1).to(output)
        if self.predict_residual:
            output[:, :, self.train_chans] = output[:, :, self.train_chans] + pred
        else:
            output[:, :, self.train_chans] = pred
        return output

    def initialize_sample(self, reference: Tensor, sample_shape: tuple[int, ...]) -> Tensor:
        if self.sample_init == "noise" or self.members > 1:
            return torch.randn(sample_shape, device=reference.device, dtype=reference.dtype) * self.noise_scale
        return torch.zeros(sample_shape, device=reference.device, dtype=reference.dtype)

    def sample_residual(self, cond: Tensor, extra: Tensor | None, sample_shape: tuple[int, ...]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def test_cascade(self, data: dict[str, Tensor]) -> Tensor:
        original_input = data["input"]
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames
        if sum(self.step_range) > total_step:
            raise ValueError(f"sum(step_range)={sum(self.step_range)} exceeds max(test_frames)={total_step}")

        channels, height, width = original_input.shape[-3:]
        sample_shape = (original_input.shape[0], self.model_out_chans, height, width)
        outputs = torch.zeros(original_input.shape[0], len(steps), channels, height, width, device=original_input.device, dtype=torch.float32)

        self.logger.info(f"Start testing with {self.members} ensemble members and step_range {self.step_range} with time:{data.get('times', None)}.")
        for member in range(self.members):
            self.logger.debug(f"Inference with member {member + 1}/{self.members}.")
            input = original_input.clone()
            for num_step in self.step_range:
                curr_step = 0
                for step in range(num_step):
                    input = self.reset_input(input)
                    cond = self.prepare_condition(input)
                    pred = self.sample_residual(cond, self.condition_extra(data, input, step=self.start_frames + step), sample_shape)
                    output = self.merge_prediction(input, pred.detach())
                    input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                    if step + 1 in steps:
                        outputs[:, curr_step : curr_step + 1, : self.out_chans].add_(output.float())
                        curr_step += 1
        return (outputs / self.members).to(original_input)


class ODEGenerativeForecast(ConditionalGenerativeForecast):
    """Shared fixed-step Euler/Heun sampler for velocity-based objectives."""

    def __init__(self, sampling_method: str = "heun", **kwargs: Any):
        super().__init__(**kwargs)
        if sampling_method not in {"euler", "heun"}:
            raise ValueError(f"sampling_method must be 'euler' or 'heun', got {sampling_method}")
        self.sampling_method = sampling_method

    def predict_velocity(self, cond: Tensor, state: Tensor, time: Tensor, extra: Tensor | None) -> Tensor:
        raise NotImplementedError

    def _integrate_residual(self, cond: Tensor, extra: Tensor | None, state: Tensor, sample_steps: int) -> Tensor:
        if sample_steps <= 0:
            raise ValueError(f"sample_steps must be positive, got {sample_steps}")
        times = torch.linspace(0.0, 1.0, sample_steps + 1, device=cond.device, dtype=torch.float32)
        for index in range(sample_steps):
            time = times[index].expand(state.shape[0])
            next_time = times[index + 1].expand(state.shape[0])
            dt = next_time[0] - time[0]
            velocity = self.predict_velocity(cond, state, time, extra)
            euler_state = state + dt * velocity
            if self.sampling_method == "heun" and index < sample_steps - 1:
                state = state + 0.5 * dt * (velocity + self.predict_velocity(cond, euler_state, next_time, extra))
            else:
                state = euler_state
        return state

    def sample_residual(self, cond: Tensor, extra: Tensor | None, sample_shape: tuple[int, ...]) -> Tensor:
        return self._integrate_residual(cond, extra, self.initialize_sample(cond, sample_shape), self.sample_steps)


@torch.jit.script
def _extract_timestep_values(values: Tensor, timesteps: Tensor, sample: Tensor) -> Tensor:
    coeff = values.index_select(0, timesteps.to(torch.long)).to(sample)
    shape: List[int] = [sample.shape[0]]
    for _ in range(sample.dim() - 1):
        shape.append(1)
    return coeff.reshape(shape)


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
    alpha_prod_t_prev = alphas_cumprod[prev_timestep.to(torch.long)].to(sample) if prev_timestep.item() >= 0 else final_alpha_cumprod.to(sample)
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
    direction = (1.0 - alpha_prod_t_prev - std_dev_t.square()).clamp_min(0.0).sqrt() * pred_epsilon
    prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + direction
    return prev_sample + std_dev_t * noise if eta > 0.0 else prev_sample


@MODELS.register()
class DiffusionForecast(ConditionalGenerativeForecast):
    """Predict a diffusion target during training and generate residuals with DDIM."""

    backbone_module = "unet"

    def __init__(
        self,
        num_steps: int = 1000,
        predict_type: str = "epsilon",
        sample_steps: int = 100,
        ddim_eta: float = 0.0,
        clip_x0: bool = True,
        scheduler_type: str = "ddim",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        sample_timestep_max: int | None = None,
        train_timestep_max: int | None = None,
        zero_init_output: bool = False,
        **kwargs: Any,
    ):
        super().__init__(sample_steps=sample_steps, **kwargs)
        self.logger.name = __name__
        if predict_type not in {"epsilon", "sample", "v_prediction"}:
            raise ValueError(f"Unknown predict_type {predict_type}")
        if scheduler_type != "ddim":
            raise ValueError("DiffusionForecast supports scheduler_type='ddim' only")
        if sample_steps > num_steps:
            raise ValueError(f"sample_steps must be in [1, {num_steps}], got {sample_steps}")

        self.predict_type = predict_type
        self.scheduler_type = scheduler_type
        self.num_steps = num_steps
        self.clip_x0 = clip_x0
        self.ddim_eta = ddim_eta
        self.sample_timestep_max = num_steps - 1 if sample_timestep_max is None else sample_timestep_max
        self.train_timestep_max = num_steps - 1 if train_timestep_max is None else train_timestep_max
        if not 0 <= self.sample_timestep_max < num_steps:
            raise ValueError(f"sample_timestep_max must be in [0, {num_steps - 1}], got {self.sample_timestep_max}")
        if not 0 <= self.train_timestep_max < num_steps:
            raise ValueError(f"train_timestep_max must be in [0, {num_steps - 1}], got {self.train_timestep_max}")

        if zero_init_output:
            output_layer: nn.Linear | nn.Conv2d
            if isinstance(self.generative_backbone, DarknetConvUNet2D):
                output_layer = self.generative_backbone.out
            elif isinstance(cast(TimeConditionedUTransformer, self.generative_backbone).head, nn.Sequential):
                output_layer = cast(nn.Linear, cast(TimeConditionedUTransformer, self.generative_backbone).head[-1])
            else:
                output_layer = cast(nn.Linear, cast(TimeConditionedUTransformer, self.generative_backbone).head)
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False)
        self.register_buffer("final_alpha_cumprod", torch.tensor(1.0, dtype=torch.float32), persistent=False)

        if sample_steps == 1:
            sample_timesteps = torch.tensor([self.sample_timestep_max], dtype=torch.long)
            sample_prev_timesteps = torch.tensor([-1], dtype=torch.long)
        else:
            sample_timesteps = torch.linspace(float(self.sample_timestep_max), 0.0, sample_steps).round().to(torch.long)
            sample_prev_timesteps = torch.cat([sample_timesteps[1:], sample_timesteps.new_tensor([-1])])
        self.register_buffer("sample_timesteps", sample_timesteps, persistent=False)
        self.register_buffer("sample_prev_timesteps", sample_prev_timesteps, persistent=False)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.training:
            return self.test(data)  # type: ignore[return-value]
        data = self.process_data(data)
        cond, target = self.preprocess_batch(data)
        extra = self.condition_extra(data, data["input"], step=self.start_frames)
        timesteps = torch.randint(0, self.train_timestep_max + 1, (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        sqrt_alpha = _extract_timestep_values(self.sqrt_alphas_cumprod, timesteps, target)
        sqrt_one_minus_alpha = _extract_timestep_values(self.sqrt_one_minus_alphas_cumprod, timesteps, target)
        prediction = self.predict_backbone(cond, sqrt_alpha * target + sqrt_one_minus_alpha * noise, timesteps, extra)
        loss_target = noise if self.predict_type == "epsilon" else target
        if self.predict_type == "v_prediction":
            loss_target = sqrt_alpha * noise - sqrt_one_minus_alpha * target
        return {"loss_rec": weighted_charbonnier_loss(prediction, loss_target, self.reconstruction_weight(target))}

    def sample_residual(self, cond: Tensor, extra: Tensor | None, sample_shape: tuple[int, ...]) -> Tensor:
        state = self.initialize_sample(cond, sample_shape)
        for timestep, prev_timestep in zip(self.sample_timesteps, self.sample_prev_timesteps):
            prediction = self.predict_backbone(cond, state, timestep, extra)
            ddim_noise = torch.randn_like(state) if self.ddim_eta > 0.0 else torch.zeros_like(state)
            state = _jit_ddim_step(
                prediction,
                timestep,
                prev_timestep,
                state,
                ddim_noise,
                self.alphas_cumprod,
                self.final_alpha_cumprod,
                self.ddim_eta,
                self.predict_type,
                self.clip_x0,
            )
        return state


@MODELS.register()
class FlowMatchingForecast(ODEGenerativeForecast):
    """Learn the straight conditional flow from Gaussian noise to a forecast residual."""

    backbone_module = "flow_net"

    def __init__(
        self,
        time_scale: float = 1000.0,
        rollout_max: int = 0,
        rollout_alpha: float = 1.0,
        rollout_sample_steps: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if time_scale <= 0:
            raise ValueError(f"time_scale must be positive, got {time_scale}")
        if rollout_max < 0:
            raise ValueError(f"rollout_max must be non-negative, got {rollout_max}")
        if not 0.0 < rollout_alpha <= 1.0:
            raise ValueError(f"rollout_alpha must be in (0, 1], got {rollout_alpha}")
        if rollout_sample_steps <= 0:
            raise ValueError(f"rollout_sample_steps must be positive, got {rollout_sample_steps}")
        self.time_scale = time_scale
        self.rollout_max = rollout_max
        self.rollout_alpha = rollout_alpha
        self.rollout_sample_steps = rollout_sample_steps

    def predict_velocity(self, cond: Tensor, state: Tensor, time: Tensor, extra: Tensor | None) -> Tensor:
        return self.predict_backbone(cond, state, time * self.time_scale, extra)

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.training:
            return self.test(data)  # type: ignore[return-value]
        data = self.process_data(data)
        input = data["input"]
        targets = data["targets"]
        future_steps = targets.shape[1]
        if future_steps == 0:
            raise ValueError("FlowMatchingForecast rollout training requires at least one target frame")

        loss_steps, rolling_weights = self.build_loss_steps_and_weights(future_steps, self.rollout_alpha, self.rollout_max)
        loss_steps = set(loss_steps)
        loss_flow = targets.new_zeros((), dtype=torch.float32)
        for step in range(future_steps):
            input = self.reset_input(input)
            cond = self.prepare_condition(input)
            target = self.prepare_target(input, targets[:, step : step + 1])
            extra = self.condition_extra(data, input, step=self.start_frames + step)
            if step in loss_steps:
                time = torch.rand(target.shape[0], device=target.device, dtype=torch.float32)
                time_view = time.reshape(target.shape[0], *([1] * (target.ndim - 1)))
                noise = torch.randn_like(target) * self.noise_scale
                state = (1.0 - time_view) * noise + time_view * target
                loss_flow = loss_flow + rolling_weights[step] * weighted_mse_loss(
                    self.predict_velocity(cond, state, time, extra), target - noise, self.reconstruction_weight(target)
                )
            if step + 1 < future_steps:
                with torch.no_grad():
                    rollout_state = torch.randn_like(target) * self.noise_scale
                    output = self.merge_prediction(input, self._integrate_residual(cond, extra, rollout_state, self.rollout_sample_steps))
                    input = torch.cat([input, output], dim=1)[:, -self.in_frames :].detach()
        return {"loss_flow": loss_flow}


@MODELS.register()
class JiTForecast(ODEGenerativeForecast):
    """Optimize the velocity induced by a clean-residual prediction backbone."""

    backbone_module = "jit"

    def __init__(self, p_mean: float = -0.8, p_std: float = 0.8, t_eps: float = 0.05, **kwargs: Any):
        super().__init__(**kwargs)
        if p_std < 0:
            raise ValueError(f"p_std must be non-negative, got {p_std}")
        if not 0 < t_eps < 1:
            raise ValueError(f"t_eps must be in (0, 1), got {t_eps}")
        self.p_mean = p_mean
        self.p_std = p_std
        self.t_eps = t_eps

    def predict_velocity(self, cond: Tensor, state: Tensor, time: Tensor, extra: Tensor | None) -> Tensor:
        clean = self.predict_backbone(cond, state, time, extra)
        denominator = (1.0 - time.to(state)).reshape(state.shape[0], 1, 1, 1).clamp_min(self.t_eps)
        return (clean - state) / denominator

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        if not self.training:
            return self.test(data)  # type: ignore[return-value]
        data = self.process_data(data)
        cond, target = self.preprocess_batch(data)
        extra = self.condition_extra(data, data["input"], step=self.start_frames)
        time = torch.sigmoid(torch.randn(target.shape[0], device=target.device) * self.p_std + self.p_mean)
        time_view = time.reshape(target.shape[0], *([1] * (target.ndim - 1))).to(target)
        noise = torch.randn_like(target) * self.noise_scale
        state = time_view * target + (1.0 - time_view) * noise
        target_velocity = (target - state) / (1.0 - time_view).clamp_min(self.t_eps)
        return {"loss_jit": weighted_mse_loss(self.predict_velocity(cond, state, time, extra), target_velocity, self.reconstruction_weight(target))}
