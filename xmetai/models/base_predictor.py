"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-11-05 17:31:46
Description: Base predictor class for weather/climate models.
Copyright (c) 2025 by XMetAI/douzsheng, All Rights Reserved.
"""

import logging
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from xmetai.loss import ChannelRegistry, pinn_loss, precip_loss

from .meta import Meta

__all__ = ["BasePredictor"]


class BasePredictor(Meta):
    """Utility mixin for XMetAI predictors.

    Provides shared logic for normalization, masking, timestamp features, and
    NetCDF exports so concrete predictor implementations only need to define
    ``forward`` and ``test_cascade``.

    Args:
        in_chans: Number of input channels provided to the model.
        out_chans: Number of channels reconstructed during decoding.
        in_frames: Historical timesteps consumed by the model.
        out_frames: Forecast timesteps produced by the model.
        members: Ensemble size for rollouts and evaluation.
        step_range: Valid forecast horizons (in hours) for cascading tests.
        start_frames: Amount of historical data skipped before training.
        test_frames: Forecast indices exported during testing.
        test_chans: Channel ids evaluated or written to disk.
        test_names: Human-readable names for ``test_chans`` (e.g., TP, T2M).
        test_rect: Optional spatial crop ``[y1, y2, x1, x2]`` used at eval.
        accumid: Channel ids treated as accumulated quantities.
        logid: Channel ids stored in log space (e.g., precipitation).
        maskid: Channel ids masked by the ocean mask buffer.
        landid: Channel ids masked by the land mask buffer.
        zero_accum: Zero accumulated channels before issuing a forecast.
        add_noise: Whether to add stochastic perturbations to the inputs.
        save_dir: Directory used to dump NetCDF predictions/observations when existing.
        buffers: Optional mapping whose numpy arrays become registered buffers.

    Notes:
        Buffers such as ``mean``, ``std``, ``mask``, and ``weight`` are consumed
        by helper methods like ``normalize`` and ``process_output``. Longitude
        shifts are handled automatically inside ``process_data`` so masks and
        constant fields stay aligned with each sample.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        in_frames: int,
        out_frames: int,
        members: int = 1,
        step_range: list[int] = [20],
        start_frames: int = 0,
        test_frames: list[int] = [1],
        test_chans: list[int] = [-1],
        test_names: list[str] = ["tp", "t2m"],
        test_rect: list[int] = [],
        freq: int = 24,
        accumid: list[int] = [],
        logid: list[int] = [],
        maskid: list[int] = [],
        landid: list[int] = [],
        zero_accum: bool = False,
        add_noise: bool = False,
        save_dir: str = "",
        buffers: dict = {},
        *args: Any,
        **kwargs: Any,
    ):
        # Extract PINN configuration from kwargs before passing remaining args to Meta
        self.pinn_enable = kwargs.pop("pinn_enable", False)
        self.pinn_coeffs = kwargs.pop("pinn_coeffs", None)
        self.in_names = kwargs.pop("in_names", None)
        self.precip_loss_enable = kwargs.pop("precip_loss_enable", False)
        self.precip_loss_coeffs = kwargs.pop("precip_loss_coeffs", None)
        precip_names = kwargs.pop("precip_names", ("tp", "ncpcp"))
        precip_names = (precip_names,) if isinstance(precip_names, str) else tuple(precip_names)
        precip_threshold_mm = float(kwargs.pop("precip_threshold_mm", 0.1))
        self.precip_occurrence_scale = float(kwargs.pop("precip_occurrence_scale", 10.0))

        # If True, convert normalized model outputs/targets back to physical
        # units before evaluating PINN losses. This preserves gradients
        # (we implement denorm inline below) so PINN losses remain
        # differentiable when used during training.
        self.pinn_use_denorm = kwargs.pop("pinn_use_denorm", False)

        super().__init__(*args, **kwargs)

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.members = members
        self.step_range = step_range
        self.start_frames = start_frames
        self.test_frames = test_frames.tolist() if isinstance(test_frames, np.ndarray) else list(test_frames)
        self.test_chans = test_chans.tolist() if isinstance(test_chans, np.ndarray) else list(test_chans)
        self.test_names = test_names
        self.test_rect = test_rect
        self.freq = freq
        self.save_dir = save_dir

        self.zero_accum = zero_accum
        self.accumid = accumid
        self.logid = logid
        self.maskid = maskid
        self.landid = landid
        self.add_noise = add_noise
        self.logger = logging.getLogger(__name__)

        for k, v in buffers.items():
            v = torch.from_numpy(v)
            self.logger.info(f"Register for buffer {k} with shape:{v.shape}")
            if k.find("mask") >= 0:
                # make sure mask is binary
                v = (v > 0).float()
                self.logger.info(f"{k} shape: {v.shape} with 1 percent: {v.sum() / v.numel() * 100:.2f}%")
            if k in ["mean", "std", "channel_weights"]:
                self.logger.info(f"{k} shape: {v.shape}, values: {v.reshape(-1)}")
            self.register_buffer(k, v)
        self.logger.info(f"using mask(ocean) id: {self.maskid}, land mask id: {self.landid}")
        self.logger.info(f"using log norm id: {self.logid}")
        self.logger.info(f"accumulate id: {self.accumid}. Do zero accumulate: {self.zero_accum}")
        if self.in_names is not None:
            self.logger.info(f"input channel names: {self.in_names}")

        if self.pinn_enable and self.in_names is not None:
            self.pinn_registry = ChannelRegistry(list(self.in_names))
            self.logger.info(f"enabling PINN losses with coefficients: {self.pinn_coeffs}")
            self.logger.info(f"PINN loss use denormalized inputs: {self.pinn_use_denorm}")
        elif self.pinn_enable and self.in_names is None:
            self.pinn_enable = False  # disable PINN if we don't have channel names to construct the registry
            self.logger.error(
                "pinn_enable=True but `in_names` was not provided to the model config. "
                "PINN registry will be constructed lazily when compute_pinn_losses is invoked; "
                "ensure `in_names` or model.buffers with channel metadata are supplied to avoid runtime errors."
            )
        else:
            self.logger.info("PINN losses are disabled.")

        if self.precip_loss_enable:
            self._init_precip_loss(precip_names, precip_threshold_mm)
        else:
            self.logger.info("Precipitation auxiliary losses are disabled.")

    def train(self, mode: bool = True) -> "BasePredictor":
        super().train(mode=mode)
        return self

    def get_timestamp(self, input: Any, times: Any, t: int) -> dict:
        times = times + pd.Timedelta(hours=t * self.freq)
        hour_list = [p.hour / 24 for p in times]
        hour_arr = np.array(hour_list, dtype=np.float32)
        hour = input.new_tensor(hour_arr)
        doy_list = [min(365, p.day_of_year) / 365 for p in times]
        doy_arr = np.array(doy_list, dtype=np.float32)
        doy = input.new_tensor(doy_arr)
        step = input.new_full(input.shape[:1], t)
        return dict(step=step, hour=hour, doy=doy)

    def build_loss_steps_and_weights(self, future_step: int, alpha: float, rollout_max: int) -> tuple:
        """Build selected loss steps and normalized rolling weights."""
        device = next(self.parameters()).device
        exponents = torch.arange(future_step, dtype=torch.float32, device=device)
        rolling_weights = torch.pow(alpha, exponents)
        loss_steps = set(range(future_step))
        if rollout_max > 0 and future_step > rollout_max:
            if rollout_max == 1:
                loss_steps = {future_step - 1}
            else:
                loss_steps = {0, future_step - 1}
                middle_steps = list(range(1, future_step - 1))
                num_additional = rollout_max - len(loss_steps)
                if num_additional > 0 and len(middle_steps) > 0:
                    num_to_sample = min(num_additional, len(middle_steps))
                    gen = torch.Generator(device=rolling_weights.device)
                    gen.manual_seed(future_step * 1070001 + rollout_max)
                    selected = torch.randperm(len(middle_steps), generator=gen, device=device)[:num_to_sample].tolist()
                    loss_steps.update(middle_steps[i] for i in selected)

        valid_steps = sorted(loss_steps)
        if len(valid_steps) < future_step:
            mask = torch.zeros_like(rolling_weights)
            mask[valid_steps] = 1.0
            rolling_weights = rolling_weights * mask

        rolling_weights = rolling_weights / rolling_weights.sum().clamp_min(torch.finfo(rolling_weights.dtype).eps)
        return valid_steps, rolling_weights

    @torch.no_grad()
    def inv_normalize(self, input, test_chans=[]):
        assert input.ndim == 5

        logids = self.logid
        if len(test_chans) > 0:
            # check if input is already only test_chans
            if input.size(2) != len(test_chans):
                x = input[:, :, test_chans].float()
            else:
                x = input.float()
            mean = self.mean[test_chans].to(x)
            std = self.std[test_chans].to(x)
            logids = []
            for id in self.logid:
                if id in test_chans:
                    logids.append(test_chans.index(id))
        else:
            x = input.float()
            mean = self.mean.to(x)
            std = self.std.to(x)

        x = x * std + mean

        if len(logids) > 0:
            # for tp should be exp-1.
            tp = x[:, :, logids].clamp(min=0.0, max=7.0)
            tp = tp.exp() - 1
            x[:, :, logids] = tp
        return x

    @torch.no_grad()
    def normalize(self, input, test_chans=[]):
        assert input.ndim == 5
        logids = self.logid
        if len(test_chans) > 0:
            # check if input is already only test_chans
            if input.size(2) != len(test_chans):
                x = input[:, :, test_chans].float()
            else:
                x = input.float()
            mean = self.mean[test_chans].to(x)
            std = self.std[test_chans].to(x)
            logids = []
            for id in self.logid:
                if id in test_chans:
                    logids.append(test_chans.index(id))
        else:
            x = input.float()
            mean = self.mean.to(x)
            std = self.std.to(x)

        x = (x - mean) / std

        if len(logids) > 0:
            # for tp should be log(1+tp).
            tp = torch.clamp(x[:, :, logids], min=0, max=7)
            x[:, :, logids] = torch.log1p(tp)
        return x

    def reset_input(self, input):
        if len(self.accumid) > 0 and self.zero_accum:
            input[:, :, self.accumid] *= 0
        if len(self.maskid) > 0:
            if hasattr(self, "mask_shift"):
                input[:, :, self.maskid] *= self.mask_shift
            else:
                input[:, :, self.maskid] *= self.mask
        if len(self.landid) > 0:
            if hasattr(self, "land_mask_shift"):
                input[:, :, self.landid] *= self.land_mask_shift
            else:
                input[:, :, self.landid] *= self.land_mask
        if len(self.logid) > 0:
            input[:, :, self.logid] = input[:, :, self.logid].clamp(min=0, max=7)
        return input

    def process_data(self, data):
        # convert inputs to proper dtype and device
        inputs = data["inputs"]
        input = inputs[:, -self.in_frames :]
        input = input.to(dtype=self.dtype, device=self.device, non_blocking=True)
        # convert times to pandas datetime
        times = data["times"]
        times = pd.to_datetime(times.cpu().numpy())

        batch = dict(times=times, input=input)

        const = None
        if "const" in self._buffers:
            const = self.const.to(input)
            batch["const"] = const

        if "targets" in data:
            targets = data["targets"]
            targets = targets.to(device=self.device, non_blocking=True)
            if self.training:
                future_frames = data.get("future_frames", self.out_frames)
                targets = targets[:, :future_frames]
            batch["targets"] = targets

        # check if shift exist in data, do rolling for const
        if "shift" in data:
            shift = data["shift"]
            batch["shift"] = shift
            if const is not None:  # const is const_dimx721x1440
                # expand const to batch size and do rolling
                const = torch.repeat_interleave(const.unsqueeze(0), repeats=input.size(0), dim=0)
                batch["const"] = torch.stack([torch.roll(const[i], shifts=int(s), dims=-1) for i, s in enumerate(shift)])
            if hasattr(self, "mask"):
                self.mask_shift = torch.stack([torch.roll(self.mask, shifts=int(s), dims=-1) for i, s in enumerate(shift)])
                self.mask_shift = self.mask_shift.unsqueeze(1).unsqueeze(2)
            if hasattr(self, "land_mask"):
                self.land_mask_shift = torch.stack([torch.roll(self.land_mask, shifts=int(s), dims=-1) for i, s in enumerate(shift)])
                self.land_mask_shift = self.land_mask_shift.unsqueeze(1).unsqueeze(2)
        else:
            if hasattr(self, "mask"):
                self.mask_shift = self.mask
            if hasattr(self, "land_mask"):
                self.land_mask_shift = self.land_mask

        return batch

    @torch.no_grad()
    def process_output(self, output, target=None):
        output = self.inv_normalize(output[:, :, : self.out_chans], self.test_chans)
        if target is not None:
            target = self.inv_normalize(target, self.test_chans)
            assert output.shape == target.shape, f"output shape: {output.shape} not match target shape: {target.shape}"

        if len(self.test_rect) == 4:
            y1, y2, x1, x2 = self.test_rect
            output = output[:, :, :, y1:y2, x1:x2]
            if target is not None:
                target = target[:, :, :, y1:y2, x1:x2]

        self.logger.debug(f"process output channels: output shape is---> {output.shape}")

        result = dict(
            output=output,
        )

        if target is not None:
            result.update(dict(target=target))

        if "weight" in self._buffers:
            weight = self.weight.float()

            if len(self.test_rect) == 4:
                if weight.ndim == 3:
                    weight = weight[:, y1:y2]
                elif weight.ndim == 4:
                    weight = weight[:, :, y1:y2, x1:x2]

            result.update(dict(weight=weight))

        return result

    def forward(self, data):
        # Concrete predictors return a dict of losses during training. BasePredictor
        # itself does not implement forward; concrete subclasses should override.
        raise NotImplementedError

    def _init_precip_loss(self, precip_names: tuple[str, ...], precip_threshold_mm: float) -> None:
        if self.in_names is None:
            raise ValueError("precip_loss_enable=True requires `in_names` to locate precipitation channels.")
        if len(precip_names) == 0:
            raise ValueError("precip_names must contain at least one channel name when precipitation loss is enabled.")
        if precip_threshold_mm < 0:
            raise ValueError(f"precip_threshold_mm must be non-negative, got {precip_threshold_mm}.")
        if self.precip_occurrence_scale <= 0:
            raise ValueError(f"precip_occurrence_scale must be positive, got {self.precip_occurrence_scale}.")
        if "mean" not in self._buffers or "std" not in self._buffers:
            raise ValueError("precip_loss_enable=True requires `mean` and `std` buffers to convert the millimeter threshold.")

        name_to_index = {name: idx for idx, name in enumerate(self.in_names)}
        indices = [name_to_index[name] for name in precip_names if name in name_to_index]
        if len(indices) == 0:
            raise ValueError(f"precip_loss_enable=True but none of precip_names={precip_names} were found in in_names.")
        if len(indices) > 1:
            matched_names = [self.in_names[idx] for idx in indices]
            raise ValueError(
                "precip_loss_enable=True with a scalar precipitation threshold supports exactly one matched channel. "
                f"Got channels={matched_names}; set precip_names to one channel."
            )

        channel_idx = indices[0]
        if self.mean.shape[0] <= channel_idx or self.std.shape[0] <= channel_idx:
            raise ValueError(f"mean/std buffers do not cover precipitation channel index {channel_idx}.")

        mean = self.mean[channel_idx].detach().float()
        std = self.std[channel_idx].detach().float()
        if mean.numel() != 1 or std.numel() != 1:
            raise ValueError("mean/std buffers for precipitation threshold conversion must be scalar per channel.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("mean/std buffers for precipitation channels must be finite and std must be positive.")

        threshold = precip_threshold_mm
        logid = {int(idx) for idx in self.logid}
        if channel_idx in logid:
            threshold = math.log1p(threshold)
        self.precip_threshold = float((threshold - mean.item()) / std.item())
        self.precip_indices = tuple(indices)
        self.logger.info(
            f"enabling precipitation losses for channels={[self.in_names[idx] for idx in indices]}, coeffs={self.precip_loss_coeffs}, "
            f"threshold_mm={precip_threshold_mm}, occurrence_scale={self.precip_occurrence_scale}"
        )

    def compute_precip_losses(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute precipitation auxiliary losses when enabled."""
        if not self.precip_loss_enable:
            return {}

        loss_dict = precip_loss(
            output,
            target,
            self.precip_indices,
            self.precip_threshold,
            coefficients=self.precip_loss_coeffs,
            occurrence_scale=self.precip_occurrence_scale,
            reduction="mean",
        )
        return {f"precip_{k}": v for k, v in loss_dict.items()}

    def compute_pinn_losses(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        """Helper that computes PINN losses when enabled.

        Returns a mapping of pinn loss key -> scalar tensor (already scaled by coeffs).
        Concrete subclasses may call self.compute_pinn_losses(output, target) after
        producing their standard loss_dict and merge the returned dict into it.
        """
        if not self.pinn_enable:
            return {}

        # Optionally evaluate PINN losses in denormalized (physical) units.
        # We avoid calling `inv_normalize` here because that helper is
        # decorated with `@torch.no_grad()` and would break gradient flow.
        weight = getattr(self, "weight", None)

        if self.pinn_use_denorm:
            # Enable gradients for denormalization to preserve gradient flow
            with torch.enable_grad():
                out_for_pinn = self.inv_normalize(output)
                tgt_for_pinn = self.inv_normalize(target)
        else:
            out_for_pinn = output
            tgt_for_pinn = target

        pinn_dict = pinn_loss(out_for_pinn, tgt_for_pinn, self.pinn_registry, weight=weight, coefficients=self.pinn_coeffs, reduction="mean")
        return {f"pinn_{k}": v for k, v in pinn_dict.items()}

    @torch.no_grad()
    def test(self, data):
        data = self.process_data(data)
        times = data["times"]
        outputs = self.test_cascade(data)
        results = self.process_output(outputs, data.get("targets", None))
        if os.path.exists(self.save_dir):
            ymd_time = times[0].strftime("%Y%m%d")
            # get pred times start from ymd_time + self.freq * i, in self.test_frames
            pred_times = [times[-1] + pd.Timedelta(hours=self.freq * i) for i in self.test_frames]
            save_path = os.path.join(self.save_dir, f"pred_{ymd_time}.nc")
            self.logger.info(f"Saving results to {save_path}")
            # save results to netcdf
            xr_output = xr.DataArray(
                results["output"].squeeze().cpu().numpy(),
                dims=["time", "level", "lat", "lon"],
                coords={
                    "time": pred_times,
                    "level": self.test_names,
                    "lat": np.linspace(90, -90, results["output"].shape[3]),
                    "lon": np.linspace(0, 360, results["output"].shape[4]),
                },
                name="data",
            )
            xr_output.to_netcdf(save_path)

            # save target for comparison if available
            if "target" in results:
                save_path = os.path.join(self.save_dir, f"obs_{ymd_time}.nc")
                xr_output = xr.DataArray(
                    results["target"].squeeze().cpu().numpy(),
                    dims=["time", "level", "lat", "lon"],
                    coords={
                        "time": pred_times,
                        "level": self.test_names,
                        "lat": np.linspace(90, -90, results["target"].shape[3]),
                        "lon": np.linspace(0, 360, results["target"].shape[4]),
                    },
                    name="data",
                )
                xr_output.to_netcdf(save_path)
        return results

    def test_cascade(self, data):
        return NotImplemented
