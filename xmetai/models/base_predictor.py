"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-11-05 17:31:46
Description: Base predictor class for weather/climate models.
Copyright (c) 2025 by XMetAI/douzsheng, All Rights Reserved.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr

from xmetai.loss.pinn_loss import ChannelRegistry, pinn_loss

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
        *args,
        **kwargs,
    ):
        # Extract PINN configuration from kwargs before passing remaining args to Meta
        self.pinn_enable = kwargs.pop("pinn_enable", False)
        self.pinn_coeffs = kwargs.pop("pinn_coeffs", None)
        self.in_names = kwargs.pop("in_names", None)

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
            self.register_buffer(k, v)
        self.logger.info(f"using mask(ocean) id: {self.maskid}, land mask id: {self.landid}")
        self.logger.info(f"using log norm id: {self.logid}")
        self.logger.info(f"accumulate id: {self.accumid}. Do zero accumulate: {self.zero_accum}")

        if self.pinn_enable and self.in_names is not None:
            self.pinn_registry = ChannelRegistry(list(self.in_names))
        else:
            self.logger.error(
                "pinn_enable=True but `in_names` was not provided to the model config. "
                "PINN registry will be constructed lazily when compute_pinn_losses is invoked; "
                "ensure `in_names` or model.buffers with channel metadata are supplied to avoid runtime errors."
            )

    def train(self, mode=True):
        super().train(mode=mode)

    def get_timestamp(self, input, times, t):
        times = times + pd.Timedelta(hours=t * self.freq)
        hour = [p.hour / 24 for p in times]
        hour = np.array(hour, dtype=np.float32)
        hour = input.new_tensor(hour)
        doy = [min(365, p.day_of_year) / 365 for p in times]
        doy = np.array(doy, dtype=np.float32)
        doy = input.new_tensor(doy)
        step = input.new_full(input.shape[:1], t)
        return dict(step=step, hour=hour, doy=doy)

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
            self.logger.debug(f"Resetting accumulate channels: {self.accumid}")
            input[:, :, self.accumid] *= 0
        if len(self.maskid) > 0:
            self.logger.debug(f"Applying mask to channels: {self.maskid}")
            if hasattr(self, "mask_shift"):
                input[:, :, self.maskid] *= self.mask_shift
            else:
                input[:, :, self.maskid] *= self.mask
        if len(self.landid) > 0:
            self.logger.debug(f"Applying land mask to channels: {self.landid}")
            if hasattr(self, "land_mask_shift"):
                input[:, :, self.landid] *= self.land_mask_shift
            else:
                input[:, :, self.landid] *= self.land_mask
        if len(self.logid) > 0:
            self.logger.debug(f"Applying clamp to log channels: {self.logid}")
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
                future_frames = data.get("future_frames", 1)
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
            assert output.shape == target.shape

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

    def compute_pinn_losses(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        """Helper that computes PINN losses when enabled.

        Returns a mapping of pinn loss key -> scalar tensor (already scaled by coeffs).
        Concrete subclasses may call self.compute_pinn_losses(output, target) after
        producing their standard loss_dict and merge the returned dict into it.
        """
        if not self.pinn_enable:
            return {}

        pinn_dict = pinn_loss(output, target, self.pinn_registry, weight=getattr(self, "weight", None), coefficients=self.pinn_coeffs, reduction="mean")
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
