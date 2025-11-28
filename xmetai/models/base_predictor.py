import os
import logging
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import pandas as pd

from xmetai.builder import MODELS
from .meta import Meta

__all__ = ["BasePredictor"]


@MODELS.register()
class BasePredictor(Meta):
    def __init__(
        self,
        in_chans,
        out_chans,
        in_frames,
        out_frames,
        members=1,
        step_range=[20],
        start_frames=0,
        test_frames=[1],
        test_chans=[-1],
        test_names=['tp'],
        test_rect=[],
        accumid=[],
        logid=[],
        maskid=[],
        landid=[],
        loss_weight=[1, 0.1],
        zero_accum=False,
        add_noise=False,
        save_dir="",
        buffers={},
        decoder=[],
        *args,
        **kwargs,
    ):
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
        self.save_dir = save_dir
        self.loss_weight = loss_weight

        self.zero_accum = zero_accum
        self.accumid = accumid
        self.logid = logid
        self.maskid = maskid
        self.landid = landid
        self.add_noise = add_noise
        self.decoder = nn.ModuleList(decoder)

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

    def train(self, mode=True):
        super().train(mode=mode)

    def get_timestamp(self, input, times, t, freq=6):
        times = times + pd.Timedelta(hours=t * freq)
        hour = [p.hour/24 for p in times]
        hour = np.array(hour, dtype=np.float32)
        hour = input.new_tensor(hour)
        doy = [min(365, p.day_of_year)/365 for p in times]
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
    def normalize(self, input):
        x = input.float()
        mean = self.mean.to(x)
        std = self.std.to(x)
        x = (x - mean) / std
        if len(self.logid) > 0:
            # for tp should be log(1+tp).
            tp = torch.clamp(x[:, :, self.logid], min=0, max=7)
            x[:, :, self.logid] = torch.log1p(tp)
        return x

    def reset_input(self, input):
        if len(self.accumid) > 0 and self.zero_accum:
            self.logger.debug(f"Resetting accumulate channels: {self.accumid}")
            input[:, :, self.accumid] *= 0
        if len(self.maskid) > 0:
            self.logger.debug(f"Applying mask to channels: {self.maskid}")
            input[:, :, self.maskid] *= self.mask_shift
        if len(self.landid) > 0:
            self.logger.debug(f"Applying land mask to channels: {self.landid}")
            input[:, :, self.landid] *= self.land_mask_shift
        if len(self.logid) > 0:
            self.logger.debug(f"Applying clamp to log channels: {self.logid}")
            input[:, :, self.logid] = input[:, :, self.logid].clamp(min=0, max=7)
        return input

    def process_data(self, data):
        # convert inputs to proper dtype and device
        inputs = data["inputs"]
        input = inputs[:, -self.in_frames:]
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
        output = self.inv_normalize(output[:, :, :self.out_chans], self.test_chans)
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
        return NotImplemented

    @torch.no_grad()
    def test(self, data, freq=24):
        data = self.process_data(data)
        times = data["times"]
        outputs = self.test_cascade(data)
        results = self.process_output(outputs, data.get("targets", None))
        if os.path.exists(self.save_dir):
            ymd_time = times[0].strftime("%Y%m%d")
            # get pred times start from ymd_time + freq * i, in self.test_frames
            pred_times = [times[-1] + pd.Timedelta(hours=freq * i) for i in self.test_frames]
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
                name="data"
            )
            xr_output.to_netcdf(save_path)

            # save target for comparison if available
            if 'target' in results:
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
                    name="data"
                )
                xr_output.to_netcdf(save_path)
        return results

    def test_cascade(self, data):
        return NotImplemented
