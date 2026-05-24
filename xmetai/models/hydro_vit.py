# xmetai/models/hydro_vit.py

import torch
import torch.nn.functional as F
from einops import rearrange

from xmetai.builder import MODELS
from xmetai.loss import charbonnier_loss
from .vit import ViTModel


@MODELS.register()
class HydroViTModel(ViTModel):
    """
    Hydro ViT model that keeps an autoregressive rollout, while supporting Cin != Cout
    (e.g. Cin=9, Cout=3) by:

    1) Predicting only Cout channels (h,u,v)
    2) Constructing a Cin-channel "next frame" for autoregressive concatenation:
       - known/diagnostic channels (rain + static) are COPIED from the last input frame
         (stopgap for many-to-one diagnostic setting where r_t is known).
       - predicted hydro channels are written into a slice (default last 3 channels).
    3) Computing loss ONLY on the Cout channels slice.

    Default channel semantics (assumed):
      inputs channels = [rain(1), static(5), hydro(3)] => total 9
      - known channels: first 6
      - predicted hydro channels: last 3
    """

    def __init__(
        self,
        residual_mode: str = "none",     # "none" or "last_channels"
        residual_in_start: int = -3,     # slice in input[:, -1, :] used for residual add (hydro channels)
        out_to_in_start: int = -3,       # where to write predicted Cout into Cin frame (hydro channels)
        known_in_start: int = 0,         # where known channels start in Cin frame
        known_in_chans: int = 6,         # number of known channels (rain+static)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.residual_mode = residual_mode
        self.residual_in_start = residual_in_start
        self.out_to_in_start = out_to_in_start

        self.known_in_start = known_in_start
        self.known_in_chans = known_in_chans

        if self.in_chans < self.out_chans:
            raise ValueError(
                f"HydroViTModel expects in_chans >= out_chans, got in_chans={self.in_chans}, out_chans={self.out_chans}"
            )
        if self.known_in_chans < 0:
            raise ValueError(f"known_in_chans must be >= 0, got {self.known_in_chans}")

        # Validate known slice and output slice
        self._slice_start_end(self.known_in_start, self.in_chans, self.known_in_chans)
        self._slice_start_end(self.out_to_in_start, self.in_chans, self.out_chans)

    def _slice_start_end(self, start: int, total: int, length: int):
        """Convert possibly-negative start to [start:end] slice indices with checks."""
        s = start
        if s < 0:
            s = total + s
        e = s + length

        if length == 0:
            return s, s
        if not (0 <= s < total) or e > total:
            raise ValueError(f"Invalid slice start={start} for total={total}, length={length} -> [{s}:{e}]")
        return s, e

    def _build_next_frame(self, last_frame: torch.Tensor, out_c: torch.Tensor) -> torch.Tensor:
        """
        Construct Cin-channel next frame for autoregressive rollout.

        last_frame: [B, Cin, H, W]  (input[:, -1])
        out_c:      [B, Cout, H, W] predicted hydro channels

        returns:    [B, Cin, H, W]
        """
        b, cin, h, w = last_frame.shape
        if cin != self.in_chans:
            raise RuntimeError(f"Expected last_frame Cin={self.in_chans}, got {cin}")
        if out_c.shape[1] != self.out_chans:
            raise RuntimeError(f"Expected out_c Cout={self.out_chans}, got {out_c.shape[1]}")

        next_frame = last_frame.new_zeros((b, self.in_chans, h, w))

        # 1) copy known channels (rain+static) from last input frame
        ks, ke = self._slice_start_end(self.known_in_start, self.in_chans, self.known_in_chans)
        if self.known_in_chans > 0:
            next_frame[:, ks:ke] = last_frame[:, ks:ke]

        # 2) write predicted hydro channels into configured slice
        os, oe = self._slice_start_end(self.out_to_in_start, self.in_chans, self.out_chans)
        next_frame[:, os:oe] = out_c

        return next_frame

    def forward_step(self, input, times=None, step=None, timestamp=None):
        """
        Return [B, 1, Cin, H, W] so autoregressive rollout can do:
            input = torch.cat([input, output], dim=1)[:, -self.in_frames:]
        """
        if timestamp is None:
            timestamp = self.get_timestamp(input, times, step)

        # time embeddings (same as ViTModel)
        time_token = self.step_embed(timestamp["step"]).unsqueeze(1)  # [B,1,C]
        time_token = torch.concat([time_token, self.hour_embed(timestamp["hour"]).unsqueeze(1)], dim=1)
        time_token = torch.concat([time_token, self.doy_embed(timestamp["doy"]).unsqueeze(1)], dim=1)

        # [B,T,C,H,W] -> [B, T*C, H, W]
        x = rearrange(input, "n t c h w -> n (t c) h w")
        if x.shape[-2:] != self.image_size:
            x = F.interpolate(x.float(), size=self.image_size, mode="bilinear", align_corners=False).to(input)

        h = self.patch_embed(x)  # [B, N, embed_dim]

        # apply time conditioning
        time_token = self.time_mlp(time_token.sum(dim=1))
        gamma, beta = time_token.chunk(2, dim=-1)
        h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # encoder + head decode to Cout channels
        h = self.encoder(h)
        out_c = self.head(h)  # [B, N, Cout*P*P]
        out_c = rearrange(
            out_c,
            "n (h w) (c p1 p2) -> n c (h p1) (w p2)",
            h=self.feat_size[0],
            p1=self.patch_size,
            p2=self.patch_size,
        )

        if out_c.shape[-2:] != input.shape[-2:]:
            out_c = F.interpolate(out_c.float(), size=input.shape[-2:], mode="bilinear", align_corners=False).to(input)

        # Optional residual add on hydro channels only (read from last_frame hydro slice)
        if self.residual_mode == "last_channels":
            rs, re = self._slice_start_end(self.residual_in_start, self.in_chans, self.out_chans)
            out_c = out_c + input[:, -1, rs:re]
        elif self.residual_mode != "none":
            raise ValueError(f"Unknown residual_mode={self.residual_mode}")

        # Build Cin-channel next frame with known channels copied + predicted hydro written
        last_frame = input[:, -1]  # [B, Cin, H, W]
        next_frame = self._build_next_frame(last_frame, out_c)  # [B, Cin, H, W]
        return next_frame.unsqueeze(1)  # [B, 1, Cin, H, W]

    @torch.no_grad()
    def test(self, data):
        """
        Override BasePredictor.test() to avoid ViTModel.test_cascade(), because our
        forward_step returns Cin-channel frames (9) for autoregressive training.

        IMPORTANT: xmetai/metrics/mse.py expects a dict with:
          outputs["output"], outputs["target"].
        """
        data = self.process_data(data)
        x = data["input"]  # [B,T,Cin,H,W]
        times = data["times"]
        y = data.get("targets", None)  # [B,Tout,Cout,H,W] (many-to-one => Tout=1)

        # one-step inference
        out_in = self.forward_step(x, times, step=self.start_frames)  # [B,1,Cin,H,W]

        # slice hydro channels where we wrote them
        os, oe = self._slice_start_end(self.out_to_in_start, self.in_chans, self.out_chans)
        out = out_in[:, :, os:oe]  # [B,1,Cout,H,W]

        # If targets are available, return both output and target for metrics.
        if y is None:
            return {"output": out}

        tgt = y[:, : out.size(1)]
        return {"output": out, "target": tgt}

    def forward(self, data):
        """
        Autoregressive training loop (many-to-one or multi-step), but compute loss ONLY on
        the Cout slice, because targets are [B,*,Cout,H,W].
        """
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]     # [B, Tin, Cin, H, W]
        times = data["times"]
        targets = data["targets"] # [B, Tf, Cout, H, W]

        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames

        # where predicted hydro channels live in the Cin frame
        os, oe = self._slice_start_end(self.out_to_in_start, self.in_chans, self.out_chans)

        loss_rec = 0.0
        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            target = targets[:, k:k + self.out_frames]  # [B,1,Cout,H,W]

            self.reset_input(input)
            output_in = self.forward_step(input, times, step=t)  # [B,1,Cin,H,W]

            # loss only on Cout slice
            pred = output_in[:, :, os:oe]  # [B,1,Cout,H,W]
            loss_rec = loss_rec + charbonnier_loss(pred, target)

            # autoregressive rollout in Cin space
            input = torch.cat([input, output_in], dim=1)[:, -self.in_frames:]

        return {"loss_rec": loss_rec / future_frames}
