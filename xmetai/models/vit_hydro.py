# xmetai/models/vit_hydro.py

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xmetai.builder import MODELS
from xmetai.loss import charbonnier_loss
from .vit import ViTModel


@MODELS.register()
class HydroViTModel(ViTModel):
    """
    HydroViTModel for flood forecasting.

    Input:
        inputs  : [B, T, 4, H, W], channels = [rain, h, u, v]
        const   : [B, 5, H, W], channels = [dem, lai, sand, clay, som]
        targets : [B, Tf, 3, H, W], channels = [h, u, v]

    Notes:
        - No time embeddings.
        - Static fields are encoded by an extra conv branch.
        - Decoder uses token grid -> conv upsampling -> refine CNN.
        - Training uses forward().
        - Evaluation / inference uses test() -> test_cascade().
    """

    def __init__(
        self,
        const_chans: int = 5,
        residual_mode: str = "prev_state",   # "none" or "prev_state"
        rain_ch_idx: int = 0,
        hydro_ch_start: int = 1,
        debug_h_scale: float = 30.0,
        save_test_fig: bool = True,
        save_test_nc: bool = False,
        test_save_dir: str = "",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.const_chans = int(const_chans)
        self.residual_mode = str(residual_mode).lower().strip()
        self.rain_ch_idx = int(rain_ch_idx)
        self.hydro_ch_start = int(hydro_ch_start)
        self.history_steps = self.in_frames - 1

        self.debug_h_scale = float(debug_h_scale)
        self.save_test_fig = bool(save_test_fig)
        self.save_test_nc = bool(save_test_nc)
        self.test_save_dir = str(test_save_dir) if test_save_dir else self.save_dir

        embed_dim = self.head[0].in_features

        self.const_patch_embed = nn.Conv2d(
            in_channels=self.const_chans,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        dec_mid1 = max(embed_dim // 2, 32)
        dec_mid2 = max(embed_dim // 4, 16)
        refine_ch = max(32, self.out_chans * 8)

        self.decoder_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(embed_dim, dec_mid1, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dec_mid1, dec_mid1, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(dec_mid1, dec_mid2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dec_mid2, dec_mid2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.decoder_out = nn.Conv2d(
            dec_mid2,
            self.out_chans,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.refine = nn.Sequential(
            nn.Conv2d(self.out_chans, refine_ch, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(refine_ch, refine_ch, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(refine_ch, self.out_chans, kernel_size=3, padding=1, bias=True),
        )

    # ------------------------------------------------------------------
    # basic data helpers
    # ------------------------------------------------------------------
    def process_data(self, data):
        inputs = data["inputs"][:, -self.in_frames:]
        const = data["const"]

        batch = {
            "input": inputs.to(dtype=self.dtype, device=self.device, non_blocking=True),
            "const": const.to(dtype=self.dtype, device=self.device, non_blocking=True),
        }

        if "targets" in data:
            targets = data["targets"].to(device=self.device, non_blocking=True)
            if self.training:
                future_frames = int(data.get("future_frames", 1))
                targets = targets[:, :future_frames]
            batch["targets"] = targets

        if "step" in data:
            batch["step"] = data["step"].to(device=self.device, non_blocking=True)
        else:
            batch["step"] = torch.zeros(inputs.size(0), dtype=torch.long, device=self.device)

        if "future_rain" in data:
            batch["future_rain"] = data["future_rain"].to(
                dtype=self.dtype, device=self.device, non_blocking=True
            )

        if "future_forcing" in data:
            batch["future_forcing"] = data["future_forcing"].to(
                dtype=self.dtype, device=self.device, non_blocking=True
            )

        if "history_mask" in data:
            batch["history_mask"] = data["history_mask"].to(
                device=self.device, non_blocking=True
            )

        return batch

    def get_timestamp(self, input, times=None, t=0):
        return {}

    def _extract_input(self, data):
        return data["input"] if "input" in data else data["inputs"]

    def _extract_targets(self, data):
        return data.get("targets", data.get("target", None))

    def _extract_step(self, data):
        return data.get("step", None)

    def _extract_future_rain(self, data):
        rain = data.get("future_rain", None)
        if rain is None:
            rain = data.get("future_forcing", None)
        if rain is None:
            return None
        if rain.ndim == 4:
            rain = rain.unsqueeze(2)
        return rain

    # ------------------------------------------------------------------
    # sequence helpers
    # ------------------------------------------------------------------
    def _build_state_frame(self, rain_now: torch.Tensor, hydro_pred: torch.Tensor) -> torch.Tensor:
        return torch.cat([rain_now, hydro_pred], dim=1)

    def _build_forcing_frame(self, rain_next: torch.Tensor, ref_frame: torch.Tensor) -> torch.Tensor:
        b, _, h, w = ref_frame.shape
        forcing = ref_frame.new_zeros((b, self.in_chans, h, w))
        forcing[:, self.rain_ch_idx:self.rain_ch_idx + 1] = rain_next
        return forcing

    def _roll_input_window(
        self,
        input_dyn: torch.Tensor,
        hydro_pred: torch.Tensor,
        rain_next: torch.Tensor,
    ) -> torch.Tensor:
        rain_now = input_dyn[:, -1, self.rain_ch_idx:self.rain_ch_idx + 1]
        state_t = self._build_state_frame(rain_now, hydro_pred).unsqueeze(1)
        forcing_next = self._build_forcing_frame(rain_next, input_dyn[:, -1]).unsqueeze(1)

        if self.history_steps > 1:
            kept_states = input_dyn[:, 1:self.history_steps]
            next_input = torch.cat([kept_states, state_t, forcing_next], dim=1)
        else:
            next_input = torch.cat([state_t, forcing_next], dim=1)

        return next_input

    # ------------------------------------------------------------------
    # model core
    # ------------------------------------------------------------------
    def _encode_tokens(self, input_dyn, const, step=None, timestamp=None):
        x_dyn = rearrange(input_dyn, "b t c h w -> b (t c) h w")
        if x_dyn.shape[-2:] != self.image_size:
            x_dyn = F.interpolate(
                x_dyn.float(),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).to(input_dyn)

        dyn_tokens = self.patch_embed(x_dyn)

        if const.shape[-2:] != self.image_size:
            const = F.interpolate(
                const.float(),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).to(input_dyn)

        const_tokens = self.const_patch_embed(const)
        const_tokens = rearrange(const_tokens, "b c h w -> b (h w) c")

        return dyn_tokens + const_tokens

    def _decode_hydro(self, tokens, out_hw, ref_tensor):
        h = self.encoder(tokens)

        feat_h, feat_w = self.feat_size
        feat = rearrange(h, "b (hh ww) c -> b c hh ww", hh=feat_h, ww=feat_w)

        feat = self.decoder_proj(feat)

        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.decoder_block1(feat)

        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.decoder_block2(feat)

        if feat.shape[-2:] != out_hw:
            feat = F.interpolate(
                feat,
                size=out_hw,
                mode="bilinear",
                align_corners=False,
            )

        output = self.decoder_out(feat)
        output = output + self.refine(output)

        return output.to(ref_tensor)

    def forward_step(self, input, times=None, step=None, timestamp=None, const=None):
        tokens = self._encode_tokens(
            input_dyn=input,
            const=const,
            step=step,
            timestamp=timestamp,
        )

        hydro_pred = self._decode_hydro(
            tokens=tokens,
            out_hw=input.shape[-2:],
            ref_tensor=input,
        )

        if self.residual_mode == "prev_state":
            hydro_base = input[:, -2, self.hydro_ch_start:self.hydro_ch_start + self.out_chans]
            hydro_pred = hydro_pred + hydro_base

        return hydro_pred.unsqueeze(1)

    # ------------------------------------------------------------------
    # test / rollout / export
    # ------------------------------------------------------------------
    @torch.no_grad()
    def test_cascade(self, data):
        input_dyn = data["input"]
        const = data["const"]
        targets = data.get("targets", None)
        start_step = data["step"]

        future_frames = 1 if targets is None else targets.size(1)
        future_rain = self._extract_future_rain(data)

        preds = []
        cur_input = input_dyn

        for k in range(future_frames):
            pred = self.forward_step(
                input=cur_input,
                const=const,
                step=start_step + k,
            )
            preds.append(pred)

            if k == future_frames - 1:
                break

            if future_rain is None:
                break

            rain_next = future_rain[:, k]
            cur_input = self._roll_input_window(
                input_dyn=cur_input,
                hydro_pred=pred[:, 0],
                rain_next=rain_next,
            )

        output = torch.cat(preds, dim=1)
        return output

    def _to_python_scalar(self, v):
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.item()
            return v.detach().cpu().tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    def _extract_meta(self, raw_data):
        def first_of(x):
            if isinstance(x, torch.Tensor):
                if x.ndim == 0:
                    return self._to_python_scalar(x)
                return self._to_python_scalar(x[0])
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return None
                return self._to_python_scalar(x[0])
            return self._to_python_scalar(x)

        return {
            "sample_id": first_of(raw_data.get("meta_sample_id", None)),
            "zarr_idx": first_of(raw_data.get("meta_zarr_idx", None)),
            "num": first_of(raw_data.get("meta_num", None)),
            "tag": first_of(raw_data.get("meta_tag", None)),
            "x": first_of(raw_data.get("meta_x", None)),
            "y": first_of(raw_data.get("meta_y", None)),
            "step": first_of(raw_data.get("step", None)),
        }

    def _build_file_stem(self, meta):
        sample_id = meta.get("sample_id", "none")
        zarr_idx = meta.get("zarr_idx", "none")
        step = meta.get("step", "none")
        num = meta.get("num", "none")
        tag = meta.get("tag", "none")
        iter_id = "eval"
        try:
            from xmetai.utils.events import get_event_storage
            iter_id = f"iter_{get_event_storage().iter}"
        except Exception:
            pass
        return f"{iter_id}_step_{step}_num_{num}_tag_{tag}_sample_{sample_id}_zarr_{zarr_idx}"

    def _save_nc_results(self, output, target, meta):
        if not self.test_save_dir or (not self.save_test_nc):
            return

        os.makedirs(self.test_save_dir, exist_ok=True)

        stem = self._build_file_stem(meta)
        pred_path = os.path.join(self.test_save_dir, f"{stem}_pred.nc")
        obs_path = os.path.join(self.test_save_dir, f"{stem}_obs.nc")

        output_np = output[0].detach().float().cpu().numpy()
        H, W = output_np.shape[-2], output_np.shape[-1]
        times = np.arange(output_np.shape[0])
        levels = ["h", "u", "v"]

        xr_pred = xr.DataArray(
            output_np,
            dims=["time", "level", "y", "x"],
            coords={
                "time": times,
                "level": levels,
                "y": np.arange(H),
                "x": np.arange(W),
            },
            name="pred",
        )
        xr_pred.attrs.update(meta)
        xr_pred.to_netcdf(pred_path)

        if target is not None:
            target_np = target[0].detach().float().cpu().numpy()
            xr_obs = xr.DataArray(
                target_np,
                dims=["time", "level", "y", "x"],
                coords={
                    "time": np.arange(target_np.shape[0]),
                    "level": levels,
                    "y": np.arange(H),
                    "x": np.arange(W),
                },
                name="obs",
            )
            xr_obs.attrs.update(meta)
            xr_obs.to_netcdf(obs_path)

    def _save_compare_grid_24pairs(self, output, target, meta):
        """
        Save one large figure with 48 subplots:
            24 obs + 24 pred
        Each time step is a close pair:
            left = obs, right = pred
        Each pair uses its own colorbar.
        """
        if not self.test_save_dir or (not self.save_test_fig):
            return
        if target is None:
            return

        os.makedirs(self.test_save_dir, exist_ok=True)

        # first sample, h channel only
        pred_h = output[0, :, 0].detach().float().cpu().numpy() * self.debug_h_scale
        true_h = target[0, :, 0].detach().float().cpu().numpy() * self.debug_h_scale

        n_frames = min(24, pred_h.shape[0], true_h.shape[0])
        if n_frames <= 1:
            return

        rows = 6
        pairs_per_row = 4
        cols = pairs_per_row * 2

        fig, axes = plt.subplots(rows, cols, figsize=(24, 20), dpi=150)
        axes = np.asarray(axes)

        for i in range(rows * pairs_per_row):
            r = i // pairs_per_row
            c0 = (i % pairs_per_row) * 2
            c1 = c0 + 1

            ax_obs = axes[r, c0]
            ax_pred = axes[r, c1]

            ax_obs.axis("off")
            ax_pred.axis("off")

            if i >= n_frames:
                continue

            obs_arr = true_h[i]
            pred_arr = pred_h[i]
            vmax = max(float(obs_arr.max()), float(pred_arr.max()), 1e-6)

            im_obs = ax_obs.imshow(obs_arr, vmin=0.0, vmax=vmax)
            im_pred = ax_pred.imshow(pred_arr, vmin=0.0, vmax=vmax)

            ax_obs.set_title(f"t={i+1} obs", fontsize=8)
            ax_pred.set_title(f"t={i+1} pred", fontsize=8)

            cbar = fig.colorbar(
                im_pred,
                ax=[ax_obs, ax_pred],
                fraction=0.025,
                pad=0.01,
            )
            cbar.ax.tick_params(labelsize=6)

        title = (
            f"Hydro depth comparison (24 obs + 24 pred)\n"
            f"sample_id={meta.get('sample_id')}, "
            f"zarr_idx={meta.get('zarr_idx')}, "
            f"num={meta.get('num')}, "
            f"tag={meta.get('tag')}, "
            f"x={meta.get('x')}, "
            f"y={meta.get('y')}"
        )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        stem = self._build_file_stem(meta)
        fig_path = os.path.join(self.test_save_dir, f"{stem}_compare_24pairs.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def test(self, data):
        raw_data = data
        data = self.process_data(data)

        output = self.test_cascade(data)
        target = data.get("targets", None)

        if target is not None:
            target = target[:, :output.size(1)]

        meta = self._extract_meta(raw_data)

        self._save_nc_results(output, target, meta)
        self._save_compare_grid_24pairs(output, target, meta)

        result = {"output": output}
        if target is not None:
            result["target"] = target
        return result

    @torch.no_grad()
    def onnx_export(self, input, const, step=None):
        """
        ONNX export interface for hydro model.

        input : [B, T, 4, H, W]
        const : [B, 5, H, W]
        """
        input = input.to(dtype=self.dtype)
        const = const.to(dtype=self.dtype)

        if step is None:
            step = torch.zeros(input.size(0), dtype=torch.long, device=input.device)

        output = self.forward_step(input=input, const=const, step=step)
        return output

    export_onnx = onnx_export

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)

        input_dyn = data["input"]
        const = data["const"]
        targets = data["targets"]
        start_step = data["step"]

        future_frames = targets.size(1)
        future_rain = self._extract_future_rain(data)

        if future_frames == 1:
            pred = self.forward_step(
                input=input_dyn,
                const=const,
                step=start_step,
            )
            loss_rec = charbonnier_loss(pred, targets[:, :1])
            return {"loss_rec": loss_rec}

        loss_rec = 0.0
        cur_input = input_dyn

        for k in range(future_frames):
            target = targets[:, k:k + 1]
            self.reset_input(cur_input)

            pred = self.forward_step(
                input=cur_input,
                const=const,
                step=start_step + k,
            )

            loss_rec = loss_rec + charbonnier_loss(pred, target)

            if k == future_frames - 1:
                break

            rain_next = future_rain[:, k]
            cur_input = self._roll_input_window(
                input_dyn=cur_input,
                hydro_pred=pred[:, 0],
                rain_next=rain_next,
            )

        return {"loss_rec": loss_rec / future_frames}