#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: shenzhewen &&
Date: 2026-04-20 12:00:00
Description: TIANQUAN-S2S: A SUBSEASONAL-TO-SEASONAL GLOBAL WEATHER MODEL VIA INCORPORATE CLIMATOLOGY STATE

Channel layout (s2s 67-channel):
    PL vars: z,t,u,v,q  x 13 levels = 65 channels
    Surface vars:  = 2 channels

Copyright (c) 2026 by XMetAI/shenzhewen, All Rights Reserved.
"""

# ==============================================================================
# 1. 标准库
# ==============================================================================
from datetime import timedelta
from functools import lru_cache

# ==============================================================================
# 2. 第三方科学计算库
# ==============================================================================
import numpy as np
import pandas as pd

# ==============================================================================
# 3. PyTorch 及深度学习生态
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from einops import rearrange

from xmetai.layers import (
    FGA,
    SCA,
    DEConv,
    PEDadd_UViT,
    Perceiver3DDecoder,
    Perceiver3DEncoder,
)
from xmetai.layers.weight_init import vit_fill

# ==============================================================================
# 4. xmetai 框架
# ==============================================================================
from xmetai.loss import charbonnier_loss
from xmetai.registries import MODELS

from .base_predictor import BasePredictor


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def mse(pred, y, vars, lat=None, mask=None):
    """
    Mean Squared Error per variable.

    Args:
        pred (Tensor[B, V, H, W]): Predictions.
        y (Tensor[B, V, H, W]): Ground truth.
        vars (list[str]): Variable names.
        mask (Tensor or None): Optional mask of shape [H, W].
    Returns:
        dict: MSE per variable and overall loss.
    """
    error = (pred - y) ** 2
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = error[:, i].mean()
        if mask is not None:
            loss_dict["loss"] = (error.mean(dim=1) * mask).sum() / mask.sum()
        else:
            loss_dict["loss"] = error.mean(dim=1).mean()
    return loss_dict


def crps(y_pred, y_true, transform, vars, lat, clim, log_postfix=None, anomaly=None):
    """Continuous Ranked Probability Score per variable and overall.

    Pure PyTorch CRPS via the standard energy-score decomposition
    (Gneiting & Raftery, 2007):

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the ensemble and y is the
    observation.  Latitude-``cos`` weighting is applied before spatial
    averaging.

    Args:
        y_pred (Tensor[B, M, V, H, W]): Ensemble predictions.
        y_true (Tensor[B, V, H, W]): Observation / ground truth.
        transform: Optional transform applied to both pred and truth.
        vars (list[str]): Variable names for per-variable logging.
        lat (Tensor[H]): Latitude vector for area weighting.
        clim, log_postfix, anomaly: Unused; kept for API compatibility.
    """
    if transform is not None:
        y_pred = transform(y_pred)
        y_true = transform(y_true)

    B, M, V, H, W = y_pred.shape

    # Latitude weights: cos(lat), mean-normalised.
    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(lat, device=y_pred.device, dtype=y_pred.dtype)
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = (w_lat / w_lat.mean()).view(1, 1, -1, 1)  # (1, 1, H, 1)

    crps_dict = {}
    with torch.no_grad():
        # --- per-variable CRPS --------------------------------------------------
        for i, var in enumerate(vars):
            pred_i = y_pred[:, :, i]  # (B, M, H, W)
            true_i = y_true[:, i]  # (B, H, W)

            # Observation term: E|X - y|
            abs_err = torch.abs(pred_i - true_i.unsqueeze(1)).mean(dim=1)  # (B, H, W)

            # Pair term: 0.5 * E|X - X'| via sorted-gap formula (O(M log M))
            sorted_pred = pred_i.sort(dim=1).values  # (B, M, H, W)
            gaps = sorted_pred[:, 1:] - sorted_pred[:, :-1]  # (B, M-1, H, W)
            idx = torch.arange(1, M, device=pred_i.device, dtype=pred_i.dtype)
            pair_weights = idx * (M - idx)  # (M-1,)
            pair_weights = pair_weights.view(1, -1, 1, 1)
            pair = (gaps * pair_weights).sum(dim=1) / (M * (M - 1))  # (B, H, W)

            crps_raw = abs_err - pair  # (B, H, W)
            crps_dict[f"crps_{var}_{log_postfix}"] = (crps_raw * w_lat.squeeze(1)).mean().item()

        # --- overall CRPS (all variables jointly) ------------------------------
        # Reshape to (B, M, V*H*W) — treat spatial+vars as a single dim.
        flat_pred = y_pred.reshape(B, M, -1)  # (B, M, V*H*W)
        flat_true = y_true.reshape(B, -1)  # (B, V*H*W)

        abs_all = torch.abs(flat_pred - flat_true.unsqueeze(1)).mean(dim=1)
        sorted_all = flat_pred.sort(dim=1).values
        gaps_all = sorted_all[:, 1:] - sorted_all[:, :-1]
        weights_all = torch.arange(1, M, device=flat_pred.device, dtype=flat_pred.dtype)
        weights_all = weights_all * (M - weights_all)
        pair_all = (gaps_all * weights_all.view(1, -1, 1)).sum(dim=1) / (M * (M - 1))

        crps_all = (abs_all - pair_all).mean()  # scalar
        crps_dict["crps"] = crps_all.item()

    return crps_dict


@MODELS.register()
class TianQuan(BasePredictor):
    """
    ResCast sub-seasonal-to-seasonal forecasting model.

    Args:
        default_vars (list[str]): List of variable names used as inputs.
        root_dir (str): Directory to save outputs (e.g., predictions).
        img_size (tuple[int,int]): Spatial resolution of input (H, W).
        patch_size (int): Size of each patch for embedding.
        embed_dim (int): Dimension of embedding space.
        decoder_depth (int): Number of decoder layers.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        drop_path (float): DropPath rate.
        drop_rate (float): Dropout rate.
    """

    def __init__(
        self,
        default_vars,
        img_size=(128, 256),
        window_size=(8, 8),
        patch_size=2,
        embed_dim=384,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4,
        drop_path=0.1,
        drop_rate=0.1,
        const_chans=2,
        needed_ids=None,
        climate_zarr="/mnt/d/cla.zarr",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if needed_ids is None:
            needed_ids = []
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.default_vars = default_vars
        self.const_chans = const_chans

        # Define atmospheric pressure levels and variable lists
        self.levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        self.upper_variables = [
            "geopotential",
            "wind_speed",
            "temperature",
            "relative_humidity",
            "specific_humidity",
        ]
        self.surface_variables = ["2m_temperature", "wind_speed_10m"] if img_size == (128, 256) else ["2m_temperature", "10m_wind_speed"]
        self.static_vars = ["lsm", "orog"]

        self.needed_ids = needed_ids

        # climit mean
        self.climate_zarr = climate_zarr
        self.climate_mean = xr.open_zarr(self.climate_zarr, consolidated=True)  # (time, channel, h, w)

        # Data fusion and anomaly branches
        self.anomaly_decoder = DEConv(default_conv, len(default_vars), 3)
        self.data_decoder = SCA(default_conv, len(default_vars), 3)
        self.fusion_encoder = FGA(len(default_vars))

        # Perceiver encoder & decoder
        self.encoder = Perceiver3DEncoder(
            surf_vars=tuple(self.surface_variables),
            static_vars=tuple(self.static_vars),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            latent_levels=8,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=64,
            drop_rate=drop_rate,
            depth=4,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = Perceiver3DDecoder(
            surf_vars=tuple(self.surface_variables),
            atmos_vars=tuple(self.upper_variables),
            atmos_levels=tuple(self.levels),
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )

        # Variable embedding
        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # UViT backbone
        self.backbone = PEDadd_UViT(embed_dim=embed_dim, depth=decoder_depth)

        # Weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings and layer weights."""
        pos = get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        self.apply(vit_fill)

    def create_var_embedding(self, dim):
        """Create learnable embeddings for each variable."""
        emb = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        var_map = {v: i for i, v in enumerate(self.default_vars)}
        return emb, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        """Map variable names to their embedding indices."""
        ids = np.array([self.var_map[v] for v in vars])
        return torch.from_numpy(ids).to(device)

    def hours_to_dates(self, hours, start_date="1979-01-01T00:00:00"):
        """
        Convert forecast hours back to datetime objects.
        Args:
            hours (torch.Tensor): Hours since reference.
            start_date (str): ISO timestamp of reference.
        Returns:
            List[pd.Timestamp]
        """
        base = pd.Timestamp(start_date)
        deltas = pd.to_timedelta(hours.cpu().numpy(), unit="h")
        return (base + deltas).tolist()

    def forward_encoder(self, x, anomaly, hours, variables, lats, lons, region_info, lead_time, const):
        """
        Encode inputs via anomaly & data branches, fuse, Perceiver, and UViT backbone.
        Returns tuple of surface and atmospheric predictions.
        """
        b, t, v, h, w = x.size()
        # Merge time & variable dims
        x_flat = rearrange(x, "b t v h w -> (b t) v h w")
        anom_flat = rearrange(anomaly, "b t v h w -> (b t) v h w")

        a = self.anomaly_decoder(anom_flat)
        d = self.data_decoder(x_flat)
        fused = self.fusion_encoder(a, d)
        fused = rearrange(fused, "(b t) v h w -> b t v h w", t=t)

        # Split surface vs atmosphere
        atmo = fused[:, :, 2:].reshape(b, t, len(self.upper_variables), len(self.levels), h, w)
        surf = fused[:, :, :2].reshape(b, t, len(self.surface_variables), h, w)

        enc = self.encoder(
            surf,
            const.to(x.device),
            atmo,
            lead_time,
            lats,
            lons,
            self.hours_to_dates(hours),
        )
        patches = (
            self.encoder.latent_levels,
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
        )
        ub = self.backbone(enc, use_diffusion=False)
        surf_pred, atmo_pred = self.decoder(ub, lats, lons, patches, lead_time)
        return surf_pred, atmo_pred

    def _times_to_hours(self, times, start_date="1979-01-01T00:00:00"):
        """
        Convert times to hours since start_date for model input.
        Args:
            times: list / tensor / array of datetime or pd.Timestamp
        Returns:
            hours: torch.Tensor (int64 dtype)
        """
        base_time = pd.Timestamp(start_date)

        # 把每个时间转成和基准时间相差的小时数
        hours = []
        for t in times:
            dt = pd.Timestamp(t)
            delta_hours = (dt - base_time) // pd.Timedelta("1h")  # 取整小时
            hours.append(delta_hours)

        return torch.tensor(hours, dtype=torch.int64)

    def resize_input(self, input):
        import torch.nn.functional as F
        from einops import rearrange

        # 1. 记录原始的 batch (n) 和 time (t) 维度大小，方便后面还原
        n, t = input.shape[0], input.shape[1]

        # 2. 压平成 4 维以适配 interpolate
        x = rearrange(input, "n t c h w -> (n t) c h w")

        # 3. 将 ListConfig 强制转换为标准的整数元组
        target_size = tuple(int(i) for i in self.img_size)

        # 4. 用转换后的 target_size 进行对比和插值
        if x.shape[-2:] != target_size:
            x = F.interpolate(x.float(), size=target_size, mode="bilinear", align_corners=False).to(input.dtype)

        # 5. 核心修复：将形状从 4 维恢复回 5 维
        x = rearrange(x, "(n t) c h w -> n t c h w", n=n, t=t)

        return x

    def resize_output(self, output, input):
        """
        Interpolate output (B, C, H, W) to match input spatial dimensions (H2, W2).
        input shape is (B, T, C', H2, W2).
        """
        # 提取 input 的最后两个维度作为目标尺寸 (H2, W2)
        target_size = input.shape[-2:]

        # Interpolate if output spatial size differs from target
        if output.shape[-2:] != target_size:
            output_dtype = output.dtype
            output = F.interpolate(
                output.float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).to(dtype=output_dtype)

        return output

    def resize_const(self, const):
        if const is not None:
            target_size = tuple(int(i) for i in self.img_size)
            if const.dim() == 3:
                # 形状: (C, H, W)
                c_tmp = const.unsqueeze(0).float()
                c_tmp = F.interpolate(c_tmp, size=target_size, mode="bilinear", align_corners=False)
                resized_const = c_tmp.squeeze(0).to(const.dtype)
            elif const.dim() == 4:
                # 形状: (B, C, H, W)
                resized_const = F.interpolate(const.float(), size=target_size, mode="bilinear", align_corners=False).to(const.dtype)
            else:
                # 如果出现其他意外维度，也可以尝试直接走你自己的 resize_input 或者报错
                resized_const = const
        else:
            resized_const = None
        return resized_const

    def replace_year_with_reference(self, idx, year=1979):
        replaced = []
        for ts in idx:
            try:
                # 尝试直接替换年份
                new_ts = ts.replace(year=year)
            except ValueError:
                # 如果触发 ValueError，检查是否是闰年 2 月 29 日的越界问题
                if ts.month == 2 and ts.day == 29:
                    # 如果目标参考年份不是闰年，将其平移到 2月 28日
                    new_ts = ts.replace(year=year, day=28)
                else:
                    # 如果是其他原因导致的 ValueError，则正常向上抛出
                    raise
            replaced.append(new_ts)

        return replaced

    def forward_step(self, input, const=None, times=None, step=None, climate_tensor=None, timestamp=None):
        """
        Single-step forward pass compatible with XMetAI framework.
        Follows UTransformer conventions: data transform, feature forward, and residual output.
        """
        if hasattr(self, "logger"):
            self.logger.debug(f"Processing step: {step}, times: {times}")

        if hasattr(self, "in_frames"):
            input = input[:, -self.in_frames :]  # e.g. bxTxcxhxw or bxtxvxhxw

        batch_size = input.shape[0]
        # 保留你原始的通道数v，方便后面裁剪
        v = input.shape[2]
        resized_input = self.resize_input(input)
        # resize
        resize_const = self.resize_const(const)
        resized_climate_tensor = self.resize_input(climate_tensor)

        if hasattr(self, "training") and (self.training or getattr(self, "use_pertub", False)):
            if getattr(self, "pertub_mixup", 0) > 0 and batch_size > 1:
                alpha = torch.rand(1, device=resized_input.device) * self.pertub_mixup
                indices = torch.randperm(batch_size, device=resized_input.device)
                resized_input = resized_input * (1 - alpha) + resized_input[indices] * alpha
                if hasattr(self, "logger"):
                    self.logger.debug(f"Applied input mixup with alpha: {alpha.item():.3f}")

        # 获取 resize 之后的宽高
        H, W = resized_input.shape[-2], resized_input.shape[-1]

        anomaly = resized_climate_tensor

        hours = self._times_to_hours(times).to(self.device)
        lead_time = timedelta(hours=24)
        lats = torch.linspace(-90, 90, H).view(-1, 1).repeat(1, W).to(self.device)
        lons = torch.linspace(0, 360, W).view(1, -1).repeat(H, 1).to(self.device)
        region_info = {"min_h": 0, "max_h": H - 1, "min_w": 0, "max_w": W - 1}

        surf_pred, atmo_pred = self.forward_encoder(resized_input, anomaly, hours, self.default_vars, lats, lons, region_info, lead_time, const=resize_const)

        # 拼接并 reshape 输出
        output = torch.cat([surf_pred, atmo_pred.reshape(batch_size, -1, H, W)], dim=1)

        # 确保输出和输入变量数一致，补齐或者裁剪
        if output.shape[1] > v:
            output = output[:, :v]
        elif output.shape[1] < v:
            pad = torch.zeros(batch_size, v - output.shape[1], H, W, device=output.device)
            output = torch.cat([output, pad], dim=1)

        if hasattr(self, "logger"):
            self.logger.debug(f"head output shape (before resize): {output.shape}")

        # 调用你的方法将输出还原为原尺寸
        output = self.resize_output(output, input)

        if hasattr(self, "logger"):
            self.logger.debug(f"final output shape: {output.shape}, min: {output.min().item()}, max: {output.max().item()}")

        output = output.unsqueeze(1) + input[:, -1:]
        return output

    def compute_step_loss(self, output, target, step):
        if "channel_weights" in self._buffers:
            loss_rec = charbonnier_loss(output, target, weight=self.channel_weights * self.weight)
        else:
            loss_rec = charbonnier_loss(output, target, weight=self.weight)
        step_losses = {"loss_rec": loss_rec}
        # accumulate PINN losses per-step when enabled
        if self.pinn_enable:
            pinn_losses = self.compute_pinn_losses(output, target)
            for k, v in pinn_losses.items():
                step_losses[k] = v
        return step_losses

    def forward(self, data):
        """
        Standard framework entry point, receives a unified data dictionary.
        Combines custom spatial resizing with the framework's autoregressive multi-step training logic.
        """
        # 1. 测试/推理模式
        if not self.training:
            return self.test(data)

        # 2. 训练模式，预处理数据解包
        data = self.process_data(data)
        input = data["input"]
        times = data["times"]
        targets = data["targets"]
        const = data.get("const", None) if self.const_chans > 0 else None

        # 3. 规范的日志记录
        self.logger.debug(f"input shape: {input.shape}, target shape: {targets.shape}")
        self.logger.debug(f"input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        # 4. 准备自回归循环变量
        future_frames = targets.size(1)
        total_frames = self.start_frames + future_frames
        loss_dict = {}

        # 5. 严格仿照的自回归多步预测循环
        for t in range(self.start_frames, total_frames):
            k = t - self.start_frames
            # 切片获取当前步的 target
            target = targets[:, k : k + self.out_frames]

            self.reset_input(input)
            # 读取 climate_tensor
            curr_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=k * self.freq) for t in times], year=1979)
            pred_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=(k + 1) * self.freq) for t in times], year=1979)

            curr_climate_tensor = torch.stack(
                [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(input, non_blocking=True) for idx in curr_climate_time],
                dim=0,
            )

            pred_climate_tensor = torch.stack(
                [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(input, non_blocking=True) for idx in pred_climate_time],
                dim=0,
            )  # (bxchannelxhxw)

            climate_tensor = torch.stack([curr_climate_tensor, pred_climate_tensor], dim=1)

            # 单步预测 (注意这里严格传递了 step=t，要求你的 forward_step 支持该 kwargs)
            output = self.forward_step(input, const=const, times=times, step=t, climate_tensor=climate_tensor)

            # 计算当前步的 Loss
            step_losses = self.compute_step_loss(output, target, step=k)

            # 累加 Loss
            for k_loss, v_loss in step_losses.items():
                loss_dict[k_loss] = loss_dict.get(k_loss, 0) + v_loss

            self.logger.debug(f"step {k} loss_rec: {step_losses.get('loss_rec', 0)} with future frames {future_frames}.")

            # 准备下一步输入 (将当前 output 拼接到 input 后面，并截取最新的 in_frames 帧)
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]

        # 6. Loss 平均并返回
        for k_loss, v_loss in loss_dict.items():
            loss_dict[k_loss] = v_loss / future_frames

        return loss_dict

    @torch.no_grad()
    def test_cascade(self, data):
        """
        Autoregressive inference / test entry point.
        Handles dynamic climate_tensor loading from forward,
        and relies on forward_step's internal resize for spatial resolution.
        """
        # 1. 提取基础数据与配置
        # 注意：不再在外部进行插值，而是保留原始尺寸传入 forward_step
        inputs = data["input"]  # (B, in_frames, V, H_orig, W_orig)
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        B, F_in, V, H_orig, W_orig = inputs.shape

        # 2. 初始化输出容器
        total_step = max(self.test_frames)
        steps_to_save = np.array(self.test_frames) // self.out_frames

        # 按照原始尺寸返回，以便框架计算指标
        outputs = inputs.new_zeros(B, len(steps_to_save), V, H_orig, W_orig)

        # 3. 级联预测逻辑
        # step_range 通常为 [1]，如果存在多轮级联扰动等则累加取平均
        for num_step in self.step_range:
            curr_step_idx = 0
            # 复制一份原始输入用于本轮滚动预测
            rolling_inputs = inputs.clone()

            for step in range(total_step):
                # ==========================================
                # A. 仿照 forward 构造动态 climate_tensor
                # ==========================================
                curr_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=step * self.freq) for t in times], year=1979)
                pred_climate_time = self.replace_year_with_reference([t + pd.Timedelta(hours=(step + 1) * self.freq) for t in times], year=1979)

                # 从 zarr/dataset 中取出对应时间的气候态并转为 tensor (bxchannelxhxw)
                curr_climate_tensor = torch.stack(
                    [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(inputs, non_blocking=True) for idx in curr_climate_time],
                    dim=0,
                )
                pred_climate_tensor = torch.stack(
                    [torch.from_numpy(self.climate_mean.sel(time=idx).data.values[self.needed_ids]).to(inputs, non_blocking=True) for idx in pred_climate_time],
                    dim=0,
                )

                # (B, 2, C, H, W)
                climate_tensor = torch.stack([curr_climate_tensor, pred_climate_tensor], dim=1)

                # ==========================================
                # B. 执行单步前向推理
                # ==========================================
                # forward_step 内部已包含 self.resize_input 和 self.resize_output
                # 因此输出的 output 尺寸直接是 (B, 1, V, H_orig, W_orig)
                output = self.forward_step(rolling_inputs, const=const, times=times, step=step, climate_tensor=climate_tensor)

                # ==========================================
                # C. 滚动更新输入窗口
                # ==========================================
                # 拼接最新预测帧，并根据 in_frames 截取最新的历史数据
                rolling_inputs = torch.cat([rolling_inputs, output], dim=1)[:, -self.in_frames :]

                # ==========================================
                # D. 按需记录结果
                # ==========================================
                if (step + 1) in steps_to_save:
                    # 此时的 output 已经是残差相加且还原尺寸后的张量，直接累加即可
                    outputs[:, curr_step_idx : curr_step_idx + 1] += output
                    curr_step_idx += 1

        # 4. 如果有多个 step_range (例如 multi-ensemble / 扰动测试)，取平均值
        outputs = outputs / len(self.step_range)

        return outputs

    @torch.no_grad()
    def export_onnx(self, inputs, step, hour, doy):
        raise NotImplementedError(
            "TianQuan ONNX export is not supported yet. " "The current forward path depends on climatology tensors and datetime-aligned inputs."
        )
