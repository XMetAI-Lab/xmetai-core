"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-04-28
Description: Spatio-Temporal Residual VQ-VAE with Physics Constraints (Phy-3D-RQ-VAE)
"""

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from xmetai.builder import MODELS
from xmetai.loss import charbonnier_loss
from .base_predictor import BasePredictor

try:
    from vector_quantize_pytorch import ResidualVQ
except ImportError:
    raise ImportError("Please install vector-quantize-pytorch: pip install vector-quantize-pytorch")


@MODELS.register()
class Phy3DRQVAE(BasePredictor):
    def __init__(
            self,
            embed_dim=256,  # 隐空间通道数 (Latent Dimension)
            num_quantizers=2,  # 残差量化级数
            codebook_size=1024,  # 每一级词表大小
            commit_weight=1.0,  # VQ Commitment 损失权重
            geo_weight=0.05,  # 地转平衡物理损失权重
            idx_phi=None,  # 位势高度 (Geopotential) 在 in_chans 中的通道索引
            idx_u=None,  # U 风分量在 in_chans 中的通道索引
            idx_v=None,  # V 风分量在 in_chans 中的通道索引
            image_size=(121, 240),  # 空间分辨率
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if idx_phi is None:
            idx_phi = []
        if idx_u is None:
            idx_u = [23, 24]
        if idx_v is None:
            idx_v = [25, 26]
        self.logger = logging.getLogger(__name__)

        self.commit_weight = commit_weight
        self.geo_weight = geo_weight
        self.image_size = tuple(image_size)


        # 物理量索引
        self.idx_phi = idx_phi
        self.idx_u = idx_u
        self.idx_v = idx_v

        # -----------------------------------------------------------------
        # 1. 3D 时空编码器 (Encoder)
        # 将 [B, C, T=4, H=121, W=240] 压缩至 [B, embed_dim, 1, H/4, W/4]
        # -----------------------------------------------------------------
        self.encoder = nn.Sequential(
            # Layer 1: 时间维度下采样2倍，空间下采样2倍
            nn.Conv3d(self.in_chans, 128, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # Layer 2: 时间维度再下采样2倍 (T=4 -> 1)，空间下采样2倍
            nn.Conv3d(128, embed_dim, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(),
        )

        # -----------------------------------------------------------------
        # 2. 残差向量量化 (Residual VQ)
        # -----------------------------------------------------------------
        self.rvq = ResidualVQ(
            dim=embed_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decay=0.99,  # EMA 更新机制，确保极度稳定
            kmeans_init=True,  # K-means 聚类初始化，防止早期词表死亡
            kmeans_iters=10,
            threshold_ema_dead_code=2  # 自动复活低频 Token
        )

        # -----------------------------------------------------------------
        # 3. 3D 时空解码器 (Decoder)
        # 将 [B, embed_dim, 1, H/4, W/4] 还原回 [B, C, T=4, H=121, W=240]
        # -----------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 128, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            # 终极修复：仅在最后一层加入 output_padding=(0, 1, 0)
            nn.ConvTranspose3d(128, self.in_chans, kernel_size=(2, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 0)),
        )

        # 预先生成纬度缓存用于物理损失 (假设范围 90 to -90, resolution=1.5)
        self.register_buffer("lat_map",
                             torch.linspace(90, -90, steps=self.image_size[0]).view(-1, 1).expand(self.image_size).clone())


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
                if isinstance(future_frames, (list, tuple, np.ndarray)):
                    future_frames = int(future_frames[0])
                targets = targets[:, :future_frames]
            batch["targets"] = targets

        # check if shift exist in data, do rolling for const
        if "shift" in data:
            shift = data["shift"]
            batch["shift"] = shift
            if const is not None:  # const is const_dimx721x1440
                # expand const to batch size and do rolling
                const = torch.repeat_interleave(const.unsqueeze(0), repeats=input.size(0), dim=0)
                batch["const"] = torch.stack(
                    [torch.roll(const[i], shifts=int(s), dims=-1) for i, s in enumerate(shift)])
            if hasattr(self, "mask"):
                self.mask_shift = torch.stack(
                    [torch.roll(self.mask, shifts=int(s), dims=-1) for i, s in enumerate(shift)])
                self.mask_shift = self.mask_shift.unsqueeze(1).unsqueeze(2)
            if hasattr(self, "land_mask"):
                self.land_mask_shift = torch.stack(
                    [torch.roll(self.land_mask, shifts=int(s), dims=-1) for i, s in enumerate(shift)])
                self.land_mask_shift = self.land_mask_shift.unsqueeze(1).unsqueeze(2)
        else:
            if hasattr(self, "mask"):
                self.mask_shift = self.mask
            if hasattr(self, "land_mask"):
                self.land_mask_shift = self.land_mask

        return batch

    def forward(self, data):
        """
        在 xmetai 框架下，forward 只需要负责输出包含各个 loss 值的字典。
        """

        if not self.training:
            return self.test(data)

        # 1. 解析并拼接数据
        data = self.process_data(data)
        # 拼接 input 和 target: [B, T_in + T_out, C, H, W]
        seq_whole = torch.cat([data["input"][:, -self.in_frames:], data["targets"]], dim=1)
        B, T, C, H, W = seq_whole.shape

        # 2. 空间插值 (Resize)
        # 将 [B, T, C, H, W] 合并为 [B*T, C, H, W] 进行双线性插值
        seq_resized = F.interpolate(
            rearrange(seq_whole, 'b t c h w -> (b t) c h w'),
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # 3. 转换为 Conv3D 形状: [B, C, T, H, W]
        # 直接从 (B*T) 还原并重排维度，一步到位
        x = rearrange(seq_resized, '(b t) c h w -> b c t h w', b=B, t=T)

        # 2. 编码 (Encoder)
        z_e = self.encoder(x)  # Shape: [B, embed_dim, 1, h, w]
        b, c, t_d, h, w = z_e.shape

        # 3. 量化 (Quantization)
        # RVQ 期望输入形状为 [B, N, C]
        z_e_flat = rearrange(z_e, 'b c t h w -> b (t h w) c')
        z_q_flat, indices, commit_loss = self.rvq(z_e_flat)
        z_q = rearrange(z_q_flat, 'b (t h w) c -> b c t h w', t=t_d, h=h, w=w)

        # 4. 解码 (Decoder)
        x_recon = self.decoder(z_q)  # Shape: [B, C, T, H, W]

        # 5. 损失计算
        loss_dict = {}

        # 5.1 重建损失 (使用框架自带的 Charbonnier Loss 或 MSE)
        loss_rec = charbonnier_loss(x_recon, x)
        loss_dict["loss_rec"] = loss_rec

        # 5.2 向量量化承诺损失 (Commitment Loss)
        # rvq 返回的 commit_loss 已经是各级的和，我们取 batch 的均值
        loss_dict["loss_commit"] = commit_loss.mean() * self.commit_weight

        # 5.3 物理约束损失 (Geostrophic Balance Loss)
        if self.geo_weight > 0:
            loss_geo = self.compute_geostrophic_loss(x_recon)
            loss_dict["loss_geo"] = loss_geo * self.geo_weight

        # Debug 日志: 监控词表使用率
        if self.training:
            unique_tokens = len(torch.unique(indices))
            self.logger.debug(f"Codebook active tokens: {unique_tokens}")

        return loss_dict

    def forward_visual(self, data):
        """
        在 xmetai 框架下，forward 只需要负责输出包含各个 loss 值的字典。
        """

        # 1. 解析并拼接数据
        data = self.process_data(data)
        # 拼接 input 和 target: [B, T_in + T_out, C, H, W]
        seq_whole = torch.cat([data["input"][:, -self.in_frames:], data["targets"]], dim=1)
        B, T, C, H, W = seq_whole.shape

        # 2. 空间插值 (Resize)
        # 将 [B, T, C, H, W] 合并为 [B*T, C, H, W] 进行双线性插值
        seq_resized = F.interpolate(
            rearrange(seq_whole, 'b t c h w -> (b t) c h w'),
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # 3. 转换为 Conv3D 形状: [B, C, T, H, W]
        # 直接从 (B*T) 还原并重排维度，一步到位
        x = rearrange(seq_resized, '(b t) c h w -> b c t h w', b=B, t=T)

        # 2. 编码 (Encoder)
        z_e = self.encoder(x)  # Shape: [B, embed_dim, 1, h, w]
        b, c, t_d, h, w = z_e.shape

        # 3. 量化 (Quantization)
        # RVQ 期望输入形状为 [B, N, C]
        z_e_flat = rearrange(z_e, 'b c t h w -> b (t h w) c')
        z_q_flat, indices, commit_loss = self.rvq(z_e_flat)
        z_q = rearrange(z_q_flat, 'b (t h w) c -> b c t h w', t=t_d, h=h, w=w)

        # 4. 解码 (Decoder)
        x_recon = self.decoder(z_q)  # Shape: [B, C, T, H, W]

        return_dict = {}
        return_dict["x_recon"] = x_recon
        return_dict["x"] = x
        return return_dict

    def compute_geostrophic_loss(self, x_recon):
        """
        计算地转平衡物理约束损失
        x_recon shape: [B, C, T, H, W]
        """
        phi = x_recon[:, self.idx_phi, :, :, :]
        u = x_recon[:, self.idx_u, :, :, :]
        v = x_recon[:, self.idx_v, :, :, :]

        # 科里奥利参数 f = 2 * Omega * sin(lat)
        omega = 7.2921e-5
        lat_rad = torch.deg2rad(self.lat_map)
        f = 2 * omega * torch.sin(lat_rad)
        f = f.view(1, 1, self.image_size[0], self.image_size[1])  # [1, 1, H, W]

        # 截断防止赤道除以 0，直接用极小值替换，反正稍后会被 mask 掉
        f_safe = f.clone()
        f_safe[torch.abs(f_safe) < 1e-5] = 1e-5

        # 空间差分计算
        # 假设分辨率为 1.5° (dy 固定，dx 随纬度变化)
        dy = 1.5 * 111000.0  # 约 166.5 km
        dx = dy * torch.cos(lat_rad).view(1, 1, self.image_size[0], self.image_size[1])
        dx = torch.clamp(dx, min=1000.0)  # 极点保护

        # 使用有限差分计算位势梯度的经向和纬向导数
        # dim=-2 为纬度 (y), dim=-1 为经度 (x)
        dphi_dy, dphi_dx = torch.gradient(phi, dim=(-2, -1))

        # 将差分除以实际物理距离
        dphi_dy = dphi_dy / dy
        dphi_dx = dphi_dx / dx

        # 理想地转风
        u_g = -(1.0 / f_safe) * dphi_dy
        v_g = (1.0 / f_safe) * dphi_dx

        # 掩码：屏蔽赤道区域 (纬度 [-15, 15] 之间不满足地转平衡)
        equator_mask = (torch.abs(self.lat_map) > 15.0).float().view(1, 1, self.image_size[0], self.image_size[1])

        loss_u = F.mse_loss(u * equator_mask, u_g * equator_mask)
        loss_v = F.mse_loss(v * equator_mask, v_g * equator_mask)

        return loss_u + loss_v

    @torch.no_grad()
    def encode_to_tokens(self, x_chunk):
        """
        供 Phase 2 (Transformer) 调用的离线 Token 化接口
        x_chunk: [B, T, C, H, W]
        返回: 离散 Token 索引 indices [B, num_quantizers, t_d * h * w]
        """
        x = rearrange(x_chunk, 'b t c h w -> b c t h w')
        z_e = self.encoder(x)
        z_e_flat = rearrange(z_e, 'b c t h w -> b (t h w) c')
        _, indices, _ = self.rvq(z_e_flat)
        return indices

    @torch.no_grad()
    def decode_tokens(self, indices):
        """
        供推断阶段调用：将离散 Token 还原为连续物理场
        indices: [B, N, L]
        返回: x_recon [B, T, C, H, W]
        """
        z_q_flat = self.rvq.get_output_from_indices(indices)

        # 根据你的 encoder 结构 (两层 stride=2)，空间分辨率被下采样 4 倍
        # H=121 -> 60 -> 30, W=240 -> 120 -> 60
        h = 30
        w = 60

        # 动态计算时间维度 t_d (7200 // 1800 = 4)
        N = z_q_flat.shape[1]
        t_d = N // (h * w)

        z_q = rearrange(z_q_flat, 'b (t h w) c -> b c t h w', t=t_d, h=h, w=w)

        x_recon = self.decoder(z_q)
        x_recon = rearrange(x_recon, 'b c t h w -> b t c h w')

        return x_recon

    def test_cascade(self, data):
        """
        推断阶段：验证 VAE 对未来目标场的重建能力 (UpperBound Test)。
        仅提取 targets，执行完整的 Encode -> Quantize -> Decode 纯重建过程。

        Returns:
            outputs: [B, T, C, H, W] 张量
        """
        # 1. 提取 target 数据
        targets = data["targets"]
        B, T, C, H, W = targets.shape

        # 2. 空间插值 (Resize) 适配模型分辨率
        # 将 [B, T, C, H, W] 展平为 [B*T, C, H, W] 进行双线性插值
        targets_resized = F.interpolate(
            rearrange(targets, 'b t c h w -> (b t) c h w'),
            size=self.image_size,
            mode='bilinear',
            align_corners=False
        )

        # 3. 转换为 Conv3D 形状: [B, C, T, H, W]
        x = rearrange(targets_resized, '(b t) c h w -> b c t h w', b=B, t=T)

        # 4. 关闭梯度，纯净前向重建
        with torch.no_grad():
            # 4.1 编码 (Encoder)
            z_e = self.encoder(x)  # Shape: [B, embed_dim, t_d, h, w]
            b, c, t_d, h_d, w_d = z_e.shape

            # 4.2 量化 (Quantization)
            # 测试阶段不需要 commit_loss，只需要拿回离散化重组后的连续特征 z_q
            z_e_flat = rearrange(z_e, 'b c t h w -> b (t h w) c')
            z_q_flat, indices, _ = self.rvq(z_e_flat)
            z_q = rearrange(z_q_flat, 'b (t h w) c -> b c t h w', t=t_d, h=h_d, w=w_d)

            # 4.3 解码 (Decoder)
            x_recon = self.decoder(z_q)  # Shape: [B, C, T, H, W]

        # 5. 转回框架标准的输出格式 [B, T, C, H, W]
        outputs = rearrange(x_recon, 'b c t h w -> b t c h w')

        return outputs