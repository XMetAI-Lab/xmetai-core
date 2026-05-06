"""
Author: DouZesheng && douzsh@gmail.com
Date: 2026-04-29
Description: Masked Regime Transformer for S2S discrete token forecasting.
Copyright (c) 2026 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from xmetai.core import FSDPCheckpointer

from xmetai.config import instantiate

from xmetai.builder import MODELS
from .base_predictor import BasePredictor


@MODELS.register()
class PhyCast(BasePredictor):
    def __init__(
            self,
            embed_dim=512,
            depths=12,
            num_heads=16,
            token_spatial_size=(15, 30),  # 假设 RQ-VAE 降采样后的 Token 空间分辨率 (H', W')
            drop_rate=0.1,
            mlp_ratio=4,
            mask_schedule="cosine",  # 掩码调度策略
            config_path="configs/s2s_phyVAE.py",  # vae 相关参数
            vocab_size=1024,
            num_quantizers=2,  # <--- 【新增】传入量化层数
            const_chans=0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_spatial_size = tuple(token_spatial_size)
        self.mask_schedule = mask_schedule
        self.const_chans = const_chans
        self.num_quantizers = num_quantizers  # <--- 【新增】保存属性
        self.logger = logging.getLogger(__name__)

        self.maskgit_steps = 8
        self.temperature = 1.0

        # 定义特殊的 [MASK] Token ID
        self.mask_token_id = vocab_size

        # 1. 离散 Token 嵌入层 (词表大小 + 1 个 MASK token)
        self.token_embed = nn.Embedding(self.vocab_size + 1, embed_dim)

        # 2. 时空位置编码 (Learnable 1D/3D absolute pos embedding)
        # 序列总长度: (in_frames + out_frames) * H' * W'
        self.total_seq_len = (self.in_frames + self.out_frames) * np.prod(self.token_spatial_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. 双向 Transformer 编码器 (非自回归)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depths)

        # 4. 交叉熵分类头
        self.head = nn.Linear(embed_dim, self.num_quantizers * self.vocab_size)
        self.rqvae = self._build_and_load_vae(config_path)

    def _build_and_load_vae(self, config_path):
        """
        全权构建+加载+冻结 Phy-3D-RQ-VAE 预训练编码器
        Returns:
            frozen rqvae model in eval mode
        """
        from xmetai import LazyConfig

        self.logger.info("Loading pre-trained Phy-3D-RQ-VAE as frozen tokenizer...")
        cfg_vae = LazyConfig.load(config_path)
        ckpt_path = cfg_vae.train.init_checkpoint

        # 1. 依据配置完整构建模型
        rqvae = instantiate(cfg_vae.model)

        # 2. FSDP 安全加载预训练权重
        try:
            FSDPCheckpointer(rqvae).load(ckpt_path)
        except Exception as e:
            self.logger.error(f"VAE checkpoint load failed: {str(e)}")
            raise

        # 3. 固定为推理模式、冻结全部梯度
        rqvae.eval()
        for param in rqvae.parameters():
            param.requires_grad = False

        # 4. 自动匹配训练精度
        if getattr(cfg_vae.model, "precision", None) == "bf16":
            rqvae = rqvae.to(torch.bfloat16)

        self.logger.info("Phy-3D-RQ-VAE loaded, frozen and ready.")
        return rqvae

    def compute_mask_ratio(self, step, total_steps):
        """计算掩码比例，推荐使用 cosine 调度以重点训练难样本"""
        ratio = step / total_steps
        if self.mask_schedule == "cosine":
            return math.cos(ratio * math.pi / 2)
        elif self.mask_schedule == "linear":
            return 1.0 - ratio
        return 1.0 - ratio  # default linear

    def forward(self, data):
        if not self.training:
            return self.test(data)

        data = self.process_data(data)
        input = data["input"]  # 连续历史场: bxT_inxCxhxw
        times = data["times"]
        targets = data["targets"]  # 连续未来场: bxT_outxCxhxw
        const = data.get("const", None) if self.const_chans > 0 else None

        self.logger.debug(f"Continuous input shape: {input.shape}, target shape: {targets.shape}")
        self.logger.debug(
            f"Input data min: {input.min():.3f}, max: {input.max():.3f}, target min: {targets.min():.3f}, max: {targets.max():.3f}")

        # =========================================================================
        # 1. 在线离散化 (Online Tokenization) - 冻结 VAE 的梯度
        # =========================================================================
        with torch.no_grad():
            self.rqvae.eval()  # 必须保证 VAE 处于 eval 模式
            # 调用你的 VAE 提取离散 Tokens
            # 假设返回的 tokens 形状为 [batch_size, time_steps, h_prime, w_prime]
            past_tokens = self.rqvae.encode_to_tokens(input)
            future_tokens = self.rqvae.encode_to_tokens(targets)

        self.logger.debug(
            f"Discrete past tokens shape: {past_tokens.shape}, future tokens shape: {future_tokens.shape}")

        # =========================================================================
        # 2. MaskGIT 并行掩码前向传播
        # =========================================================================
        # 我们不需要像 ViTModel 那样写 for t in range(...)，而是将整个序列送入
        output_logits, mask = self.forward_step(past_tokens, future_tokens, const=const, times=times)

        # =========================================================================
        # 3. 计算仅作用于被掩码位置的交叉熵损失
        # =========================================================================
        step_losses = self.compute_step_loss(output_logits, future_tokens, mask)

        loss_dict = step_losses  # MaskGIT 是一次性计算总损失，无需时间维度的累加平均

        self.logger.debug(f"Total CE Loss: {loss_dict.get('loss_ce', 0):.4f}")
        return loss_dict

    def compute_step_loss(self, logits, targets, mask):
        """
        计算交叉熵损失。
        logits: [B, N_future, L, vocab_size]
        targets: [B, N_future, L] (因为输入 targets 已经是展平的 B N L)
        mask: [B, N_future]
        """
        # 因为 targets 已经是展平的，不需要 rearrange 了！
        targets_flat = targets

        # 提取被 Mask 的位置:
        # logits_masked: [num_masked_items, L, vocab_size]
        # targets_masked: [num_masked_items, L]
        logits_masked = logits[mask]
        targets_masked = targets_flat[mask]

        if targets_masked.numel() > 0:
            # CrossEntropy 接受的输入需要是 (N, C) 格式
            # 我们把被 Mask 的数量 (m) 和层级维度 (l) 合并，得到真正的样本数
            loss_ce = F.cross_entropy(
                rearrange(logits_masked, 'm l v -> (m l) v'),
                rearrange(targets_masked, 'm l -> (m l)')
            )
        else:
            loss_ce = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return {"loss_ce": loss_ce}
    def forward_step(self, past_tokens, future_tokens, const=None, times=None, timestamp=None):
        """
        适配 Phy-3D-RQ-VAE 的多级残差量化 MaskGIT 前向过程
        past_tokens: [B, N_past, L]  (N_past 是历史展平后的 Token 数，L 为量化层数)
        future_tokens: [B, N_future, L] (N_future 是未来展平后的 Token 数)
        """
        batch_size = past_tokens.shape[0]
        L = past_tokens.shape[-1]  # 获取残差量化层数 (例如 num_quantizers=2)

        # 1. 因为输入已经是展平的 B N L 格式，直接赋值，省去 rearrange
        past_seq = past_tokens
        future_seq = future_tokens
        N_past = past_seq.shape[1]
        N_future = future_seq.shape[1]

        # 2. 生成动态掩码 (基于空间位置的统一掩码)
        # 掩码维度为 [B, N_future]，确保同一个位置的 L 个 Token 被同时掩蔽
        if self.training:
            r = torch.rand(1, device=past_tokens.device).item()
            mask_ratio = self.compute_mask_ratio(r, 1.0)
            # mask 针对的是时空位置 N_future，而非具体的 token 层级
            mask = torch.rand(batch_size, N_future, device=future_tokens.device) < mask_ratio
        else:
            mask = torch.ones(batch_size, N_future, dtype=torch.bool, device=future_tokens.device)

        # 将 mask 扩展到 L 维度以匹配 future_seq: [B, N_future, L]
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, L)

        # 3. 将未来序列中的目标替换为 [MASK] (mask_token_id)
        masked_future_seq = torch.where(mask_expanded, self.mask_token_id, future_seq)

        # 4. 拼接上下文与掩码目标: [B, N_past + N_future, L]
        input_seq = torch.cat([past_seq, masked_future_seq], dim=1)

        # ---------- 5. 适配 RVQ 的 Token Embedding ----------
        # 分别对每个层级做 Embedding，然后相加 (等价于 VAE 解码时的 Z_q 累加)
        h = 0
        for l in range(L):
            # 取出第 l 级的 token 进行 embedding
            h += self.token_embed(input_seq[:, :, l])

        # 注入位置编码 (pos_embed 的 shape 需要是 [1, N_past + N_future, embed_dim])
        h = h + self.pos_embed[:, :input_seq.shape[1], :]

        # 6. Transformer 主干前向传播
        h = self.transformer(h)

        # 7. 交叉熵分类头映射 (Logits)
        # 取出属于 Future 部分的特征进行分类
        future_features = h[:, N_past:, :]  # [B, N_future, embed_dim]

        # Head 输出 L 个独立分布: Linear(embed_dim, L * vocab_size)
        logits = self.head(future_features)

        # 重塑为 [B, N_future, L, vocab_size] 供算 Loss
        logits = rearrange(logits, 'b n (l v) -> b n l v', l=L, v=self.vocab_size)

        # 返回 logits 和 原始的空间掩码 (mask) 用于计算损失
        return logits, mask
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

    def _maskgit_decode(self, past_tokens, const=None, times=None):
        """
        MaskGIT 核心迭代解码逻辑 (内部辅助函数)
        对应论文中的 Algorithm 1: Parallel Decoding with Iterative Refinement

        输入:
            past_tokens: [B, N_past, L]  历史离散 Tokens
        输出:
            future_tokens: [B, N_future, L]  生成的未来离散 Tokens
        """
        batch_size = past_tokens.shape[0]
        device = past_tokens.device
        L = self.num_quantizers

        # 1. 计算序列长度 (基于 token_spatial_size)
        N_past = self.in_frames * np.prod(self.token_spatial_size)
        N_future = self.out_frames * np.prod(self.token_spatial_size)

        # 2. 初始化：未来序列全部填充为 [MASK]
        future_tokens = torch.full(
            (batch_size, N_future, L),
            self.mask_token_id,
            device=device,
            dtype=torch.long
        )
        # mask: [B, N_future], True 表示该位置尚未预测
        mask = torch.ones(batch_size, N_future, dtype=torch.bool, device=device)

        # 3. 迭代解码循环
        for step in range(self.maskgit_steps):
            # --- 3.1 构建当前输入序列 ---
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, L)
            # 仅在 mask 为 True 的地方保持 MASK，其他地方用已预测的 token
            masked_future_seq = torch.where(mask_expanded, self.mask_token_id, future_tokens)
            input_seq = torch.cat([past_tokens, masked_future_seq], dim=1)

            # --- 3.2 前向传播 (复用 Embedding 逻辑) ---
            h = 0
            for l in range(L):
                h += self.token_embed(input_seq[:, :, l])

            # 注入位置编码 (截取当前需要的长度)
            h = h + self.pos_embed[:, :input_seq.shape[1], :]

            # Transformer 前向
            h = self.transformer(h)

            # 提取 Future 部分的特征
            future_features = h[:, N_past:, :]  # [B, N_future, embed_dim]

            # --- 3.3 预测 Logits 与采样 ---
            logits = self.head(future_features)
            logits = rearrange(logits, 'b n (l v) -> b n l v', l=L, v=self.vocab_size)

            # 应用温度采样 (如果 temperature != 1.0)
            if self.temperature != 1.0:
                probs = F.softmax(logits / self.temperature, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)

            # 贪婪采样 (Greedy Sampling)
            pred_tokens = torch.argmax(probs, dim=-1)  # [B, N_future, L]

            # --- 3.4 计算置信度 (Confidence) ---
            # 取 L 层量化的平均置信度作为该时空位置的置信度
            confidence = torch.max(probs, dim=-1)[0]  # [B, N_future, L]
            confidence = confidence.mean(dim=-1)  # [B, N_future]

            # 仅在当前仍被 Mask 的位置计算置信度，其他位置设为 -inf
            confidence_masked = torch.where(mask, confidence, -torch.inf)

            # --- 3.5 确定本次迭代要揭示 (Unmask) 的 Token 数量 ---
            # 线性调度策略：随着 step 增加，剩余 mask 比例减少
            ratio = (step + 1) / self.maskgit_steps
            # 计算当前应该总共揭示的数量
            total_revealed = int(ratio * N_future)
            # 计算上一轮结束时已揭示的数量
            prev_revealed = int(step / self.maskgit_steps * N_future)
            # 本次需要揭示的数量
            num_unmask = total_revealed - prev_revealed
            num_unmask = max(num_unmask, 1) if mask.sum() > 0 else 0

            # --- 3.6 执行 Unmask ---
            if num_unmask > 0:
                # 对 Batch 中每个样本单独处理 (防止不同样本 mask 数量不一致)
                for b in range(batch_size):
                    # 当前样本剩余的 mask 数量
                    remaining_mask = mask[b].sum().item()
                    if remaining_mask == 0:
                        continue

                    # 选取置信度最高的 k 个位置
                    k = min(num_unmask, remaining_mask)
                    # topk 返回 (values, indices)
                    _, topk_indices = torch.topk(confidence_masked[b], k=k)

                    # 更新 mask: 将这些位置设为 False (已揭示)
                    mask[b, topk_indices] = False
                    # 更新 future_tokens: 填入预测值
                    future_tokens[b, topk_indices] = pred_tokens[b, topk_indices]

            # 早停机制：如果没有 mask 了，提前退出
            if not mask.any():
                break

        return future_tokens

    def test_cascade(self, data):
        """
        推断阶段：MaskGIT 迭代解码 + S2S 滚动预测 (Cascade Rollout)
        支持生成超出自.out_frames的长序列，通过不断将预测结果回喂作为输入。

        Returns:
            outputs: [B, T, C, H, W] 张量，其中 T 为 max(self.test_frames) 或 self.out_frames
        """
        # 1. 初始化与数据预处理
        inputs = data["input"]  # (B, in_frames, V, H_orig, W_orig)
        times = data["times"]
        const = data.get("const", None) if self.const_chans > 0 else None

        # 2. 确定总预测步数
        if hasattr(self, 'test_frames') and self.test_frames:
            total_step = max(self.test_frames)
        else:
            total_step = self.out_frames

        # 计算需要滚动多少轮
        num_rollout_steps = int(np.ceil(total_step / self.out_frames))

        current_input = inputs.clone()
        all_predicted_frames = []

        # 3. 滚动预测循环
        for rollout_idx in range(num_rollout_steps):
            self.logger.debug(f"Cascade rollout step {rollout_idx + 1}/{num_rollout_steps}")

            # --- 3.1 单步生成 ---
            with torch.no_grad():
                self.rqvae.eval()
                past_tokens = self.rqvae.encode_to_tokens(current_input)
                future_tokens = self._maskgit_decode(past_tokens, const=const, times=times)
                pred_frames = self.rqvae.decode_tokens(future_tokens)  # [B, T_out, C, H, W]

            # --- 3.2 保存结果 ---
            all_predicted_frames.append(pred_frames)

            # --- 3.3 更新下一轮的输入 Current Input ---
            combined = torch.cat([current_input, pred_frames], dim=1)
            current_input = combined[:, -self.in_frames:]

            # 早停检查
            concat_pred = torch.cat(all_predicted_frames, dim=1)
            if concat_pred.shape[1] >= total_step:
                break

        # 4. 最终拼接与切片，确保输出形状为 [B, T, C, H, W]
        final_concat = torch.cat(all_predicted_frames, dim=1)
        outputs = final_concat[:, :total_step]

        return outputs
