"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-11-03 08:05:44
Description: ViT model with ensemble strategy.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from xmetai.builder import MODELS
from xmetai.layers import DropPath
from xmetai.loss import afcrps_loss_lite, charbonnier_loss

from .vit import PerturbUMamba


def get_exponential_weights(steps: int, alpha: float) -> torch.Tensor:
    """
    Generate normalized exponential weights for sequence loss.

    Args:
        steps (int): number of timesteps (e.g., future frames).
        alpha (float): exponential factor (>1 = emphasize later, <1 = emphasize earlier).

    Returns:
        torch.Tensor: normalized weights summing to 1.
    """
    exponents = torch.arange(steps, dtype=torch.float32)
    raw = torch.pow(alpha, exponents)
    weights = raw / raw.sum()
    return weights


def noise_magnitude_loss(delta_h, min_std=0.1, lambda_mag=0.1):
    """Ensure that noise amplitude does not collapse."""
    std = delta_h.std(dim=-1)  # [B, pH*pW]
    loss = F.relu(min_std - std).pow(2).mean()
    return lambda_mag * loss


class EnsNoiseMLP(nn.Module):
    def __init__(self, embed_dim, init_scale=0.1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, embed_dim),
            nn.Softsign(),
        )
        # learnable global scaling factor
        self.noise_scale = nn.Parameter(torch.full((1,), init_scale))
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization 适合对称激活（Tanh/Softsign）
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, noise):
        output = self.net(noise) * self.noise_scale.view(1, 1, 1)
        output = self.drop_path(output)
        return output


@MODELS.register()
class PerturbUMambaEns(PerturbUMamba):
    def __init__(self, embed_dim=256, alpha=1.0, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.logger.name = __name__
        self.alpha = alpha
        self.ens_noise_mlp = EnsNoiseMLP(embed_dim)
        self.delta_h = None

        # Register a forward pre-hook on the middle layer to inject noise.
        # This completely delegates the main loop to the base PerturbUMamba class.
        if hasattr(self, "encoder") and hasattr(self.encoder, "layers"):
            mid_layer = len(self.encoder.layers) // 2
            self.encoder.layers[mid_layer].register_forward_pre_hook(self._inject_noise_hook)

    def _inject_noise_hook(self, module, args):
        h = args[0]  # args is a tuple of (h,) according to standard pytorch
        # In base class, perturbations happen when self.training or self.use_pertub is True.
        if self.training or self.use_pertub:
            noise = torch.randn_like(h)
            self.delta_h = self.ens_noise_mlp(noise)
            # Add delta_h on top of any mixup/dropout that might be present
            return (h + self.delta_h,)
        return args

    def forward(self, data):
        if not self.training:
            return self.test(data)

        # Pre-compute exponential rolling weights explicitly for the base AR loop
        targets = data.get("targets", None)
        if targets is not None:
            future_frames = targets.size(1)
            self.rolling_weights = get_exponential_weights(steps=future_frames, alpha=self.alpha).to(targets.device)
            # base class inherently divides final acc_loss by future_frames.
            # We compensate this during step computation.
            self.future_frames = future_frames

        # Complete dependency on the base autoregressive loop
        return super().forward(data)

    def compute_step_loss(self, output, target, step):
        # Base class calculates unweighted charbonnier and PINN losses
        base_losses = super().compute_step_loss(output, target, step)

        # Compensate for base class division since rolling weights sum to 1
        w = self.rolling_weights[step] * self.future_frames

        ens_losses = {}
        # Apply weights to base losses
        for k, v in base_losses.items():
            if k == "loss_rec" and "channel_weights" in getattr(self, "_buffers", {}):
                # Override charb specifically if channel_weights are applied differently for ENS
                channel_weight = getattr(self, "channel_weights") * getattr(self, "weight", 1.0)
                ens_losses[k] = charbonnier_loss(output, target, weight=channel_weight) * w
            else:
                ens_losses[k] = v * w

        # Add Ensemble-specific CRPS loss
        ens_losses["loss_crps"] = afcrps_loss_lite(output, target, weight=getattr(self, "weight", None)) * w

        # Add Noise Magnitude enforcement loss
        if getattr(self, "delta_h", None) is not None:
            ens_losses["loss_noise"] = noise_magnitude_loss(self.delta_h) * w

        return ens_losses

    @torch.no_grad()
    def test_cascade(self, data):
        self.logger.info(f"Start testing with {self.members} ensemble members and step_range {self.step_range} with time:{data.get('times', None)}.")

        # Enable perturbation logic to trigger the forward_pre_hook
        self.use_pertub = True

        original_input = data["input"].clone()
        outputs = 0
        for member in range(self.members):
            self.logger.debug(f"inference with member:{member} for step_range:{self.step_range}.")
            data["input"] = original_input.clone()
            # The base class cleanly handles single trajectory simulation and frame caching
            outputs += super().test_cascade(data)

        self.use_pertub = False
        data["input"] = original_input  # Restore untouched state

        outputs /= self.members  # ensemble prediction with mean
        return outputs
