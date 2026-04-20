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

from .vit import ViTModel


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


def spread_skill_loss(outputs: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """
    Spread-Skill loss to explicitly promote ensemble reliability.
    Ideal ensembles should have their spread (variance) matching their RMSE (skill).
    Constraint: E[(pred_mean - target)^2] ≈ E[var(outputs)]
    """
    # Assuming standard all-gather is implicitly done in afcrps_lite,
    # but for manual spread matching we can approximate across the local batch
    # if DDP is not handling the member dimension directly here.
    # We will compute the deviation of the scale/gamma to prevent mode collapse.
    pass


def magnitude_penalty(gamma, beta, min_var=0.01, lambda_mag=0.1):
    """
    Replaces crude delta_h.std() with a specific penalty on the Modulation parameters.
    Prevents the MLP from setting gamma and beta strictly to 0.
    """
    # Encourage the spatial variation of gamma and beta
    var_gamma = gamma.var(dim=1).mean()
    var_beta = beta.var(dim=1).mean()
    loss = F.relu(min_var - var_gamma) + F.relu(min_var - var_beta)
    return lambda_mag * loss


class EnsNoiseMLP(nn.Module):
    def __init__(self, embed_dim, sample_dim=128, dropout=0.1):
        super().__init__()
        self.sample_dim = sample_dim
        # Input gets concatenated: noise (sample_dim) + h.detach() (embed_dim)
        in_features = sample_dim + embed_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            # Output gamma and beta (so 2 * embed_dim)
            nn.Linear(512, embed_dim * 2),
        )
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, noise, h):
        x = torch.cat([noise, h], dim=-1)
        output = self.net(x)
        output = self.drop_path(output)
        return output


@MODELS.register()
class ViTModelEns(ViTModel):
    def __init__(self, embed_dim=256, alpha=1.0, **kwargs):
        super().__init__(embed_dim=embed_dim, **kwargs)
        self.logger.name = __name__
        self.alpha = alpha
        self.sample_dim = kwargs.get("sample_dim", 128)
        self.ens_noise_mlp = EnsNoiseMLP(embed_dim, sample_dim=self.sample_dim)
        self.delta_h = None

        # Register a forward pre-hook on the middle layer to inject noise.
        # This completely delegates the main loop to the base ViTModel class.
        if hasattr(self, "encoder") and hasattr(self.encoder, "layers"):
            mid_layer = len(self.encoder.layers) // 2
            self.encoder.layers[mid_layer].register_forward_pre_hook(self._inject_noise_hook)

    def _inject_noise_hook(self, module, args):
        h = args[0]  # args is a tuple of (h,) according to standard pytorch
        # In base class, perturbations happen when self.training or self.use_pertub is True.
        if self.training or self.use_pertub:
            B, N, C = h.shape
            # 1. Macro / Global Noise:
            # Draw a single low-dimensional noise vector per batch item and dimension expand it.
            # This acts as a macroscopic uncertainty factor avoiding spatial high-frequency smoothing.
            noise = torch.randn(B, 1, self.sample_dim, device=h.device, dtype=h.dtype).expand(B, N, -1)
            # 2. State-Conditional injection
            # Give the detach()ed weather state to the MLP to let it know where it is
            self.delta_h = self.ens_noise_mlp(noise, h.detach())
            # 3. Scale-Shift (AdaLN/FiLM) Modulation
            gamma, beta = self.delta_h.chunk(2, dim=-1)

            # Store modulation factors for explicit regularization
            self.gamma_mod = gamma
            self.beta_mod = beta

            # Modulate the hidden state natively (Highly non-linear interaction)
            h = h * (1 + gamma) + beta
            return (h,)
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

        # Replace naive noise_magnitude_loss with explicit parameter variance penalty
        if getattr(self, "gamma_mod", None) is not None:
            ens_losses["loss_mag_pen"] = magnitude_penalty(self.gamma_mod, self.beta_mod) * w

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
