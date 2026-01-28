"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-10 10:02:24
Description: Losses for KL Divergence and Entropy Regularization.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

__all__ = ["kl_loss", "kl_entropy_loss"]


def compute_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl


def kl_loss(mean1, logvar1, mean2, logvar2):
    kl = compute_kl(mean1, logvar1, mean2, logvar2)
    loss = torch.mean(kl.flatten(1).sum(-1))
    return loss


def kl_entropy_loss(dist_p, dist_q, w=1e-4, entropy_weight=1e-5, min_std=1e-3, target_std=0.1):
    # 1. Apply soft lower bound to stddev to avoid collapse
    std_p = torch.clamp(dist_p.stddev, min=min_std)
    std_q = torch.clamp(dist_q.stddev, min=min_std)

    # 2. Replace distributions with stabilized versions
    dist_p = torch.distributions.Normal(dist_p.mean, std_p)
    dist_q = torch.distributions.Normal(dist_q.mean, std_q)

    # 3. KL divergence
    kl = kl_divergence(dist_q, dist_p)  # KL(q || p)
    if kl.ndim > 2:
        kl = kl.flatten(1)
    loss = kl.sum(dim=-1).mean()

    # 4. Entropy regularization
    entropy_p = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0, device=std_p.device)) * std_p.pow(2))
    entropy_q = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0, device=std_q.device)) * std_q.pow(2))
    entropy = (entropy_p + entropy_q).mean()

    # penalty: only when std is smaller than target_std
    penalty_factor = torch.mean(F.relu(target_std - std_p) + F.relu(target_std - std_q)) / target_std
    entropy_reg = -entropy_weight * penalty_factor * entropy

    return w * loss + entropy_reg
