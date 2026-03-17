"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-10 10:02:24
Description: Losses for KL Divergence and Entropy Regularization.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

__all__ = ["kl_entropy_loss"]


def kl_entropy_loss(dist_p, dist_q, w=1e-4, entropy_weight=1e-5, min_std=1e-3, target_std=0.1, mu_penalty=0.0):
    # 1. Apply soft lower bound to stddev to avoid collapse
    std_p = torch.clamp(dist_p.stddev, min=min_std)
    std_q = torch.clamp(dist_q.stddev, min=min_std)

    # 2. Replace distributions with stabilized versions
    dist_p = torch.distributions.Normal(dist_p.mean, std_p)
    dist_q = torch.distributions.Normal(dist_q.mean, std_q)

    # 3. KL divergence
    kl = kl_divergence(dist_q, dist_p)  # KL(q || p)

    # Optional mean penalty to encourage means to be close (if mu_penalty > 0)
    if mu_penalty > 0.0:
        var_p = std_p.pow(2)
        # mean_term: (mu_q - mu_p)^2 / (2 * var_p)
        mean_term = (dist_q.mean - dist_p.mean).pow(2) / (2 * var_p)
        kl = kl + mu_penalty * mean_term

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
