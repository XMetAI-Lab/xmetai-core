"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

This module provides LoRA layers that can replace standard nn.Linear layers
with low-rank decomposition for parameter-efficient fine-tuning.

Reference: https://arxiv.org/abs/2106.09685

Author: XMetAI Team
Date: 2026-02-02
"""

import logging
import re
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearLoRA", "apply_lora_to_model", "merge_lora_weights", "unmerge_lora_weights", "set_lora_step"]

logger = logging.getLogger(__name__)


class LinearLoRA(nn.Module):
    """LoRA-enhanced Linear layer.

    Replaces nn.Linear with low-rank adaptation:
        h = W₀x + (B @ A)x * (alpha/r)

    where W₀ is the frozen pretrained weight, A∈R^(r×in_features),
    B∈R^(out_features×r) are trainable low-rank matrices.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA path
        bias: Whether to include bias (matches original layer)
        merge_weights: Whether to merge LoRA into base weights
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        merge_weights: bool = False,
        original_weight: Optional[torch.Tensor] = None,
        original_bias: Optional[torch.Tensor] = None,
        num_steps: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        self.num_steps = num_steps
        self.current_step = 0

        # Frozen pretrained weights
        self.weight = nn.Parameter(torch.zeros(out_features, in_features) if original_weight is None else original_weight.clone(), requires_grad=False)
        if bias:
            if original_bias is not None:
                self.bias = nn.Parameter(original_bias.clone(), requires_grad=False)
            else:
                self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # LoRA low-rank matrices
        if num_steps == 1:
            # Single LoRA: store as lora_A and lora_B
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            # Multi-step LoRA: store as lora_A_step_0, lora_A_step_1, etc.
            for step in range(num_steps):
                self.register_parameter(f"lora_A_step_{step}", nn.Parameter(torch.zeros(r, in_features)))
                self.register_parameter(f"lora_B_step_{step}", nn.Parameter(torch.zeros(out_features, r)))

        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform for A and zeros for B."""
        if self.num_steps == 1:
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
        else:
            for step in range(self.num_steps):
                lora_a = getattr(self, f"lora_A_step_{step}")
                lora_b = getattr(self, f"lora_B_step_{step}")
                nn.init.kaiming_uniform_(lora_a, a=5**0.5)
                nn.init.zeros_(lora_b)

    def set_step(self, step: int):
        """Set current LoRA step for multi-step LoRA.
        This is optional - you can also pass step directly to forward().
        Useful for setting a default step for multiple forward passes.
        """
        if step >= self.num_steps:
            logger.warning(f"Step {step} exceeds num_steps {self.num_steps}, clamping to {self.num_steps-1}")
            step = self.num_steps - 1
        self.current_step = step

    def merge(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            # W = W₀ + (B @ A) * scaling
            if self.num_steps == 1:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            else:
                # For multi-step, merge all steps
                for step in range(self.num_steps):
                    lora_a = getattr(self, f"lora_A_step_{step}")
                    lora_b = getattr(self, f"lora_B_step_{step}")
                    self.weight.data += (lora_b @ lora_a) * self.scaling
            self.merged = True
            logger.debug(f"Merged LoRA weights: {self.out_features}x{self.in_features}, rank={self.r}")

    def unmerge(self):
        """Separate LoRA weights from base weights."""
        if self.merged:
            # W = W - (B @ A) * scaling
            if self.num_steps == 1:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            else:
                # For multi-step, unmerge all steps
                for step in range(self.num_steps):
                    lora_a = getattr(self, f"lora_A_step_{step}")
                    lora_b = getattr(self, f"lora_B_step_{step}")
                    self.weight.data -= (lora_b @ lora_a) * self.scaling
            self.merged = False
            logger.debug(f"Unmerged LoRA weights: {self.out_features}x{self.in_features}")

    def forward(self, x: torch.Tensor, step: Optional[int] = None) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (*, in_features)
            step: Optional step index for multi-step LoRA. If provided, overrides current_step.
                  If not provided, uses self.current_step.

        Returns:
            Output tensor of shape (*, out_features)
        """
        # Base linear transformation (frozen)
        result = F.linear(x, self.weight, self.bias)

        if not self.merged:
            # Use provided step or current step
            if step is not None:
                if step >= self.num_steps:
                    logger.warning(f"Step {step} exceeds num_steps {self.num_steps}, clamping to {self.num_steps-1}")
                    step = self.num_steps - 1
            else:
                step = self.current_step

            # Add LoRA contribution: (B @ A) @ x * scaling
            x_dropped = self.lora_dropout(x)

            # Get LoRA params for the specified step
            if self.num_steps == 1:
                lora_a, lora_b = self.lora_A, self.lora_B
            else:
                lora_a = getattr(self, f"lora_A_step_{step}")
                lora_b = getattr(self, f"lora_B_step_{step}")

            lora_out = F.linear(F.linear(x_dropped, lora_a), lora_b)
            result = result + lora_out * self.scaling

        return result

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " f"r={self.r}, alpha={self.alpha}, merged={self.merged}"


def _get_target_modules(
    model: nn.Module,
    target_patterns: List[str],
    exclude_patterns: Optional[List[str]] = None,
) -> Dict[str, nn.Linear]:
    """Find all Linear modules matching target patterns.

    Args:
        model: Model to search
        target_patterns: List of regex patterns to match module names
        exclude_patterns: Optional list of patterns to exclude

    Returns:
        Dictionary mapping module names to Linear modules
    """
    target_modules = {}
    exclude_patterns = exclude_patterns or []

    for name, module in model.named_modules():
        # Only consider nn.Linear layers
        if not isinstance(module, nn.Linear):
            continue

        # Check if matches any target pattern
        is_target = any(re.search(pattern, name) for pattern in target_patterns)
        if not is_target:
            continue

        # Check if matches any exclude pattern
        is_excluded = any(re.search(pattern, name) for pattern in exclude_patterns)
        if is_excluded:
            continue

        target_modules[name] = module

    return target_modules


def apply_lora_to_model(
    model: nn.Module,
    lora_config: Union[dict, object],
    verbose: bool = True,
    num_steps: int = 1,
) -> nn.Module:
    """Apply LoRA to target modules in the model.

    Args:
        model: Model to apply LoRA to
        lora_config: LoRA configuration dict with keys:
            - r: LoRA rank (default: 8)
            - alpha: LoRA scaling factor (default: 16)
            - dropout: Dropout rate (default: 0.0)
            - target_modules: List of regex patterns for module names
            - exclude_modules: Optional list of patterns to exclude
            - modules_to_save: Optional list of module names to keep trainable
        verbose: Whether to print detailed information
        num_steps: Number of forecast steps for multi-step LoRA (default: 1)

    Returns:
        Model with LoRA applied
    """
    # Handle both dict and LazyConfig objects
    if hasattr(lora_config, "r"):
        r = lora_config.r
        alpha = lora_config.alpha
        dropout = getattr(lora_config, "dropout", 0.0)
        target_modules = lora_config.target_modules
        exclude_modules = getattr(lora_config, "exclude_modules", [])
        modules_to_save = getattr(lora_config, "modules_to_save", [])
    else:
        r = lora_config.get("r", 8)
        alpha = lora_config.get("alpha", 16)
        dropout = lora_config.get("dropout", 0.0)
        target_modules = lora_config.get("target_modules", [])
        exclude_modules = lora_config.get("exclude_modules", [])
        modules_to_save = lora_config.get("modules_to_save", [])

    if not target_modules:
        logger.warning("No target_modules specified in LoRA config, returning model unchanged")
        return model

    # Find all target Linear modules
    targets = _get_target_modules(model, target_modules, exclude_modules)

    if not targets:
        logger.warning(
            f"No modules matched the target patterns: {target_modules}. "
            f"Available Linear modules: {[n for n, m in model.named_modules() if isinstance(m, nn.Linear)][:5]}..."
        )
        return model

    # Replace each target module with LoRA version
    replaced_count = 0
    for name, original_module in targets.items():
        # Get parent module and attribute name
        *parent_path, attr_name = name.split(".")
        parent = model
        for part in parent_path:
            parent = getattr(parent, part)

        # Create LoRA replacement
        lora_module = LinearLoRA(
            in_features=original_module.in_features,
            out_features=original_module.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=original_module.bias is not None,
            original_weight=original_module.weight.data,
            original_bias=original_module.bias.data if original_module.bias is not None else None,
            num_steps=num_steps,
        )

        # Replace the module
        setattr(parent, attr_name, lora_module)
        replaced_count += 1

        if verbose:
            logger.info(
                f"✓ Applied LoRA to {name}: "
                f"{original_module.in_features}→{original_module.out_features}, "
                f"rank={r}, params={2*r*(original_module.in_features + original_module.out_features)}"
            )

    # Mark modules_to_save as trainable
    for pattern in modules_to_save:
        for name, param in model.named_parameters():
            if re.search(pattern, name) and not name.endswith(("lora_A", "lora_B")):
                param.requires_grad = True
                if verbose:
                    logger.info(f"✓ Marked as trainable: {name}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"\n{'='*60}\n"
        f"LoRA Applied Successfully!\n"
        f"  Replaced modules: {replaced_count}\n"
        f"  LoRA rank: {r}, alpha: {alpha}\n"
        f"  Num steps: {num_steps}\n"
        f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)\n"
        f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)\n"
        f"  Trainable ratio: {100*trainable_params/total_params:.2f}%\n"
        f"{'='*60}"
    )

    return model


def set_lora_step(model: nn.Module, step: int):
    """Set current LoRA step for all LinearLoRA modules in the model.

    Args:
        model: Model containing LinearLoRA modules
        step: Current step index
    """
    for module in model.modules():
        if isinstance(module, LinearLoRA):
            module.set_step(step)


def merge_lora_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Merge all LoRA weights into base weights for inference/export.

    This should be called before exporting to ONNX or for faster inference.
    After merging, LoRA parameters (lora_A, lora_B) are not used in forward pass.

    Args:
        model: Model with LoRA layers
        verbose: Whether to print merge information

    Returns:
        Model with merged weights
    """
    merge_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA) and not module.merged:
            module.merge()
            merge_count += 1

    if verbose:
        if merge_count > 0:
            logger.info(f"✓ Merged {merge_count} LoRA layers into base weights")
        else:
            logger.info("No LoRA layers found or all already merged")

    return model


def unmerge_lora_weights(model: nn.Module, verbose: bool = True) -> nn.Module:
    """Separate LoRA weights from base weights.

    This should be called if you want to continue training after merging.

    Args:
        model: Model with merged LoRA layers
        verbose: Whether to print unmerge information

    Returns:
        Model with unmerged weights
    """
    unmerge_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA) and module.merged:
            module.unmerge()
            unmerge_count += 1

    if verbose:
        if unmerge_count > 0:
            logger.info(f"✓ Unmerged {unmerge_count} LoRA layers from base weights")
        else:
            logger.info("No merged LoRA layers found")

    return model
