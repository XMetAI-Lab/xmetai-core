"""
Test script for LoRA functionality.

This script tests the LoRA implementation by:
1. Creating a simple model
2. Applying LoRA
3. Verifying parameter counts
4. Testing forward pass
5. Testing merge/unmerge functionality

Usage:
    python -m pytest tests/unit/test_lora.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from xmetai.layers.lora import apply_lora_to_model, merge_lora_weights, unmerge_lora_weights


class SimpleModel(nn.Module):
    """Simple test model with encoder and decoder."""

    def __init__(self, dim=256, num_layers=2):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn_qkv": nn.Linear(dim, dim * 3),
                        "attn_proj": nn.Linear(dim, dim),
                        "mlp_fc1": nn.Linear(dim, dim * 4),
                        "mlp_fc2": nn.Linear(dim * 4, dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        for layer in self.encoder:
            # Simple forward (not actual attention, just for testing)
            qkv = layer["attn_qkv"](x)
            attn_out = layer["attn_proj"](qkv[:, :, : x.size(-1)])
            x = x + attn_out
            x = x + layer["mlp_fc2"](torch.relu(layer["mlp_fc1"](x)))
        return self.head(x)


def test_lora_application():
    """Test applying LoRA to a model."""
    print("\n" + "=" * 60)
    print("Test 1: Applying LoRA to Model")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=2)

    # Count original parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {total_params:,}")

    # Apply LoRA
    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=[
            r"encoder.*attn_qkv$",
            r"encoder.*attn_proj$",
        ],
    )

    model = apply_lora_to_model(model, lora_config, verbose=True)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    assert trainable_params < total_params * 0.1, "LoRA should reduce trainable params significantly"
    print("✓ Test passed: LoRA successfully reduced trainable parameters")

    return model


def test_forward_pass():
    """Test forward pass with LoRA."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass with LoRA")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=2)
    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=[
            r"encoder.*attn_qkv$",
            r"encoder.*attn_proj$",
        ],
    )
    model = apply_lora_to_model(model, lora_config, verbose=False)

    batch_size, seq_len, dim = 2, 10, 256
    x = torch.randn(batch_size, seq_len, dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert output.shape == (batch_size, seq_len, 10), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Test passed: Forward pass successful")


def test_merge_unmerge():
    """Test merging and unmerging LoRA weights."""
    print("\n" + "=" * 60)
    print("Test 3: Merge/Unmerge LoRA Weights")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=2)
    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        dropout=0.1,
        target_modules=[
            r"encoder.*attn_qkv$",
            r"encoder.*attn_proj$",
        ],
    )
    model = apply_lora_to_model(model, lora_config, verbose=False)

    batch_size, seq_len, dim = 2, 10, 256
    x = torch.randn(batch_size, seq_len, dim)

    model.eval()
    with torch.no_grad():
        # Output before merge
        output_before = model(x)

        # Merge weights
        model = merge_lora_weights(model, verbose=True)
        output_merged = model(x)

        # Unmerge weights
        model = unmerge_lora_weights(model, verbose=True)
        output_after = model(x)

    # Check that outputs are the same
    diff_merged = (output_before - output_merged).abs().max().item()
    diff_after = (output_before - output_after).abs().max().item()

    print(f"Max difference (before vs merged): {diff_merged:.8f}")
    print(f"Max difference (before vs after): {diff_after:.8f}")

    assert diff_merged < 1e-5, "Merged output should match original"
    assert diff_after < 1e-5, "Output after unmerge should match original"
    print("✓ Test passed: Merge/unmerge preserves outputs")


def test_gradient_flow():
    """Test that gradients flow through LoRA parameters."""
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow Through LoRA")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=1)

    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        target_modules=[r"encoder.*attn_qkv$"],
    )

    model = apply_lora_to_model(model, lora_config, verbose=False)
    model.train()

    # Create input and target
    x = torch.randn(2, 10, 256, requires_grad=True)
    target = torch.randn(2, 10, 10)

    # Forward and backward
    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    # Check gradients
    lora_grads = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            if param.grad is not None:
                lora_grads.append(param.grad.abs().max().item())
                print(f"{name}: grad_max = {lora_grads[-1]:.6f}")

    assert len(lora_grads) > 0, "No LoRA parameters found"
    assert all(g > 0 for g in lora_grads), "All LoRA parameters should have gradients"
    print("✓ Test passed: Gradients flow through LoRA parameters")


def test_pattern_matching():
    """Test regex pattern matching for target modules."""
    print("\n" + "=" * 60)
    print("Test 5: Pattern Matching for Target Modules")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=2)

    # Test different patterns
    test_cases = [
        {
            "patterns": [r"encoder\.0\..*attn.*"],
            "expected_match": ["encoder.0.attn_qkv", "encoder.0.attn_proj"],
            "expected_no_match": ["encoder.1.attn_qkv", "head"],
        },
        {
            "patterns": [r".*mlp.*"],
            "expected_match": ["encoder.0.mlp_fc1", "encoder.1.mlp_fc2"],
            "expected_no_match": ["encoder.0.attn_qkv", "head"],
        },
        {
            "patterns": [r"encoder.*attn_(qkv|proj)$"],
            "expected_match": ["encoder.0.attn_qkv", "encoder.1.attn_proj"],
            "expected_no_match": ["encoder.0.mlp_fc1", "head"],
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: patterns = {test_case['patterns']}")

        lora_config = dict(
            enabled=True,
            r=4,
            alpha=8,
            target_modules=test_case["patterns"],
        )

        test_model = SimpleModel(dim=256, num_layers=2)
        test_model = apply_lora_to_model(test_model, lora_config, verbose=False)

        # Check which modules have LoRA
        lora_modules = []
        for name, module in test_model.named_modules():
            if "lora_A" in dict(module.named_parameters(recurse=False)):
                # This is a LoRA module, get its parent name
                parent_name = ".".join(name.split(".")[:-1]) if "." in name else name
                lora_modules.append(parent_name)

        print(f"  LoRA applied to: {lora_modules}")

        # Verify expected matches
        for expected in test_case["expected_match"]:
            # Check if any lora module matches (since we match the parent)
            matched = any(expected in lm for lm in lora_modules)
            assert matched, f"Expected {expected} to have LoRA"

        print(f"  ✓ Pattern matching correct for case {i}")

    print("\n✓ Test passed: All pattern matching tests successful")
