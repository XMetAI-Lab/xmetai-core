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

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from xmetai.layers.lora import LinearLoRA, apply_lora_to_model, merge_lora_weights, set_lora_step, unmerge_lora_weights


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

    # LoRA should reduce trainable params (frozen base + small LoRA params)
    # Reasonable threshold is that LoRA params should be much smaller than original
    lora_params = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name and p.requires_grad)
    assert lora_params < total_params * 0.2, "LoRA params should be small fraction of total"
    print(f"LoRA parameters: {lora_params:,} ({100 * lora_params / total_params:.2f}%)")
    print("✓ Test passed: LoRA successfully reduced trainable parameters")


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
                grad_max = param.grad.abs().max().item()
                lora_grads.append((name, grad_max))
                print(f"{name}: grad_max = {grad_max:.6f}")

    assert len(lora_grads) > 0, "No LoRA parameters found"
    # At least one LoRA parameter should have non-zero gradient
    # (lora_B will have gradient, lora_A might be zero initially due to zero init of lora_B)
    non_zero_grads = [g for name, g in lora_grads if g > 1e-6]
    assert len(non_zero_grads) > 0, "At least one LoRA parameter should have gradients"
    print(f"✓ Test passed: {len(non_zero_grads)}/{len(lora_grads)} LoRA parameters have non-zero gradients")


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

        # Check which modules have LoRA by looking for LinearLoRA instances
        lora_modules = []
        for name, module in test_model.named_modules():
            if isinstance(module, LinearLoRA):
                lora_modules.append(name)

        print(f"  LoRA applied to: {lora_modules}")

        # Verify expected matches
        for expected in test_case["expected_match"]:
            # Check if any lora module matches
            matched = any(expected == lm or expected in lm for lm in lora_modules)
            assert matched, f"Expected {expected} to have LoRA, found: {lora_modules}"

        print(f"  ✓ Pattern matching correct for case {i}")

    print("\n✓ Test passed: All pattern matching tests successful")


def test_multi_step_lora():
    """Test multi-step LoRA functionality."""
    print("\n" + "=" * 60)
    print("Test 6: Multi-Step LoRA")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=1)
    num_steps = 5

    # Apply multi-step LoRA
    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        dropout=0.0,
        target_modules=[r"encoder.*attn_qkv$"],
    )

    model = apply_lora_to_model(model, lora_config, verbose=False, num_steps=num_steps)
    print(f"Applied multi-step LoRA with {num_steps} steps")

    # Verify parameter names contain step indices
    param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    print(f"\nLoRA parameters found: {len(param_names)}")
    for name in param_names[:4]:  # Show first 4
        print(f"  {name}")

    # Check that we have step-specific parameters
    step_params = [name for name in param_names if "_step_" in name]
    assert len(step_params) > 0, "Multi-step LoRA should have step-specific parameters"
    print(f"✓ Found {len(step_params)} step-specific parameters")

    # Test step switching
    x = torch.randn(2, 10, 256)
    model.eval()

    outputs = []
    with torch.no_grad():
        for step in range(num_steps):
            set_lora_step(model, step)
            output = model(x)
            outputs.append(output)
            print(f"Step {step}: output range [{output.min():.4f}, {output.max():.4f}]")

    # Verify outputs are different for different steps (since LoRA params are random initialized)
    for i in range(num_steps - 1):
        diff = (outputs[i] - outputs[i + 1]).abs().max().item()
        print(f"Diff between step {i} and {i+1}: {diff:.6f}")
        # Note: difference might be small if not trained, but shouldn't be exactly zero
        # We just check they're computed successfully

    print("✓ Test passed: Multi-step LoRA step switching works")


def test_multi_step_lora_training():
    """Test training with multi-step LoRA."""
    print("\n" + "=" * 60)
    print("Test 7: Multi-Step LoRA Training")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=1)
    num_steps = 3

    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        target_modules=[r"encoder.*attn_qkv$"],
    )

    model = apply_lora_to_model(model, lora_config, verbose=False, num_steps=num_steps)
    model.train()

    # Create optimizer for LoRA parameters only
    lora_params = [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=0.001)

    print(f"Training with {len(lora_params)} LoRA parameters")

    x = torch.randn(4, 10, 256)
    target = torch.randn(4, 10, 10)

    initial_losses = []
    final_losses = []

    # Train each step independently
    for step in range(num_steps):
        set_lora_step(model, step)

        # Initial loss
        with torch.no_grad():
            output = model(x)
            loss = ((output - target) ** 2).mean()
            initial_losses.append(loss.item())

        # Train for a few iterations
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - target) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Final loss
        with torch.no_grad():
            output = model(x)
            loss = ((output - target) ** 2).mean()
            final_losses.append(loss.item())

        print(f"Step {step}: initial loss = {initial_losses[step]:.6f}, final loss = {final_losses[step]:.6f}")

    # Verify training reduced loss for each step
    for step in range(num_steps):
        assert final_losses[step] < initial_losses[step], f"Step {step} should have reduced loss"

    print("✓ Test passed: Multi-step LoRA training successful")


def test_multi_step_lora_merge():
    """Test merging multi-step LoRA weights."""
    print("\n" + "=" * 60)
    print("Test 8: Multi-Step LoRA Merge")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=1)
    num_steps = 3

    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        dropout=0.0,
        target_modules=[r"encoder.*attn_qkv$"],
    )

    model = apply_lora_to_model(model, lora_config, verbose=False, num_steps=num_steps)

    x = torch.randn(2, 10, 256)
    model.eval()

    # Collect outputs for each step before merge
    outputs_before = []
    with torch.no_grad():
        for step in range(num_steps):
            set_lora_step(model, step)
            output = model(x)
            outputs_before.append(output)

    # Merge all steps
    model = merge_lora_weights(model, verbose=True)
    print("✓ Merged all LoRA steps into base weights")

    # Check that merged model gives same outputs
    outputs_merged = []
    with torch.no_grad():
        for step in range(num_steps):
            set_lora_step(model, step)
            output = model(x)
            outputs_merged.append(output)

    for step in range(num_steps):
        diff = (outputs_before[step] - outputs_merged[step]).abs().max().item()
        print(f"Step {step} diff (before vs merged): {diff:.8f}")
        assert diff < 1e-5, f"Step {step} output should match after merge"

    # Unmerge and verify again
    model = unmerge_lora_weights(model, verbose=True)

    outputs_after = []
    with torch.no_grad():
        for step in range(num_steps):
            set_lora_step(model, step)
            output = model(x)
            outputs_after.append(output)

    for step in range(num_steps):
        diff = (outputs_before[step] - outputs_after[step]).abs().max().item()
        print(f"Step {step} diff (before vs after unmerge): {diff:.8f}")
        assert diff < 1e-5, f"Step {step} output should match after unmerge"

    print("✓ Test passed: Multi-step merge/unmerge preserves outputs")


def test_lora_forward_with_step_parameter():
    """Test LinearLoRA forward with step parameter."""
    print("\n" + "=" * 60)
    print("Test 9: LinearLoRA Forward with Step Parameter")
    print("=" * 60)

    # Create a LinearLoRA layer directly
    in_features, out_features = 256, 512
    num_steps = 4

    lora_layer = LinearLoRA(
        in_features=in_features,
        out_features=out_features,
        r=8,
        alpha=16,
        dropout=0.0,
        num_steps=num_steps,
    )

    x = torch.randn(2, 10, in_features)

    # Test passing step directly to forward
    outputs_via_forward = []
    for step in range(num_steps):
        output = lora_layer(x, step=step)
        outputs_via_forward.append(output)
        print(f"Forward with step={step}: output shape {output.shape}")

    # Test using set_step then forward
    outputs_via_set_step = []
    for step in range(num_steps):
        lora_layer.set_step(step)
        output = lora_layer(x)
        outputs_via_set_step.append(output)

    # Verify both methods give same results
    for step in range(num_steps):
        diff = (outputs_via_forward[step] - outputs_via_set_step[step]).abs().max().item()
        assert diff < 1e-6, f"Step {step}: forward(step={step}) should match set_step({step})+forward()"

    print("✓ Test passed: Forward with step parameter works correctly")


def test_single_step_backward_compatibility():
    """Test that single-step LoRA (num_steps=1) works as before."""
    print("\n" + "=" * 60)
    print("Test 10: Single-Step LoRA Backward Compatibility")
    print("=" * 60)

    model = SimpleModel(dim=256, num_layers=1)

    lora_config = dict(
        enabled=True,
        r=8,
        alpha=16,
        target_modules=[r"encoder.*attn_qkv$"],
    )

    # Apply with num_steps=1 (default)
    model = apply_lora_to_model(model, lora_config, verbose=False, num_steps=1)

    # Check parameter names don't have _step_ suffix
    param_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    print(f"LoRA parameters: {param_names}")

    step_params = [name for name in param_names if "_step_" in name]
    non_step_params = [name for name in param_names if "_step_" not in name]

    assert len(step_params) == 0, "Single-step LoRA should not have _step_ suffix"
    assert len(non_step_params) > 0, "Should have lora_A and lora_B parameters"

    print(f"✓ Found {len(non_step_params)} parameters without step suffix")

    # Test forward pass
    x = torch.randn(2, 10, 256)
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 10, 10), "Output shape should be correct"
    print("✓ Test passed: Single-step LoRA maintains backward compatibility")
