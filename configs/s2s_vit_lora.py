"""
Example LoRA fine-tuning configuration for S2S ViT model.

This config demonstrates how to use LoRA for parameter-efficient fine-tuning
of a pretrained ViT model. It loads a pretrained checkpoint and only trains
LoRA parameters, drastically reducing memory usage and training time.

Usage:
    # Fine-tune with LoRA
    bash scripts/train.bash -m configs/s2s_vit_lora.py
    
    # Evaluate
    bash scripts/train.bash -m configs/s2s_vit_lora.py --eval

Author: XMetAI Team
Date: 2026-02-02
"""

from s2s_vit import *  # Import base config

# Override training settings for LoRA fine-tuning
train.init_checkpoint = f"{output_dir}/../model_best.pth"  # Load pretrained model
train.max_iter = 10000  # Reduce iterations for fine-tuning
train.eval_period = 1000
train.checkpointer.period = 500

# Enable LoRA
train.lora.enabled = True
train.lora.r = 16  # LoRA rank
train.lora.alpha = 32  # LoRA scaling (alpha/r = 2.0)
train.lora.dropout = 0.05

# Target attention layers in the encoder
train.lora.target_modules = [
    # Mamba/Transformer attention layers
    r"encoder.*\.qkv$",  # QKV projection in WindowAttention
    r"encoder.*\.proj$",  # Output projection in WindowAttention
    r"encoder.*\.to_(q|k|v)$",  # Q/K/V in CrossAttention
    r"encoder.*\.to_out\.0$",  # Output in CrossAttention
    # Optionally include MLP layers for deeper fine-tuning
    # r"encoder.*\.mlp\.fc[12]$",
]

# Optionally keep some modules trainable even if frozen
# train.lora.modules_to_save = ["head.2"]  # Keep final prediction layer trainable

# Reduce learning rate for fine-tuning (LoRA params will use 2x this rate automatically)
optimizer.lr = 5e-5
optimizer.weight_decay = 0.01

# Use cosine annealing for fine-tuning
scheduler = LazyConfig.load(os.path.join(ROOT_PATH, "common/schedule.py")).cosine_lr
scheduler.warmup_iters = 500
scheduler.end_value = 1e-7

# Adjust batch size if needed (LoRA saves memory, so you might be able to use larger batch)
# total_batch_size = 2

# Log more frequently for monitoring fine-tuning
train.log_period = 10

print(f"LoRA Fine-tuning Config:")
print(f"  Pretrained checkpoint: {train.init_checkpoint}")
print(f"  LoRA rank: {train.lora.r}, alpha: {train.lora.alpha}")
print(f"  Target modules: {len(train.lora.target_modules)} patterns")
print(f"  Max iterations: {train.max_iter}")
print(f"  Learning rate: {optimizer.lr}")
