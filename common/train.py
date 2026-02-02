# Common training-related configs that are designed for "tools/lazyconfig_train_net.py"
# You can use your own instead, together with your own train_net.py

train = dict(
    output_dir="",
    init_checkpoint="",
    pretrained=dict(),
    max_iter=90000,
    float32_precision="high",  # "high" or "highest" or ""
    freeze=set(),
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=True,
        fp16_compression=False,
    ),
    fsdp=dict(  # options for FullyShardedDataParallel
        cpu_offload=False,
        auto_wrap_policy=None,  # None means no auto wrapping
        precision=dict(param_dtype="bf16", reduce_dtype="bf16", buffer_dtype="fp32"),
    ),
    checkpointer=dict(
        period=1000,
        max_to_keep=3,
    ),
    eval_period=4000,
    val_metric="RMSE/z500",
    log_period=20,
    device_type="cuda",
    use_ddp=True,  # Use DDP or not, default is True (use DDP)
    use_fsdp=False,  # Use FSDP or not, default is False (use DDP)
    use_fsdp2=False,  # Use FSDP2 or not, default is False (use DDP)
    activation_checkpointing=True,  # Use activation checkpointing or not, default is True
    use_ema=True,  # Use Exponential Moving Average (EMA) for model weights, default is True
    ens_train=False,  # Use ensemble training or not, default is False
    teacher_forcing=dict(  # for teacher forcing
        teach_iters=-1,  # -1 means no teacher forcing
        teach_steps=10,  # maximum number of steps for teacher forcing
    ),
    lora=dict(  # LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
        enabled=False,  # Whether to enable LoRA
        r=16,  # LoRA rank (dimension of low-rank matrices)
        alpha=32,  # LoRA scaling factor (effective scale = alpha/r)
        dropout=0.05,  # Dropout probability for LoRA path
        target_modules=[  # Regex patterns for modules to apply LoRA
            # Default: attention layers in encoder
            r"encoder.*\.qkv$",  # Swin-style attention QKV projection
            r"encoder.*\.proj$",  # Swin-style attention output projection
            r"encoder.*\.to_(q|k|v)$",  # CrossAttention Q/K/V projections
            r"encoder.*\.to_out\.0$",  # CrossAttention output projection
        ],
        exclude_modules=[],  # Regex patterns for modules to exclude from LoRA
        modules_to_save=[],  # Additional modules to keep trainable (even if frozen)
        merge_weights=False,  # Whether to merge LoRA weights into base (for export)
        num_steps=1,  # Number of LoRA steps for rollout training. Each step has independent LoRA params
    ),
)
