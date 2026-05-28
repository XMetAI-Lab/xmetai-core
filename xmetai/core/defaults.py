# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import functools
import os
import sys
from typing import Optional

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from xmetai.config import CfgNode, LazyConfig
from xmetai.utils import PathManager, comm, seed_all_rng, setup_logger
from xmetai.utils.collect_env import collect_env_info
from xmetai.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

__all__ = [
    "create_ddp_model",
    "create_fsdp_model",
    "create_fsdp2_model",
    "default_argument_parser",
    "default_setup",
    "default_writers",
]


def create_fsdp_model(model, **kwargs):
    """
    Create a FullyShardedDataParallel (FSDP) model if world_size > 1.
    If world_size == 1, return the model unchanged.
    """
    # local training also uses FSDP, so we don't need to check world size.
    if comm.get_world_size() == 1:
        return model
    if kwargs.get("fsdp2", False):
        return create_fsdp2_model(model, **kwargs)
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    precision = kwargs.get("precision", None)
    if precision is None:
        raise ValueError("You must provide precision={param_dtype, reduce_dtype, buffer_dtype}")

    mp = MixedPrecision(
        param_dtype=dtype_dict[precision.param_dtype],
        reduce_dtype=dtype_dict[precision.reduce_dtype],
        buffer_dtype=dtype_dict[precision.buffer_dtype],
    )

    # 处理 cpu_offload
    cpu_offload_flag = kwargs.get("cpu_offload", False)
    if isinstance(cpu_offload_flag, bool):
        cpu_offload = CPUOffload(offload_params=cpu_offload_flag)
    elif isinstance(cpu_offload_flag, CPUOffload):
        cpu_offload = cpu_offload_flag
    else:
        raise ValueError("cpu_offload must be bool or CPUOffload")

    # 使用基于大小的自动包装策略
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=200_000,  # 只对参数超过20万的模块应用FSDP
    )

    fsdp = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=True,
        use_orig_params=True,
        cpu_offload=cpu_offload,
        limit_all_gathers=True,
        # ignored_states=[param for param in model.parameters() if not param.requires_grad],
    )
    # 设置 dtype 供 autocast 使用
    fsdp.dtype = mp.param_dtype
    return fsdp


def create_fsdp2_model(model, **kwargs):
    """
    create FSDP2 model
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        fully_shard,
    )

    # device mesh
    world_size = comm.get_world_size()
    device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    # mix precision
    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    precision = kwargs.get("precision", None)
    if precision is None:
        raise ValueError("You must provide precision={param_dtype, reduce_dtype}")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype_dict[precision.param_dtype],
        reduce_dtype=dtype_dict[precision.reduce_dtype],
    )
    # cpu offload
    cpu_offload_flag = kwargs.get("cpu_offload", False)
    cpu_offload = CPUOffloadPolicy() if cpu_offload_flag else None
    # get mods to tuple for isinstance check
    mods = tuple(set(kwargs.get("mods", ())))
    fsdp_kwargs = dict(
        mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=cpu_offload,
    )
    for module in model.modules():
        if isinstance(module, mods):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs, reshard_after_forward=False)
    model.dtype = mp_policy.param_dtype
    return model


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [torch.cuda.current_device()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    # 设置 dtype 供 autocast 使用
    ddp.dtype = model.dtype
    return ddp


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. " "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval", action="store_true", help="perform evaluation only")
    parser.add_argument("--export", action="store_true", help="perform onnx model export")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we  a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See " "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


# adapted from:
# https://github.com/pytorch/tnt/blob/ebda066f8f55af6a906807d35bc829686618074d/torchtnt/utils/device.py#L328-L346


def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision: The setting to determine which datatypes to use for matrix
        multiplication and convolution operations.
    """
    if not (torch.cuda.is_available()):  # Not relevant for non-CUDA devices
        return
    # for torch < 2.8.0
    if torch.__version__ != "2.9.0":
        torch.set_float32_matmul_precision(precision)
        if precision == "highest":
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = True
        return
    # set precision for torch 2.9.0 or higher
    if precision == "highest":
        torch.backends.fp32_precision = "ieee"  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
        torch.backends.cudnn.fp32_precision = "ieee"  # type: ignore[attr-defined]
        torch.backends.cudnn.conv.fp32_precision = "ieee"  # type: ignore[attr-defined]
    elif precision == "high":
        torch.backends.fp32_precision = "tf32"  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # type: ignore[attr-defined]
        torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore[attr-defined]
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
    elif precision == "medium":
        torch.backends.fp32_precision = "bf16"  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.fp32_precision = "none"  # type: ignore[attr-defined]
        torch.backends.cudnn.fp32_precision = "none"  # type: ignore[attr-defined]
        torch.backends.cudnn.conv.fp32_precision = "none"  # type: ignore[attr-defined]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = False


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info(f"Rank of current process: {rank}. World size: {comm.get_world_size()}")
    logger.info(f"Environment info:\n{collect_env_info()}")

    if comm.is_main_process() and output_dir:
        logger.info(f"Command line arguments: {str(args)}")
        if hasattr(args, "config_file") and args.config_file != "":
            logger.info(f"Contents of args.config_file={args.config_file}:\n{PathManager.open(args.config_file, 'r').read()}")

        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            logger.info(f"Running with full config:\n{cfg.dump()}")
            with PathManager.open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info(f"Full config saved to {path}")

    # make sure each worker has a different, yet deterministic seed if specified
    seed = _try_get_key(cfg, "SEED", "train.seed", default=-1)
    seed_all_rng(None if seed < 0 else seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval") and args.eval):
        torch.backends.cudnn.benchmark = _try_get_key(cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False)

    fp32_precision = _try_get_key(cfg, "FLOAT32_PRECISION", "train.float32_precision", default="")
    if fp32_precision != "":
        logger.info(f"Set fp32 precision to {fp32_precision}")
        _set_float32_precision(fp32_precision)


def default_writers(output_dir: str, max_iter: Optional[int] = None) -> list:
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.

    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations

    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    PathManager.mkdirs(output_dir)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]
