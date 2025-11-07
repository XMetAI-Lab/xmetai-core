import os
import sys
import logging
from functools import partial
from typing import Dict, Any
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

# set python path from ../
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_PATH + "/../")
sys.path.append(CUR_PATH + "/../configs/")

try:
    from xmetai.config import LazyConfig, instantiate
    from xmetai.models.transformer import SwinBlock
    from xmetai.core import (
        LRMultiplier,
        FSDPCheckpointer,
        SimpleTrainer,
        ExpandTrainer,
        default_argument_parser,
        default_setup,
        default_writers,
        hooks,
        launch,
        inference_on_dataset,
        create_ddp_model,
        create_fsdp_model,
    )
    from xmetai.utils import comm
except ImportError as e:
    print(f"ImportError: {e}. You can do this by running `pip install -e .` in the root directory.")
    sys.exit(1)
try:
    from xmetai.layers import MambaVisionLayer
except Exception:
    MambaVisionLayer = None

logger = logging.getLogger("xmetai")


def build_model(cfg):
    model = instantiate(cfg.model)
    if comm.is_main_process():
        logger.info("Model parameters: {:.3f}M".format(model.params() / 1e6))
    model.freeze(cfg.train.freeze)
    model.to(cfg.train.device_type)
    mods = (SwinBlock, torch.nn.TransformerEncoderLayer)
    if MambaVisionLayer:
        mods += (MambaVisionLayer,)
    cfg.train.fsdp.mods = mods
    # Apply activation checkpointing
    if cfg.train.activation_checkpointing:
        def wrap_f1(module):
            return checkpoint_wrapper(module,
                                      checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                                      preserve_rng_state=False)

        def check_fn(submodule):
            return isinstance(submodule, mods)

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrap_f1,
            check_fn=check_fn
        )

    # clear freezed parameters grad
    def clear_grad(module):
        for param in module.parameters():
            if not param.requires_grad and param.grad is not None:
                param.grad = None
    model.apply(clear_grad)
    # Wrap model with DDP or FSDP
    if cfg.train.use_ddp:
        model = create_ddp_model(model, **cfg.train.ddp)
    elif cfg.train.use_fsdp:
        logger.info(f"fsdp for mods:{cfg.train.fsdp.mods}")
        model = create_fsdp_model(model, **cfg.train.fsdp)
    else:
        logger.warning("Neither DDP nor FSDP is enabled, using single GPU training.")
    model.device = cfg.train.device_type
    return model


def convert_model(model: torch.nn.Module, dtype=torch.float16, ignore_list=["mean", "std"]):
    """ Convert all floating point parameters and buffers of the model to the specified dtype,
        except those in the ignore_list.
    """
    for name, param in list(model.named_parameters(recurse=True)):
        if "." in name:
            module_name, param_name = name.rsplit(".", 1)
            module = dict(model.named_modules())[module_name]
        else:
            module = model
            param_name = name
        if torch.is_floating_point(param):
            new_param = torch.nn.Parameter(param.to(dtype=dtype), requires_grad=False)
            setattr(module, param_name, new_param)
    for name, buf in list(model.named_buffers(recurse=True)):
        if "." in name:
            module_name, buf_name = name.rsplit(".", 1)
            module = dict(model.named_modules())[module_name]
        else:
            module = model
            buf_name = name
        if torch.is_floating_point(buf) and name not in ignore_list:
            setattr(module, buf_name, buf.to(dtype=dtype))
    return model


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        return ret


def do_train(args, cfg):
    model = build_model(cfg)
    logger.debug(model)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    train_loader = instantiate(cfg.dataloader.train)
    scheduler = instantiate(cfg.scheduler)
    if cfg.train.teacher_forcing.teach_iters > 0:
        trainer = ExpandTrainer(model, train_loader, optimizer, increment_iter=cfg.train.teacher_forcing.teach_iters,
                                use_ema=cfg.train.use_ema, ens_train=cfg.train.ens_train)
    else:
        trainer = SimpleTrainer(model, train_loader, optimizer, use_ema=cfg.train.use_ema)

    checkpointer = FSDPCheckpointer(
        model,
        save_dir=cfg.train.output_dir,
        trainer=trainer,
    )
    lr_scheduler = LRMultiplier(optimizer, scheduler, cfg.train.max_iter)
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=lr_scheduler),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer),
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.GradientClipHook(),
            hooks.BestCheckpointer(eval_period=cfg.train.eval_period, checkpointer=checkpointer, val_metric=cfg.train.val_metric, mode="min"),
            (
                hooks.PeriodicWriter(
                    default_writers(cfg.train.output_dir, cfg.train.max_iter),
                    period=cfg.train.log_period,
                )
                if comm.is_main_process()
                else None
            ),
            (
                hooks.TorchMemoryStats(cfg.train.log_period, 20)
                if comm.is_main_process()
                else None
            ),
            # hooks.TorchProfiler(lambda trainer: 10 < trainer.iter < 20, cfg.train.output_dir, save_tensorboard=False),
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)

    if cfg.train.use_ema:
        trainer.ema.update_params(model)

    if args.resume:
        start_iter = trainer.iter + 1
    else:
        start_iter = 0

    trainer.train(start_iter, cfg.train.max_iter)


def do_export(
    model,
    data: Dict[str, Any],
    save_name: str = "model.onnx",
    forward_name: str = "onnx_export",
    device: str = "cuda",
    opset_version: int = 17,
) -> None:
    model.eval()
    if model.dtype != torch.float32:
        convert_model(model, dtype=model.dtype)  # convert model to float16 by default
    model.forward = partial(getattr(model, forward_name))

    # Prepare input tensors
    input_tensors = []
    input_names = []
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            input_tensors.append(v.to(device))
        else:
            input_tensors.append(torch.tensor(v, device=device))
        input_names.append(k)
        logger.info(f'input:{k} with shape:{input_tensors[-1].shape} and dtype:{input_tensors[-1].dtype} on device:{input_tensors[-1].device}')

    torch.onnx.export(
        model,
        tuple(input_tensors),
        save_name,
        export_params=True,
        verbose=True if logging.DEBUG == logger.getEffectiveLevel() else False,
        input_names=input_names,
        output_names=["pred"],
        opset_version=opset_version,
        do_constant_folding=False,
    )
    logger.info(f"Exported model to {save_name}")

    type_size = torch.tensor([], dtype=model.dtype).element_size()
    if model.params() * type_size > 2**31:  # 2GB limit
        logger.warning(f"Model size {model.params() * type_size / 1e9:.2f}GB exceeds the ONNX limit of 2GB. Consider export params to one extra data file.")
        import onnx
        model_onnx = onnx.load(save_name, load_external_data=True)
        location = os.path.basename(save_name).replace(".onnx", ".data")
        onnx.save_model(
            model_onnx,
            save_name,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location,
            size_threshold=1024,
            convert_attribute=True,
        )
        logger.info(f"Exported model to {save_name} with external data file {location}")


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval:
        model = build_model(cfg)
        FSDPCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model)
        torch.distributed.destroy_process_group()
    elif args.export:
        model = build_model(cfg)
        FSDPCheckpointer(model).load(cfg.train.init_checkpoint)
        do_export(model, **cfg.train.export_kwargs)
    else:
        do_train(args, cfg)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # check if logging level is set
    if logging.getLogger().getEffectiveLevel() > logging.INFO:
        logging.getLogger().setLevel(logging.INFO)

    args = default_argument_parser().parse_args()

    if args.export:
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
