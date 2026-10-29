# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import datetime
import itertools
import logging
import math
import operator
import os
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_

from xmetai.utils import EventStorage, EventWriter, PathManager, Timer, get_rank, synchronize
from xmetai.utils.memory import bytes_to_mb, model_cpu_memory_bytes, object_cpu_memory_bytes, optimizer_cpu_memory_bytes, process_memory_bytes

from .checkpoint import Checkpointer, PeriodicCheckpointer
from .scheduler import LRMultiplier, ParamScheduler
from .trainer import HookBase

__all__ = [
    "get_bn_modules",
    "update_bn_stats",
    "flatten_results_dict",
    "CallbackHook",
    "IterationTimer",
    "PeriodicWriter",
    "PeriodicCheckpointerHook",
    "BestCheckpointer",
    "LRScheduler",
    "SchedulerMultiplier",
    "AutogradProfiler",
    "EvalHook",
    "EarlyStoppingHook",
    "PreciseBN",
    "TorchProfiler",
    "TorchMemoryStats",
    "CPUMemoryStats",
    "GradientClipHook",
]


def get_bn_modules(model):
    """Collect batch norm modules for precise BN updates."""
    return [m for m in model.modules() if isinstance(m, _BatchNorm)]


def update_bn_stats(model, data_loader, num_iter):
    """Recompute running BN stats using a few batches."""
    bn_modules = get_bn_modules(model)
    if len(bn_modules) == 0:
        return

    momentum = {bn: bn.momentum for bn in bn_modules}
    training_mode = {bn: bn.training for bn in bn_modules}

    for bn in bn_modules:
        bn.train()
        bn.momentum = 1.0
        bn.reset_running_stats()

    with torch.no_grad():
        for _ in range(num_iter):
            model(next(data_loader))

    for bn in bn_modules:
        bn.momentum = momentum[bn]
        bn.train(training_mode[bn])


# results pattern: {key:{times:values, level1: values, ...}}
def flatten_results_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_results_dict(v)
            for kk, vv in v.items():
                if isinstance(vv, (Sequence, np.ndarray)) and not isinstance(vv, (str, bytes)):
                    vv = np.mean(vv)
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


"""
Implement some common hooks.
"""


class CallbackHook(HookBase):
    """
    Create a hook using callback functions provided by the user.
    """

    def __init__(self, *, before_train=None, after_train=None, before_step=None, after_step=None):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self._before_train = before_train
        self._before_step = before_step
        self._after_step = after_step
        self._after_train = after_train

    def before_train(self):
        if self._before_train:
            self._before_train(self.trainer)

    def after_train(self):
        if self._after_train:
            self._after_train(self.trainer)
        # The functions may be closures that hold reference to the trainer
        # Therefore, delete them to avoid circular reference.
        del self._before_train, self._after_train
        del self._before_step, self._after_step

    def before_step(self):
        if self._before_step:
            self._before_step(self.trainer)

    def after_step(self):
        if self._after_step:
            self._after_step(self.trainer)


class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.

    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        logger = logging.getLogger(__name__)
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.trainer.storage.iter + 1 - self.trainer.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.trainer.storage.iter - self.trainer.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.trainer.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()


class PeriodicWriter(HookBase):
    """
    Write events to EventStorage (by calling ``writer.write()``) periodically.

    It is executed every ``period`` iterations and after the last iteration.
    Note that ``period`` does not affect how data is smoothed by each writer.
    """

    def __init__(self, writers, period=20):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period

    def after_step(self):
        if (self.trainer.iter + 1) % self._period == 0 or (self.trainer.iter == self.trainer.max_iter - 1):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            # If any new data is found (e.g. produced by other after_train),
            # write them before closing
            writer.write()
            writer.close()


class PeriodicCheckpointerHook(PeriodicCheckpointer, HookBase):
    """
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        if ((self.trainer.iter + 1) % self.period == 0) or (self.trainer.iter >= self.max_iter - 1):
            with self.trainer.model_ema():
                self.step(self.trainer.iter)


class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
        self,
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        should_save = False
        additional_state = {}
        if metric_tuple is None:
            self._logger.warning(f"Given val metric {self._val_metric} does not seem to be computed/stored." "Will not be checkpointing based on it.")
        else:
            latest_metric, metric_iter = metric_tuple
            if self.best_metric is None:
                if self._update_best(latest_metric, metric_iter):
                    should_save = True
                    additional_state = {"iteration": metric_iter}
                    self._logger.info(f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps")
            elif self._compare(latest_metric, self.best_metric):
                additional_state = {"iteration": metric_iter}
                should_save = True
                self._logger.info(
                    f"Saved best model as latest eval score for {self._val_metric} is "
                    f"{latest_metric:0.5f}, better than last best score "
                    f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
                )
                self._update_best(latest_metric, metric_iter)
            else:
                self._logger.info(
                    f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                    f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
                )
        # --- 广播 should_save 给所有 rank ---
        obj = [should_save, additional_state]
        if dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        should_save, additional_state = obj

        # --- 所有 rank 都进入 save ---
        if should_save:
            with self.trainer.model_ema():
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0 and next_iter != self.trainer.max_iter:
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()


class SchedulerMultiplier(HookBase):
    def __init__(self, optimizer: Any = None, schedulers: dict | None = None):
        self._optimizer = optimizer
        self._schedulers = {} if schedulers is None else schedulers
        self._logger = logging.getLogger(__name__)

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        self.hyp = {k: self._schedulers[k][0] for k in self._schedulers}

    def after_step(self):
        for k in self._schedulers:
            v = self._schedulers[k][self.trainer.iter]
            self.hyp[k] = v
            self._logger.debug("SchedulerMultiplier %s @ iter %d: %s", k, self.trainer.iter, v)
            self.trainer.storage.put_scalar(k, v, smoothing_hint=False)

        for param_group in self._optimizer.param_groups:
            lr_multiplier = param_group["lr_multiplier"]
            wd_multiplier = param_group["wd_multiplier"]
            lr = self.hyp["lr"] * lr_multiplier
            wd = self.hyp["wd"] * wd_multiplier
            param_group["lr"] = lr
            param_group["weight_decay"] = wd

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer=None, scheduler=None):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim.LRScheduler or fvcore.common.param_scheduler.ParamScheduler):
                if a :class:`ParamScheduler` object, it defines the multiplier over the base LR
                in the optimizer.

        If any argument is not given, will try to obtain it from the trainer.
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

    def before_train(self):
        self._optimizer = self._optimizer or self.trainer.optimizer
        if isinstance(self.scheduler, ParamScheduler):
            self._scheduler = LRMultiplier(
                self._optimizer,
                self.scheduler,
                self.trainer.max_iter,
                last_iter=self.trainer.iter - 1,
            )
        self._best_param_group_id = LRScheduler.get_best_param_group_id(self._optimizer)

    @staticmethod
    def get_best_param_group_id(optimizer):
        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    return i
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    return i

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()

    @property
    def scheduler(self):
        return self._scheduler or self.trainer.scheduler

    def state_dict(self):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            return self.scheduler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            logger = logging.getLogger(__name__)
            logger.info(f"Loading scheduler: {state_dict}")
            self.scheduler.load_state_dict(state_dict)


class TorchProfiler(HookBase):
    """
    A hook which runs `torch.profiler.profile`.

    Examples:
    ::
        hooks.TorchProfiler(
             lambda trainer: 10 < trainer.iter < 20, self.cfg.OUTPUT_DIR
        )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser,
    and the tensorboard visualizations can be visualized using
    ``tensorboard --logdir OUTPUT_DIR/log``
    """

    def __init__(self, enable_predicate, output_dir, *, activities=None, save_tensorboard=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            activities (iterable): same as in `torch.profiler.profile`.
            save_tensorboard (bool): whether to save tensorboard visualizations at (output_dir)/log/
        """
        self._enable_predicate = enable_predicate
        self._activities = activities
        self._output_dir = output_dir
        self._save_tensorboard = save_tensorboard

    def before_step(self):
        if self._enable_predicate(self.trainer):
            if self._save_tensorboard:
                on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    os.path.join(
                        self._output_dir,
                        "log",
                        "profiler-tensorboard-iter{}".format(self.trainer.iter),
                    ),
                    f"worker{get_rank()}",
                )
            else:
                on_trace_ready = None
            self._profiler = torch.profiler.profile(
                activities=self._activities,
                on_trace_ready=on_trace_ready,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
            )
            self._profiler.__enter__()
        else:
            self._profiler = None

    def after_step(self):
        if self._profiler is None:
            return
        self._profiler.__exit__(None, None, None)
        if not self._save_tensorboard:
            PathManager.mkdirs(self._output_dir)
            out_file = os.path.join(self._output_dir, "profiler-trace-iter{}.json".format(self.trainer.iter))
            if "://" not in out_file:
                self._profiler.export_chrome_trace(out_file)
            else:
                # Support non-posix filesystems
                with tempfile.TemporaryDirectory(prefix="detectron2_profiler") as d:
                    tmp_file = os.path.join(d, "tmp.json")
                    self._profiler.export_chrome_trace(tmp_file)
                    with open(tmp_file) as f:
                        content = f.read()
                with PathManager.open(out_file, "w") as f:
                    f.write(content)


class AutogradProfiler(TorchProfiler):
    """
    A hook which runs `torch.autograd.profiler.profile`.

    Examples:
    ::
        hooks.AutogradProfiler(
             lambda trainer: 10 < trainer.iter < 20, self.cfg.OUTPUT_DIR
        )

    The above example will run the profiler for iteration 10~20 and dump
    results to ``OUTPUT_DIR``. We did not profile the first few iterations
    because they are typically slower than the rest.
    The result files can be loaded in the ``chrome://tracing`` page in chrome browser.

    Note:
        When used together with NCCL on older version of GPUs,
        autograd profiler may cause deadlock because it unnecessarily allocates
        memory on every device it sees. The memory management calls, if
        interleaved with NCCL calls, lead to deadlock on GPUs that do not
        support ``cudaLaunchCooperativeKernelMultiDevice``.
    """

    def __init__(self, enable_predicate, output_dir, *, use_cuda=True):
        """
        Args:
            enable_predicate (callable[trainer -> bool]): a function which takes a trainer,
                and returns whether to enable the profiler.
                It will be called once every step, and can be used to select which steps to profile.
            output_dir (str): the output directory to dump tracing files.
            use_cuda (bool): same as in `torch.autograd.profiler.profile`.
        """
        warnings.warn("AutogradProfiler has been deprecated in favor of TorchProfiler.")
        self._enable_predicate = enable_predicate
        self._use_cuda = use_cuda
        self._output_dir = output_dir

    def before_step(self):
        if self._enable_predicate(self.trainer):
            self._profiler = torch.autograd.profiler.profile(use_cuda=self._use_cuda)
            self._profiler.__enter__()
        else:
            self._profiler = None


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.

    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train

    def _do_eval(self):
        with self.trainer.model_ema():
            results = self._func()

        if results:
            assert isinstance(results, dict), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError("[EvalHook] eval_function should return a nested dict of float. " "Got '{}: {}' instead.".format(k, v)) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class PreciseBN(HookBase):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, period, model, data_loader, num_iter):
        """
        Args:
            period (int): the period this hook is run, or 0 to not run during training.
                The hook will always run in the end of training.
            model (nn.Module): a module whose all BN layers in training mode will be
                updated by precise BN.
                Note that user is responsible for ensuring the BN layers to be
                updated are in training mode when this hook is triggered.
            data_loader (iterable): it will produce data to be run by `model(data)`.
            num_iter (int): number of iterations used to compute the precise
                statistics.
        """
        self._logger = logging.getLogger(__name__)
        if len(get_bn_modules(model)) == 0:
            self._logger.info("PreciseBN is disabled because model does not contain BN layers in training mode.")
            self._disabled = True
            return

        self._model = model
        self._data_loader = data_loader
        self._num_iter = num_iter
        self._period = period
        self._disabled = False

        self._data_iter = None

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self._disabled:
            return

        if self._data_iter is None:
            self._data_iter = iter(self._data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    self._logger.info("Running precise-BN ... {}/{} iterations.".format(num_iter, self._num_iter))
                # This way we can reuse the same iterator
                yield next(self._data_iter)

        with EventStorage():  # capture events in a new storage to discard them
            self._logger.info(
                "Running precise-BN for {} iterations...  ".format(self._num_iter) + "Note that this could produce different statistics every time."
            )
            update_bn_stats(self._model, data_loader(), self._num_iter)


class TorchMemoryStats(HookBase):
    """
    Writes pytorch's cuda memory statistics periodically.
    """

    def __init__(self, period=20, max_runs=10):
        """
        Args:
            period (int): Output stats each 'period' iterations
            max_runs (int): Stop the logging after 'max_runs'
        """

        self._logger = logging.getLogger(__name__)
        self._period = period
        self._max_runs = max_runs
        self._runs = 0

    def after_step(self):
        if self._runs > self._max_runs:
            return

        if (self.trainer.iter + 1) % self._period == 0 or (self.trainer.iter == self.trainer.max_iter - 1):
            if torch.cuda.is_available():
                max_reserved_mb = torch.cuda.max_memory_reserved() / 1024.0 / 1024.0
                reserved_mb = torch.cuda.memory_reserved() / 1024.0 / 1024.0
                max_allocated_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                allocated_mb = torch.cuda.memory_allocated() / 1024.0 / 1024.0

                self._logger.info(
                    (" iter: {} " " max_reserved_mem: {:.0f}MB " " reserved_mem: {:.0f}MB " " max_allocated_mem: {:.0f}MB " " allocated_mem: {:.0f}MB ").format(
                        self.trainer.iter,
                        max_reserved_mb,
                        reserved_mb,
                        max_allocated_mb,
                        allocated_mb,
                    )
                )

                self._runs += 1
                if self._runs == self._max_runs:
                    mem_summary = torch.cuda.memory_summary()
                    self._logger.info("\n" + mem_summary)

                torch.cuda.reset_peak_memory_stats()


class CPUMemoryStats(HookBase):
    """
    Periodically records current CPU memory split by data, model, optimizer, and
    untracked RSS. This intentionally does not query CUDA memory.
    """

    def __init__(self, period=20):
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self._logger = logging.getLogger(__name__)
        self._period = period

    def after_optimizer(self, data):
        if (self.trainer.iter + 1) % self._period != 0 and self.trainer.iter != self.trainer.max_iter - 1:
            return

        batch_stats = object_cpu_memory_bytes(data)
        model_stats = model_cpu_memory_bytes(self.trainer.model)
        optimizer_bytes = optimizer_cpu_memory_bytes(self.trainer.optimizer)
        process_stats = process_memory_bytes(include_children=True)
        tracked_main_bytes = batch_stats["total_cpu_bytes"] + model_stats["model_cpu_bytes"] + optimizer_bytes

        process_rss_bytes = process_stats["process_rss_bytes"]
        rss_untracked_bytes = None
        if process_rss_bytes is not None:
            rss_untracked_bytes = max(process_rss_bytes - tracked_main_bytes, 0)

        cpu_memory_data_mb = bytes_to_mb(batch_stats["total_cpu_bytes"])
        cpu_memory_data_tensor_mb = bytes_to_mb(batch_stats["tensor_cpu_bytes"])
        cpu_memory_data_pinned_tensor_mb = bytes_to_mb(batch_stats["pinned_tensor_cpu_bytes"])
        cpu_memory_data_numpy_mb = bytes_to_mb(batch_stats["numpy_bytes"])
        cpu_memory_model_mb = bytes_to_mb(model_stats["model_cpu_bytes"])
        cpu_memory_model_param_mb = bytes_to_mb(model_stats["model_param_cpu_bytes"])
        cpu_memory_model_buffer_mb = bytes_to_mb(model_stats["model_buffer_cpu_bytes"])
        cpu_memory_model_grad_mb = bytes_to_mb(model_stats["model_grad_cpu_bytes"])
        cpu_memory_optimizer_mb = bytes_to_mb(optimizer_bytes)
        cpu_memory_tracked_main_mb = bytes_to_mb(tracked_main_bytes)
        cpu_memory_process_rss_mb = bytes_to_mb(process_rss_bytes) or 0.0
        cpu_memory_process_vms_mb = bytes_to_mb(process_stats["process_vms_bytes"]) or 0.0
        cpu_memory_child_process_rss_mb = bytes_to_mb(process_stats["child_process_rss_bytes"]) or 0.0
        cpu_memory_total_process_rss_mb = bytes_to_mb(process_stats["total_process_rss_bytes"]) or 0.0
        cpu_memory_untracked_or_fragmented_mb = bytes_to_mb(rss_untracked_bytes) or 0.0

        self.trainer.storage.put_scalars(
            smoothing_hint=False,
            cpu_memory_data_mb=cpu_memory_data_mb,
            cpu_memory_data_tensor_mb=cpu_memory_data_tensor_mb,
            cpu_memory_data_pinned_tensor_mb=cpu_memory_data_pinned_tensor_mb,
            cpu_memory_data_numpy_mb=cpu_memory_data_numpy_mb,
            cpu_memory_model_mb=cpu_memory_model_mb,
            cpu_memory_model_param_mb=cpu_memory_model_param_mb,
            cpu_memory_model_buffer_mb=cpu_memory_model_buffer_mb,
            cpu_memory_model_grad_mb=cpu_memory_model_grad_mb,
            cpu_memory_optimizer_mb=cpu_memory_optimizer_mb,
            cpu_memory_tracked_main_mb=cpu_memory_tracked_main_mb,
            cpu_memory_process_rss_mb=cpu_memory_process_rss_mb,
            cpu_memory_process_vms_mb=cpu_memory_process_vms_mb,
            cpu_memory_child_process_rss_mb=cpu_memory_child_process_rss_mb,
            cpu_memory_total_process_rss_mb=cpu_memory_total_process_rss_mb,
            cpu_memory_untracked_or_fragmented_mb=cpu_memory_untracked_or_fragmented_mb,
        )
        self._logger.info(
            (
                " iter: %s rank: %s CPU memory "
                "data: %.0fMB (tensor: %.0fMB pinned: %.0fMB numpy: %.0fMB) "
                "model: %.0fMB (param: %.0fMB buffer: %.0fMB grad: %.0fMB) "
                "optimizer: %.0fMB tracked: %.0fMB rss: %.0fMB child_rss: %.0fMB total_rss: %.0fMB untracked_or_fragmented: %.0fMB"
            ),
            self.trainer.iter,
            get_rank(),
            cpu_memory_data_mb,
            cpu_memory_data_tensor_mb,
            cpu_memory_data_pinned_tensor_mb,
            cpu_memory_data_numpy_mb,
            cpu_memory_model_mb,
            cpu_memory_model_param_mb,
            cpu_memory_model_buffer_mb,
            cpu_memory_model_grad_mb,
            cpu_memory_optimizer_mb,
            cpu_memory_tracked_main_mb,
            cpu_memory_process_rss_mb,
            cpu_memory_child_process_rss_mb,
            cpu_memory_total_process_rss_mb,
            cpu_memory_untracked_or_fragmented_mb,
        )


class GradientClipHook(HookBase):
    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def after_backward(self):
        # FSDP2's DTensor gradients support native distributed norm clipping.
        self.trainer._grad_norm = clip_grad_norm_(self.trainer.model.parameters(), self.max_norm, self.norm_type, foreach=True)


class EarlyStoppingHook(HookBase):
    """Stop training early when a monitored validation metric stops improving.

    The hook reads the metric from ``trainer.storage.latest()`` every
    ``eval_period`` iterations (aligned with :class:`EvalHook`).  If the metric
    does not improve by at least ``min_delta`` for ``patience`` consecutive
    evaluation rounds, it sets ``trainer._early_stop_requested = True``, which
    causes :class:`TrainerBase` to break out of its training loop after the
    current step and call ``after_train`` as if training finished normally.

    Args:
        eval_period (int): How often (in iterations) to check the metric.
            Must match the ``eval_period`` passed to :class:`EvalHook`.
        val_metric (str): Flattened key as written to the event storage, e.g.
            ``"RMSE/z500"`` or ``"IPS/t2m"``.
        mode (str): ``"min"`` to stop when the metric stops decreasing,
            ``"max"`` to stop when it stops increasing.
        patience (int): Number of evaluation rounds with no improvement before
            training is stopped.
        min_delta (float): Minimum absolute change in the metric to qualify as
            an improvement.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        eval_period: int,
        val_metric: str,
        mode: str = "min",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f'mode must be "min" or "max", got {mode!r}')
        if patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience}")
        if eval_period <= 0:
            raise ValueError(f"eval_period must be a positive integer, got {eval_period}")

        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        self._patience = patience
        self._min_delta = float(min_delta)
        self._mode = mode

        self.best_metric: float | None = None
        self.best_iter: int | None = None
        self._num_no_improve: int = 0

    def _is_improvement(self, val: float) -> bool:
        if self.best_metric is None:
            return True
        delta = self._min_delta
        if self._mode == "max":
            return val > self.best_metric + delta
        return val < self.best_metric - delta

    def after_step(self) -> None:
        assert self.trainer is not None
        next_iter = self.trainer.iter + 1
        if next_iter % self._period != 0:
            return
        # Skip at the very last iteration — EvalHook's after_train handles it.
        if next_iter == self.trainer.max_iter:
            return

        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                "EarlyStoppingHook: metric %r not found in storage at iter %d. " "Make sure EvalHook runs before EarlyStoppingHook.",
                self._val_metric,
                self.trainer.iter,
            )
            return

        latest_metric, _ = metric_tuple
        if math.isnan(latest_metric) or math.isinf(latest_metric):
            self._logger.warning(
                "EarlyStoppingHook: metric %r is %s at iter %d, skipping.",
                self._val_metric,
                latest_metric,
                self.trainer.iter,
            )
            return

        if self._is_improvement(latest_metric):
            self._logger.info(
                "EarlyStoppingHook: %r improved from %s to %.6g at iter %d.",
                self._val_metric,
                f"{self.best_metric:.6g}" if self.best_metric is not None else "N/A",
                latest_metric,
                self.trainer.iter,
            )
            self.best_metric = latest_metric
            self.best_iter = self.trainer.iter
            self._num_no_improve = 0
        else:
            self._num_no_improve += 1
            self._logger.info(
                "EarlyStoppingHook: %r did not improve (best=%.6g @ iter %d). " "No-improvement count: %d / %d.",
                self._val_metric,
                self.best_metric,
                self.best_iter,
                self._num_no_improve,
                self._patience,
            )

        if self._num_no_improve >= self._patience:
            self._logger.info(
                "EarlyStoppingHook: stopping training — %r has not improved for " "%d evaluations (best=%.6g @ iter %d).",
                self._val_metric,
                self._patience,
                self.best_metric,
                self.best_iter,
            )
            self.trainer._early_stop_requested = True

    def state_dict(self) -> dict:
        return {
            "best_metric": self.best_metric,
            "best_iter": self.best_iter,
            "num_no_improve": self._num_no_improve,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best_metric = state["best_metric"]
        self.best_iter = state["best_iter"]
        self._num_no_improve = state["num_no_improve"]
