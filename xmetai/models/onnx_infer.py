"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-28 19:12:10
Description: Only for evaluation with onnx model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

from xmetai.registries import MODELS
from xmetai.utils import get_local_rank

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]
    logger.warning("ONNX Runtime is not available. ONNX inference requires onnxruntime-gpu.")

__all__ = ["OnnxInferModel"]


@MODELS.register()
class OnnxInferModel(BasePredictor):
    def __init__(
        self,
        onnx_path,
        perturb_std_scale=1.0,
        onnx_batch_size=1,
        stream_ensemble=False,
        custom_ops_library="",
        defer_metrics=True,
        **kwargs,
    ):
        self.gpu_mem_fraction = kwargs.pop("gpu_mem_fraction", 0.7)
        self.use_cpu_initializers = kwargs.pop("use_cpu_initializers", True)
        super().__init__(**kwargs)
        self.perturb_std_scale = perturb_std_scale
        self.onnx_batch_size = int(onnx_batch_size)
        self.stream_ensemble = bool(stream_ensemble)
        self.custom_ops_library = custom_ops_library
        self.defer_metrics = bool(defer_metrics)
        if self.onnx_batch_size < 1:
            raise ValueError("onnx_batch_size must be positive")
        self.device_id = get_local_rank()
        self.logger.name = __name__
        self.load_model(onnx_path)

    def load_model(self, onnx_path):
        if ort is None:
            raise ImportError("OnnxInferModel requires onnxruntime-gpu.")
        # ort.set_default_logger_severity(0)  # 0 = VERBOSE
        options = self._build_session_options()

        providers = []
        cuda_provider_options = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
        }
        gpu_mem_limit = self._default_gpu_mem_limit()
        if gpu_mem_limit is not None:
            cuda_provider_options["gpu_mem_limit"] = gpu_mem_limit
            self.logger.info(f"Limiting CUDA provider memory to {gpu_mem_limit / (1024 ** 3):.2f} GB")
        cuda_provider = ("CUDAExecutionProvider", cuda_provider_options)

        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append(cuda_provider)
        else:
            self.logger.warning(f"CUDA Execution Provider not available for device {self.device_id}. Falling back to CPU.")

        # Always add CPU provider as fallback
        if len(providers) == 0:
            self.logger.error("No suitable execution providers found. Ensure ONNX Runtime is installed with CUDA support.")
        else:
            self.logger.info(f"Using CUDA provider on device {self.device_id}")

        providers.append(("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))

        if os.path.exists(self.custom_ops_library):
            options.register_custom_ops_library(self.custom_ops_library)
            self.logger.info(f"Registered custom op library from {self.custom_ops_library}")

        self.session = ort.InferenceSession(onnx_path, sess_options=options, providers=providers)
        self.input_metadata = {node.name: node for node in self.session.get_inputs()}
        self.output_metadata = self.session.get_outputs()[0]
        batch_dimension = self.input_metadata["input"].shape[0]
        self.fixed_batch_size = batch_dimension if isinstance(batch_dimension, int) else None
        if self.fixed_batch_size not in (None, 1):
            raise ValueError(f"Only dynamic batch or fixed batch=1 is supported, got {self.fixed_batch_size}")
        self.runtime_batch_size = 1 if self.fixed_batch_size else self.onnx_batch_size
        self.logger.info(f"ONNX batch contract={batch_dimension}, runtime microbatch={self.runtime_batch_size}")

        self.logger.info(f"Device {self.device_id}: Using providers {self.session.get_providers()}")
        self.logger.info(f"Loaded ONNX model from {onnx_path}")
        return

    def _build_session_options(self):
        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.enable_mem_reuse = True
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.add_session_config_entry("cudnn_conv_algo_search", "HEURISTIC")
        if self.use_cpu_initializers:
            options.add_session_config_entry("session.use_device_allocator_for_initializers", "0")
            options.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
            self.logger.info("Keeping model initializers on host memory to reduce GPU usage")
        return options

    def _default_gpu_mem_limit(self):
        if not torch.cuda.is_available():
            return None
        try:
            props = torch.cuda.get_device_properties(self.device_id)
        except (AssertionError, RuntimeError):
            return None
        limit = int(props.total_memory * self.gpu_mem_fraction)
        return max(limit, 0)

    def forward(self, data):
        if not self.training:
            return self.test(data)
        self.logger.error("Model is in training mode, cannot perform inference.")
        return

    @torch.no_grad()
    def _run_onnx_step(self, input, step, times=None):
        """Run member batches synchronously with independently owned outputs."""
        input = input.float()
        timestamp = self.get_timestamp(input, times, step)
        if "perturb_std_scale" in self.input_metadata:
            timestamp["perturb_std_scale"] = input.new_full((input.shape[0],), self.perturb_std_scale)
        outputs = []
        for batch_index in range(0, input.shape[0], self.runtime_batch_size):
            batch_end = min(batch_index + self.runtime_batch_size, input.shape[0])
            sample = input[batch_index:batch_end].contiguous()
            device_type = sample.device.type
            device_id = self.device_id if sample.is_cuda else 0
            binding = self.session.io_binding()
            binding.bind_input("input", device_type=device_type, device_id=device_id, element_type=np.float32, shape=sample.shape, buffer_ptr=sample.data_ptr())
            # Keep all bound tensors alive until the synchronous ORT call returns.
            scalars = []
            for name in ("step", "hour", "doy", "perturb_std_scale"):
                if name in self.input_metadata:
                    value = timestamp[name][batch_index:batch_end].float().contiguous()
                    scalars.append(value)
                    binding.bind_input(
                        name, device_type=device_type, device_id=device_id, element_type=np.float32, shape=value.shape, buffer_ptr=value.data_ptr()
                    )
            output = sample.new_zeros((sample.shape[0], self.out_frames, self.out_chans, *sample.shape[-2:]))
            binding.bind_output(
                self.output_metadata.name,
                device_type=device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=output.shape,
                buffer_ptr=output.data_ptr(),
            )
            if sample.is_cuda:
                # Torch prepares buffers on its stream; ORT now owns its compute
                # stream. Complete producers before handing ORT raw pointers.
                torch.cuda.current_stream(sample.device).synchronize()
            self.session.run_with_iobinding(binding)
            outputs.append(output)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    @torch.no_grad()
    def stream_test(self, data):
        """Advance members in batches and yield physical fields for verification.

        With defer_metrics, selected-channel results are cached in CPU memory
        and yielded after the complete input batch forecast finishes. Disable
        it for GPU verification: evaluators consume one complete lead before
        recurrence advances, without retaining earlier member fields.
        """
        if not self.stream_ensemble:
            yield self.test(data)
            return
        if self.out_frames != 1 or self.test_rect:
            raise ValueError("Streaming ensemble verification requires out_frames=1 and the complete global grid")
        batch = self.process_data(data)
        input = self.inv_normalize(batch["input"])
        batch_size = input.shape[0]
        member_count = self.members
        states = input.repeat_interleave(member_count, dim=0).reshape(batch_size, member_count, *input.shape[1:])
        del input
        max_step = max(self.test_frames)
        save_steps = {step: index for index, step in enumerate(self.test_frames)}
        finite = torch.ones((), dtype=torch.bool, device=states.device)
        cached_results = []
        for step in range(1, max_step + 1):
            index = save_steps.get(step)
            members = states.new_empty((batch_size, member_count, len(self.test_chans), *states.shape[-2:])) if index is not None else None
            for sample_index in range(batch_size):
                for member_start in range(0, member_count, self.runtime_batch_size):
                    member_end = min(member_start + self.runtime_batch_size, member_count)
                    member_states = states[sample_index, member_start:member_end]
                    times = pd.DatetimeIndex([batch["times"][sample_index]] * (member_end - member_start))
                    output = self._run_onnx_step(member_states, step=self.start_frames + step - 1, times=times)
                    finite.logical_and_(torch.isfinite(output).all())
                    updated_states = torch.cat([member_states, output[:, -1:]], dim=1)[:, -self.in_frames :]
                    member_states.copy_(updated_states)
                    del updated_states
                    if members is not None:
                        members[sample_index, member_start:member_end].copy_(output[:, -1, self.test_chans])
                    del output
            if members is not None:
                if not finite.item():
                    raise FloatingPointError(f"Non-finite ONNX output through forecast step={step}")
                result = {
                    "output": members.mean(1)[:, None],
                    "frame_start": index,
                    "times": batch["times"],
                    "n_members": member_count,
                    "ensemble_output": members,
                }
                if "targets" in batch:
                    result["target"] = self.inv_normalize(batch["targets"][:, index : index + 1], self.test_chans)
                if "weight" in self._buffers:
                    result["weight"] = self.weight.float()
                if self.defer_metrics:
                    cached_results.append(
                        {key: value.detach().to("cpu", non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in result.items()}
                    )
                else:
                    yield result
                del members, result
        # This same-stream scalar read also completes the queued CPU cache copies.
        if not finite.item():
            raise FloatingPointError("Non-finite ONNX output in the forecast trajectory")
        # Recurrence state is no longer needed while evaluators consume the CPU cache.
        del states, member_states, batch
        for result in cached_results:
            # Mean-field metrics remain on GPU; CSV ensemble metrics consume CPU fields.
            for key in ("output", "target", "weight"):
                if key in result:
                    result[key] = result[key].to(self.device)
            yield result

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        input = self.inv_normalize(input)
        total_step = max(self.test_frames)
        assert sum(self.step_range) <= total_step

        steps = np.array(self.test_frames) // self.out_frames
        save_steps = {int(forecast_step): index for index, forecast_step in enumerate(steps)}
        hw = list(input.shape[-2:])
        outputs = torch.zeros(input.shape[0], len(self.test_frames), len(self.test_chans), *hw, device=input.device, dtype=torch.float32)
        self.logger.debug(f"finale outputs shape is {outputs.shape}")
        step = self.start_frames
        num_step = self.step_range[0]

        finite = torch.ones((), dtype=torch.bool, device=input.device)
        for sample_index in range(input.shape[0]):
            for member_start in range(0, self.members, self.runtime_batch_size):
                member_count = min(self.runtime_batch_size, self.members - member_start)
                curr_input = input[sample_index : sample_index + 1].repeat(member_count, 1, 1, 1, 1)
                member_times = pd.DatetimeIndex([times[sample_index]] * member_count)
                for curr_step in range(step, step + num_step):
                    output = self._run_onnx_step(curr_input, step=curr_step, times=member_times)
                    finite.logical_and_(torch.isfinite(output).all())
                    out_index = save_steps.get(curr_step + 1 - self.start_frames)
                    if out_index is not None:
                        outputs[sample_index, out_index].add_(output[:, -1, self.test_chans].sum(0))
                    curr_input = torch.cat([curr_input, output[:, -1:]], dim=1)[:, -self.in_frames :]
                    del output
                del curr_input

        if not finite.item():
            raise FloatingPointError("Non-finite ONNX output in the forecast trajectory")
        # Sum every physical member, including a smaller final batch, before averaging.
        outputs.div_(self.members)
        return self.normalize(outputs, self.test_chans)
