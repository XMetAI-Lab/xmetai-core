"""
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-28 19:12:10
Description: Only for evaluation with onnx model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved.
"""

import logging
import os

import numpy as np
import torch

from xmetai.registries import MODELS
from xmetai.utils import get_local_rank

from .base_predictor import BasePredictor

logger = logging.getLogger(__name__)
CUSTOM_OPS_LIB_PATH = "/workspace/cma/xmetai_onnx_plugins.cpython-311-x86_64-linux-gnu.so"

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]
    logger.warning("ONNX Runtime is not available. ONNX inference requires onnxruntime-gpu.")

__all__ = ["OnnxInferModel"]


@MODELS.register()
class OnnxInferModel(BasePredictor):
    def __init__(self, onnx_path, **kwargs):
        self.gpu_mem_fraction = kwargs.pop("gpu_mem_fraction", 0.7)
        self.use_cpu_initializers = kwargs.pop("use_cpu_initializers", True)
        super().__init__(**kwargs)
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

        if os.path.exists(CUSTOM_OPS_LIB_PATH):
            options.register_custom_ops_library(CUSTOM_OPS_LIB_PATH)
            self.logger.info(f"Registered custom op library from {CUSTOM_OPS_LIB_PATH}")

        self.session = ort.InferenceSession(onnx_path, sess_options=options, providers=providers)

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
    def inference(
        self,
        input,
        step,
        num_step,
        times=None,
    ):
        model = self.session
        outputs = input.new_zeros(input.shape[0], num_step, len(self.test_chans), *input.shape[-2:])
        output = input.new_zeros(model.get_outputs()[0].shape)
        step_tensor = input.new_zeros([1])
        hour_tensor = input.new_zeros([1])
        doy_tensor = input.new_zeros([1])
        self.logger.debug(f"inference for output shape:{outputs.shape}, dtype:{outputs.dtype}")
        input_names = [x.name for x in model.get_inputs()]
        io_binding = model.io_binding()
        for k in range(num_step):
            self.logger.debug(f"inference step {step} with input shape:{input.shape}, dtype:{input.dtype}, min: {input.min():.3f}, max: {input.max():.3f}")
            io_binding.bind_input(
                "input",
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=input.shape,
                buffer_ptr=input.data_ptr(),
            )
            timestamp = self.get_timestamp(input, times, step)
            self.logger.debug(f"timestamp: {timestamp}")
            if "step" in input_names:
                step_tensor.fill_(timestamp["step"].item())
                self.logger.debug(f"step_tensor:{step_tensor}")
                io_binding.bind_input(
                    "step",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                    shape=[1],
                    buffer_ptr=step_tensor.data_ptr(),
                )
            if "hour" in input_names:
                hour_tensor.fill_(timestamp["hour"].item())
                self.logger.debug(f"hour_tensor:{hour_tensor}")
                io_binding.bind_input(
                    "hour",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                    shape=[1],
                    buffer_ptr=hour_tensor.data_ptr(),
                )
            if "doy" in input_names:
                doy_tensor.fill_(timestamp["doy"].item())
                self.logger.debug(f"doy_tensor:{doy_tensor}")
                io_binding.bind_input(
                    "doy",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                    shape=[1],
                    buffer_ptr=doy_tensor.data_ptr(),
                )
            self.logger.debug(f"model output shape:{output.shape}, dtype:{output.dtype}")
            io_binding.bind_output(
                model.get_outputs()[0].name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=output.shape,
                buffer_ptr=output.data_ptr(),
            )
            model.run_with_iobinding(io_binding)
            outputs[:, k] = output[:, -1:, self.test_chans]
            input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
            step += 1
        return outputs

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        input = self.inv_normalize(input)
        total_step = max(self.test_frames)
        assert sum(self.step_range) <= total_step

        steps = np.array(self.test_frames) // self.out_frames
        hw = list(input.shape[-2:])
        outputs = input.new_zeros(input.shape[0], len(steps), len(self.test_chans), *hw)  # only save test channels
        self.logger.debug(f"finale outputs shape is {outputs.shape}")
        step = self.start_frames
        num_step = self.step_range[0]

        # for each member, we inference for step_range times. Each time we get num_step results.
        for member in range(self.members):
            self.logger.debug(f"inference with member:{member} for step_range:{self.step_range}.")
            curr_input = input.clone()
            curr_step = step

            output = self.inference(
                curr_input,
                step=curr_step,
                num_step=num_step,
                times=times,
            )
            curr_input = output[:, -self.in_frames :]
            normed_output = self.normalize(output, self.test_chans)
            # accumulate result with current step.
            outputs[:, curr_step : curr_step + num_step] += normed_output

        outputs /= self.members  # ensemble prediction with mean
        return outputs
