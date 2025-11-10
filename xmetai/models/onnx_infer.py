'''
Author: DouZesheng && douzsh@gmail.com
Date: 2025-07-28 19:12:10
Description: Only for evaluation with onnx model.
Copyright (c) 2025 by XMetAI/douzesheng, All Rights Reserved. 
'''
import torch
import numpy as np
from .base_predictor import BasePredictor
from xmetai.utils.comm import get_local_rank
from xmetai.builder import MODELS
import onnxruntime as ort
try:
    from onnxruntime_extensions import onnx_op, PyOp, get_library_path
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    @onnx_op(op_type="SelectiveScanFn", inputs=[PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float,
                                                PyOp.dt_float, PyOp.dt_float, PyOp.dt_float, PyOp.dt_float],
             outputs=[PyOp.dt_float],
             attrs={"delta_softplus": PyOp.dt_int64, "return_last_state": PyOp.dt_int64})
    def selective_scan_onnx(u, delta, A, B, C, D, z, delta_bias, **kwargs):
        delta_softplus = kwargs.get("delta_softplus", 0) == 1
        return_last_state = kwargs.get("return_last_state", 0) == 1
        # convert all inputs to torch tensors
        u = torch.from_numpy(u).to(device="cuda")
        delta = torch.from_numpy(delta).to(device="cuda")
        A = torch.from_numpy(A).to(device="cuda")
        B = torch.from_numpy(B).to(device="cuda")
        C = torch.from_numpy(C).to(device="cuda")
        D = torch.from_numpy(D).to(device="cuda")
        z = None
        delta_bias = torch.from_numpy(delta_bias).to(device="cuda")
        return selective_scan_fn(u, delta, A, B, C, D, z, delta_bias,
                                 delta_softplus, return_last_state)
except ImportError as e:
    print(f"import plugin error: {e}.")
    get_library_path = None

__all__ = ["OnnxInferModel"]


@MODELS.register()
class OnnxInferModel(BasePredictor):
    def __init__(
        self,
        onnx_path,
        freq=24,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.device_id = get_local_rank()
        self.freq = freq
        self.logger.name = __name__
        self.load_model(onnx_path)

    def load_model(self, onnx_path):
        ort.set_default_logger_severity(3)  # 0 = VERBOSE
        options = ort.SessionOptions()

        # Performance optimization settings
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        options.add_session_config_entry("arena_extend_strategy", "kNextPowerOfTwo")

        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("cudnn_conv_algo_search", "EXHAUSTIVE")
        options.add_session_config_entry("session.enable_bfloat16", "1")

        providers = []
        # Configure CUDA provider with the specific GPU
        cuda_provider = ('CUDAExecutionProvider', {
            'device_id': self.device_id,
            'arena_extend_strategy': 'kSameAsRequested',
        })

        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append(cuda_provider)
        else:
            self.logger.warning(f"CUDA Execution Provider not available for device {self.device_id}. Falling back to CPU.")

        # Always add CPU provider as fallback
        if len(providers) == 0:
            self.logger.error("No suitable execution providers found. Ensure ONNX Runtime is installed with CUDA support.")
        else:
            self.logger.info(f"Using CUDA provider on device {self.device_id}")

        # register custom op library for Mamba if get_library_path() is available
        if get_library_path is not None:
            options.register_custom_ops_library(get_library_path())
            self.logger.info(f"Registered custom op library from {get_library_path()}")

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=options,
            providers=providers
        )

        self.logger.info(f"Device {self.device_id}: Using providers {self.session.get_providers()}")
        self.logger.info(f"Loaded ONNX model from {onnx_path}")
        return

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
        outputs = input.new_zeros(input.shape[0], num_step, *input.shape[-3:])
        output = input.new_zeros(input.shape)
        step_tensor = input.new_zeros([1])
        hour_tensor = input.new_zeros([1])
        doy_tensor = input.new_zeros([1])
        self.logger.debug(f"inference for output shape:{outputs.shape}, dtype:{outputs.dtype}")
        model = self.session
        input_names = [x.name for x in model.get_inputs()]
        io_binding = model.io_binding()
        for k in range(num_step):
            self.logger.debug(f"inference step {step} with input shape:{input.shape}, dtype:{input.dtype}, min: {input.min():.3f}, max: {input.max():.3f}")
            io_binding.bind_input("input", device_type="cuda", device_id=self.device_id,
                                  element_type=np.float32, shape=input.shape, buffer_ptr=input.data_ptr())
            timestamp = self.get_timestamp(input, times, step, freq=self.freq)
            self.logger.debug(f"timestamp: {timestamp}")
            if "step" in input_names:
                step_tensor.fill_(timestamp["step"].item())
                self.logger.debug(f"step_tensor:{step_tensor}")
                io_binding.bind_input("step", device_type="cuda", device_id=self.device_id, element_type=np.float32,
                                      shape=[1], buffer_ptr=step_tensor.data_ptr())
            if "hour" in input_names:
                hour_tensor.fill_(timestamp["hour"].item())
                self.logger.debug(f"hour_tensor:{hour_tensor}")
                io_binding.bind_input("hour", device_type="cuda", device_id=self.device_id, element_type=np.float32,
                                      shape=[1], buffer_ptr=hour_tensor.data_ptr())
            if "doy" in input_names:
                doy_tensor.fill_(timestamp["doy"].item())
                self.logger.debug(f"doy_tensor:{doy_tensor}")
                io_binding.bind_input("doy", device_type="cuda", device_id=self.device_id, element_type=np.float32,
                                      shape=[1], buffer_ptr=doy_tensor.data_ptr())
            self.logger.debug(f"model output shape:{output.shape}, dtype:{output.dtype}")
            io_binding.bind_output(model.get_outputs()[0].name, device_type="cuda", device_id=self.device_id,
                                   element_type=np.float32, shape=output.shape, buffer_ptr=output.data_ptr())
            model.run_with_iobinding(io_binding)
            outputs[:, k] = output[:, -1:]
            input = output
            step += 1
        return outputs

    @torch.no_grad()
    def test_cascade(self, data):
        input = data["input"]
        times = data["times"]
        self.logger.debug(
            f"input: min/max/mean for channel: \n{input[:, :, self.test_chans].amin(axis=(0, 1, 3, 4))}\n{input[:, :, self.test_chans].amax(axis=(0, 1, 3, 4))}\n{input[:, :, self.test_chans].mean(axis=(0, 1, 3, 4))}")
        input = self.inv_normalize(input)
        self.logger.debug(
            f"after denorm: min/max/mean for channel: \n{input[:, :, self.test_chans].amin(axis=(0, 1, 3, 4))}\n{input[:, :, self.test_chans].amax(axis=(0, 1, 3, 4))}\n{input[:, :, self.test_chans].mean(axis=(0, 1, 3, 4))}")
        total_step = max(self.test_frames)
        assert sum(self.step_range) <= total_step

        steps = np.array(self.test_frames) // self.out_frames
        chw = list(input.shape[-3:])
        outputs = input.new_zeros(input.shape[0], len(steps), *chw)  # batch_size, saved_steps, c,h,w
        self.logger.debug(f"output shape is {outputs.shape}")
        step = self.start_frames

        # for each member, we inference for step_range times. Each time we get num_step results.
        for member in range(self.members):
            self.logger.debug(f"inference with member:{member} for step_range:{self.step_range}.")
            curr_input = input.clone()
            curr_step = step

            for num_step in self.step_range:
                output = self.inference(
                    curr_input,
                    step=curr_step,
                    num_step=num_step,
                    times=times,
                )
                output = self.normalize(output)
                curr_input = output[:, -self.in_frames:]
                # n x 42 x 76 x 121 x 240
                out_chans = output.shape[2]
                # accumulate result with current step.
                outputs[:, curr_step: curr_step + num_step, :out_chans] += output

                curr_step += num_step
            self.logger.debug(
                f"outputs: min/max/mean for channel: \n{outputs[:, :, self.test_chans].amin(axis=(0, 1, 3, 4))}\n{outputs[:, :, self.test_chans].amax(axis=(0, 1, 3, 4))}\n{outputs[:, :, self.test_chans].mean(axis=(0, 1, 3, 4))}")

        outputs /= self.members  # ensemble prediction with mean
        return outputs
