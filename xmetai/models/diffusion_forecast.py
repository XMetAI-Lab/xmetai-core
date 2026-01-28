"""
Detectron2 Diffusion Meta-Architecture + Custom Trainer
Author: douzsh
Date: 2025-09-01

Design goals:
- Minimal external deps: pure PyTorch + Detectron2.
- Inputs follow Detectron2's convention: a batch is a list[dict].
  Required keys per sample:
    - "input": Tensor of shape (T_in, C, H, W)  (conditioning frames)
    - "target": Tensor of shape (T_out, C, H, W) (we train on the first future frame by default)
  Optional: "times" (any metadata you need; not used here)
- Conditioning: uses the *last* conditioning frame (T_in[-1]) concatenated to the noised target as model input.
- Training objective: DDPM epsilon-prediction (can switch to x0 prediction via cfg).
- Inference: DDIM sampler for speed; autoregressive over out_frames if >1.
"""

from typing import Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F

from xmetai.builder import MODELS

from .base_predictor import BasePredictor


def pad_to_even(x):
    H, W = x.shape[-2:]
    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    return F.pad(x, (0, pad_w, 0, pad_h))


@MODELS.register()
class DiffusionForecast(BasePredictor):
    """
    Conditional diffusion forecasting meta-arch.
    Training: predict noise eps from noised target x_t, conditioned on last input frame.
    Inference: DDIM sampling; if OUT_FRAMES>1, autoregressively generate next frames.
    """

    def __init__(
        self,
        base_channels: int,
        ch_mults: Tuple[int, ...],
        attn_res: Tuple[int, ...],
        num_heads: int,
        dropout: float = 0.0,
        num_steps: int = 1000,
        predict_type: str = "epsilon",
        sample_steps: int = 100,
        ddim_eta: float = 0.0,  # for DDIM sampling
        clip_x0: bool = True,
        scheduler_type: str = "ddim",  # "ddpm" or "ddim"
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

        self.logger.name = __name__
        self.predict_type = predict_type
        self.sample_steps = sample_steps
        self.clip_x0 = clip_x0
        self.ddim_eta = ddim_eta

        # Cond + target are concatenated along channel dim
        model_in_ch = self.in_chans * (self.in_frames + 1)  # input conditioning frame + noised target

        t_emb_dim = base_channels * 4
        self.unet = UNet2DModel(
            sample_size=(124, 240),
            in_channels=model_in_ch,
            out_channels=self.in_chans,
            layers_per_block=3,  # 每个 block 里多少个 resnet 层
            block_out_channels=np.array(ch_mults) * base_channels,  # 每层的通道数
            down_block_types=(
                "DownBlock2D",  # 正常下采样
                "DownBlock2D",
                "AttnDownBlock2D",  # 可以混 attention
            ),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            time_embedding_dim=t_emb_dim,
            attention_head_dim=num_heads,
            dropout=dropout,
        )

        # 直接用 diffusers scheduler 管理 betas/alphas
        if scheduler_type == "ddpm":
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_steps,
                prediction_type=predict_type,
                beta_schedule="linear",
            )
        elif scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=num_steps,
                prediction_type=predict_type,
                beta_schedule="linear",
                clip_sample=clip_x0,
            )
        else:
            raise ValueError(f"Unknown scheduler_type {scheduler_type}")

    # -------------- I/O helpers --------------
    @staticmethod
    def preprocess_batch(
        batched_inputs: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert list[dict] to tensors.
        Returns:
            cond: (B, C, H, W) last conditioning frame
            target: (B, C, H, W) first target frame (train step uses one frame)
        """
        conds = batched_inputs["input"]  # list of (B, in_frames, C, H, W)
        targets = batched_inputs["targets"]  # list of (B, out_frames, C, H, W)
        residual = targets - conds[:, -1:]  # (B, out_frames, C, H, W)
        conds = einops.rearrange(conds, "B T C H W -> B (T C) H W")
        targets = einops.rearrange(residual, "B T C H W -> B (T C) H W")  # learn residual
        return conds, targets

    # -------------- forward --------------
    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.training:
            return self.test(data, 24)

        data = self.process_data(data)
        cond, target = self.preprocess_batch(data)
        batch_size = target.shape[0]

        # generate random time step for each sample in the batch
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config["num_train_timesteps"],
            (batch_size,),
            device=target.device,
        )
        # add noise to target
        noise = torch.randn_like(target)
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)
        model_in = torch.cat([cond, noisy_target], dim=1)
        _, _, H, W = model_in.shape
        model_in = pad_to_even(model_in)
        self.logger.debug(f"model input shape: {model_in.shape}, min: {model_in.min().item()}, max: {model_in.max().item()}")
        pred = self.unet(model_in, timesteps, return_dict=False)[0][:, :, :H, :W]
        self.logger.debug(f"model output shape: {pred.shape}, min: {pred.min().item()}, max: {pred.max().item()}")

        if self.predict_type == "epsilon":
            loss = F.mse_loss(pred, noise)
        elif self.predict_type == "sample":
            loss = F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown predict_type {self.predict_type}")

        return {"loss_rec": loss}

    # -------------- inference --------------
    @torch.no_grad()
    def test_cascade(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        input = data["input"]
        total_step = max(self.test_frames)
        steps = np.array(self.test_frames) // self.out_frames
        assert sum(self.step_range) <= total_step

        chw = list(input.shape[-3:])
        outputs = input.new_zeros(input.shape[0], len(steps), *chw)  # batch_size, saved_steps, c,h,w
        self.logger.debug(f"output shape is {outputs.shape}")

        for num_step in self.step_range:
            curr_step = 0
            for step in range(num_step):
                self.reset_input(input)
                last_frame = einops.rearrange(input, "B T C H W -> B (T C) H W")
                x_t = torch.randn(input.shape[0], *chw, device=input.device)
                # 设置 scheduler 的 sampling steps
                self.noise_scheduler.set_timesteps(self.sample_steps, device=input.device)
                for t in self.noise_scheduler.timesteps:
                    model_in = torch.cat([last_frame, x_t], dim=1)
                    _, _, H, W = model_in.shape
                    model_in = pad_to_even(model_in)
                    noise_pred = self.unet(model_in, t, return_dict=False)[0][:, :, :H, :W]

                    # 调用 scheduler 更新
                    x_t = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=x_t,
                        eta=self.ddim_eta,
                    ).prev_sample

                output = x_t.detach().unsqueeze(1) + input[:, -1:]  # x_t is the residual
                input = torch.cat([input, output], dim=1)[:, -self.in_frames :]
                # save for output
                if step + 1 in steps:
                    self.logger.debug(f"Saving output for step {step} with curr_step {curr_step}")
                    outputs[:, curr_step : curr_step + 1, : self.out_chans] += output
                    curr_step += 1

        return outputs
