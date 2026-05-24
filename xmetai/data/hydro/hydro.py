"""
Author: Learning Flood Model (Assisted by Gemini)
Description: Flood Dataset (Input: 3 frames History | Target: 1 frame Current)
"""

import logging
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
from xmetai.builder import DATASETS

__all__ = ["FloodDataset"]


@DATASETS.register()
class FloodDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            training: bool = True,
            window_size: int = 3,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.training = training
        self.window_size = window_size

        # 1. 打开 Zarr 文件
        try:
            self.ds_static = xr.open_zarr(f"{data_dir}/static_fp16.zarr", consolidated=False)
            self.ds_dynamic = xr.open_zarr(f"{data_dir}/dynamic_fp16.zarr", consolidated=False)
        except Exception as e:
            self.logger.error(f"Failed to load Zarr from {data_dir}. Error: {e}")
            raise e

        # 2. 确定样本数量
        self.num_samples = self.ds_static.sizes['sample']

        # 3. 定义变量名
        self.static_vars = ["dem", "lai", "sand", "clay", "som"]
        self.rain_var = "rainfall_depth"
        self.target_vars = ["h", "u", "v"]

        # 4. 获取最大时间步长
        self.max_time_steps = self.ds_dynamic.sizes['time']

        self.logger.info(f"FloodDataset loaded! Many-to-One Mode. Window Size={self.window_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- 1. 确定时间窗口 ---
        end_t = np.random.randint(0, self.max_time_steps)
        start_t = end_t - self.window_size + 1

        # ---------------------------------------------------------
        # A. 准备基础输入 (地形 & 降雨)
        # ---------------------------------------------------------

        # 1. 静态数据 [window, 5, H, W]
        static_raw = self.ds_static['data'].isel(sample=idx).sel(level=self.static_vars).values
        static_data = np.repeat(static_raw[np.newaxis, :, :, :], self.window_size, axis=0)

        # 2. 动态降雨 [window, 1, H, W] (含补零)
        rain_final = np.zeros((self.window_size, 1000, 1000), dtype=np.float32)
        valid_indices = [t for t in range(start_t, end_t + 1) if t >= 0]

        if len(valid_indices) > 0:
            rain_valid = self.ds_dynamic['data'].isel(sample=idx, time=valid_indices).sel(level=self.rain_var).values
            if len(valid_indices) == 1 and rain_valid.ndim == 2:
                rain_valid = rain_valid[np.newaxis, :, :]
            fill_length = len(valid_indices)
            rain_final[-fill_length:] = rain_valid

        rain_data = rain_final[:, np.newaxis, :, :]

        # ---------------------------------------------------------
        # B. 准备水流数据 (构造 Input 和 Target)
        # ---------------------------------------------------------

        # 先读取完整的 [window, 3, H, W] 数据
        hydro_final = np.zeros((self.window_size, 3, 1000, 1000), dtype=np.float32)

        if len(valid_indices) > 0:
            hydro_valid = self.ds_dynamic['data'].isel(sample=idx, time=valid_indices).sel(
                level=self.target_vars).values
            if len(valid_indices) == 1 and hydro_valid.ndim == 3:
                hydro_valid = hydro_valid[np.newaxis, :, :, :]
            fill_length = len(valid_indices)
            hydro_final[-fill_length:] = hydro_valid

        # --- 构造 Input ---
        # 复制一份作为输入
        hydro_input = hydro_final.copy()
        # 关键：把 Input 里最后一帧 (当前时刻) 的水深流速抹除 (Mask)
        # 这样输入就是: [真实(t-2), 真实(t-1), 0.0(t)]
        hydro_input[-1, :, :, :] = 0.0

        # --- 构造 Target (修改了这里) ---
        # 目标只取最后一帧 (当前时刻)
        # Shape 变化: [3, 3, H, W] -> [1, 3, H, W]
        # 注意：使用 [-1:] 可以保持维度不消失，变成 (1, 3, ...)，而不是 (3, ...)
        # 这对保持 4D Tensor 结构很重要
        target_np = hydro_final[-1:, :, :, :]

        # ---------------------------------------------------------
        # C. 拼接与返回
        # ---------------------------------------------------------

        # Input: [Rain(1) + Static(5) + Hydro(3)] = 9 通道
        input_np = np.concatenate([rain_data, static_data, hydro_input], axis=1)

        inputs = torch.from_numpy(input_np).float()
        inputs = torch.nan_to_num(inputs, nan=0.0)

        targets = torch.from_numpy(target_np).float()
        targets = torch.nan_to_num(targets, nan=0.0)

        return dict(
            inputs=inputs,
            targets=targets,
            # === 【新增】塞一个假的时间给它，骗过模型 ===
            times=torch.tensor(0.0),
            meta_time_indices=torch.tensor(valid_indices),  # 可以删除
            meta_sample_idx=idx  # 可以删除
        )