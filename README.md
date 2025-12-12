# 雄安气象人工智能创新研究院-核心框架

## 项目概述

这是一个基于深度学习的天气预报模型研究项目，专注于气象人工智能模型的训练开发。项目包含多种先进的神经网络架构，支持从天气预报与气候预测(Weather, Sub-season)。

### 核心特性

- 🌍 **多种模型架构**： ViT,Diffusion(扩散模型)
- 🚀 **分布式训练**：支持单机多卡和多机多卡分布式训练
- 📊 **灵活配置系统**：基于LazyConfig的配置管理(基于[Detectron2](https://github.com/facebookresearch/detectron2)开发)
- 📈 **全面的评估指标**：MSE、TCC等多种评估指标

---

## 项目结构

```
xmetai-core/
├── xmetai/                       # 核心库
│   ├── models/                   # 模型架构
│   │   ├── base_predictor.py     # 基础推理模型
│   │   ├── vit.py                # Vision Transformer模型
│   │   ├── vit_ens.py            # ViT模型的集合训练版本
│   │   └── onnx_infer.py         # ONNX推理模型
│   ├── data/                     # 数据处理
│   │   ├── weather/              # 天气数据工具
│   │   ├── loader.py             # 数据加载器
│   │   └── sampler.py            # 数据采样器
│   ├── core/                     # 核心模块
│   │   ├── trainer.py            # 训练器
│   │   ├── evaluator.py          # 评估器
│   │   ├── hooks.py              # 训练钩子
│   │   └── checkpoint.py         # 检查点管理
│   ├── config/                   # 配置系统
│   ├── layers/                   # 神经网络层
│   ├── loss/                     # 损失函数
│   └── metrics/                  # 评估指标
├── configs/                      # 配置文件
│   ├── s2s_base.py               # S2S基础配置
│   ├── s2s_vit.py                # S2S确定性预报配置
│   └── ...
├── tools/                        # 训练和导出脚本
│   ├── train.py                  # 标准训练脚本
│   └── train_lightning.py        # PyTorch Lightning训练脚本
└── scripts/                      # 辅助脚本
    └── train.bash                # 训练、评估和导出的统一脚本
```

---

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 数据准备

当前支持的数据集为Zarr格式，包含ERA5的1.5°逐天数据，以及0.25°逐6小时数据。数据目录结构如下：

```
data/
├── s2s.1950-2024.c76/     # 1.5°逐天数据
│   ├── mean.npy             # 数据均值
│   ├── std.npy              # 数据标准差
│   └── weight.npy           # 纬度权重
├── land.1950-2024.c8/     # 1.5°逐天陆地数据
│   ├── mean.npy             # 数据均值
│   ├── std.npy              # 数据标准差
│   └── weight.npy           # 纬度权重
├── ens.1979-2024.c78/     # 0.25°逐6小时数据
│   ├── mean.npy             # 数据均值
│   ├── std.npy              # 数据标准差
│   └── weight.npy           # 纬度权重
└── ...
```

数据文件应包含以下变量：
- **大气变量**：z(位势高度), t(温度), u(纬向风), v(经向风), q(比湿)
- **地表变量**：sst(海表温度), tp(总降水), ssr(地表净太阳辐射)等

如需获取数据集请与作者联系。

---

## 模型简介

### ViT (Vision Transformer模型)

ViT模型将天气预报问题视为视觉任务，使用Vision Transformer架构来处理气象数据。该模型能够捕捉全局依赖关系，并在处理高分辨率数据时表现出色。

- **特点**：
  - 使用标准Vision Transformer 或 Mamba架构
  - 支持时间嵌入以处理时间序列数据
  - 可选Mamba风格的Transformer块
  - 适用于确定性预报

### ViTEns (使用集合训练的ViT模型)

ViTEns模型通过在初始场中引入扰动来生成集合预报。该模型使用U-Transformer架构，并在潜在空间中进行扰动采样。

- **特点**：
  - 在初始场中引入扰动
  - 使用U-Transformer作为主干网络
  - 支持2D扰动采样
  - 适用于集合预报生成

### DiffusionForecast (扩散模型)

DiffusionForecast模型基于扩散模型框架，通过逐步去噪过程生成天气预报。该模型使用UNet作为主干网络，并支持DDIM采样以提高推理速度。

- **特点**：
  - 基于扩散模型框架
  - 使用UNet作为主干网络
  - 支持DDIM快速采样
  - 适用于高分辨率预报

---

## 训练/评估/导出指南

### 训练工作流

#### 1. 训练模型

##### 单GPU训练
```bash
bash scripts/train.bash \
    --stage train \
    --model configs/s2s_vit.py \
    --lr 0.0005 \
    --gpu 1 \
    --batch 1 \
    --depth-list "6,6,12,12"
```

##### 多GPU训练(单机多卡)
```bash
bash scripts/train.bash \
    --stage train \
    --model configs/s2s_vit.py \
    --gpu 4 \
    --batch 2 \
    --nnodes 1 \
    --lr 0.0005
```

##### 多机多卡训练
```bash
bash scripts/train.bash \
    --stage train \
    --model configs/s2s_vit.py \
    --gpu 4 \
    --nnodes 4 \
    --batch 2 \
    --master-addr 10.0.0.1 \
    --lr 0.0005
```

##### 使用PyTorch Lightning训练
```bash
bash scripts/train.bash \
    --stage ltrain \
    --model configs/s2s_vit.py \
    --gpu 2 \
    --batch 2 \
    --lr 0.0005
```

#### 2. 模型评估

##### 单GPU评估
```bash
bash scripts/train.bash \
    --stage eval \
    --model configs/s2s_vit.py \
    --members 10 \
    --save
```

##### 多GPU评估
```bash
./scripts/train.bash \
    --stage eval \
    --model configs/s2s_vit.py \
    --gpu 4 \
    --members 10 \
    --save
```

##### 使用PyTorch Lightning评估
```bash
bash scripts/train.bash \
    --stage leval \
    --model configs/s2s_vit.py \
    --gpu 2 \
    --members 8
```

#### 3. 导出ONNX模型

```bash
# 导出为ONNX格式
./scripts/train.bash \
    --stage export \
    --model configs/s2s_vit.py

# 导出输出位置: ./output/6_6_12_12-s2s_vit-0.0005/model.onnx
```

### 参数说明

```bash
--stage              # 运行阶段: train(训练)/eval(评估)/export(导出)/ltrain(Lightning训练)/leval(Lightning评估)
--model, -m          # 模型配置文件路径
--lr, -l             # 学习率 (默认: 0.0005)
--depth-list, -d     # 模型深度配置,逗号分隔 (默认: "6,6,12,12")
--gpu, -g            # GPU数量 (默认: 1)
--batch, -b          # 每GPU批大小 (默认: 1)
--members            # 集合成员数(评估用) (默认: 8)
--save               # 保存成员平均结果
--nnodes, -n         # 节点总数 (默认: 1)
--master-addr, -a    # 主节点地址 (默认: 127.0.0.1)
--verbose, -v        # 详细日志
--help, -h           # 显示帮助信息
```

---

## 损失函数和指标

### 支持的损失函数

| 损失函数 | 描述 | 用途 |
|---------|------|------|
| `Charbonnier` | 鲁棒L1损失 | 主要预报损失 |
| `CRPS` | 连续秩概率评分 | 集合预报评估 |
| `KL散度` | Kullback-Leibler散度 | 潜变量约束 |

### 评估指标

| 指标 | 描述 |
|-----|------|
| `RMSE` | 均方根误差 |
| `TCC` | 时间相关系数 |

---

## 相关文献

该项目基于以下研究工作：

- **FuXi**: 伏羲模型
  - 论文: [FuXi: A Large-Scale Climate Foundation Model for Medium-Range Weather Forecasting](https://arxiv.org/abs/2306.12873)
  
- **SwinTransformer**: 窗口注意力机制
  - 论文: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
  
- **Mamba**: 状态空间模型
  - 论文: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  - 论文: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)

---

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 联系方式

- **项目负责人**: XMetAI Team
- **邮箱**: douzsh@gmail.com

---

## 快速命令参考

```bash
# 训练
./scripts/train.bash --stage train --model configs/s2s_vit.py --lr 0.0005 --gpu 1

# 评估
./scripts/train.bash --stage eval --model configs/s2s_vit.py --members 10 --save

# 导出
./scripts/train.bash --stage export --model configs/s2s_vit.py

# Lightning训练
./scripts/train.bash --stage ltrain --model configs/s2s_vit.py --gpu 2

# Lightning评估
./scripts/train.bash --stage leval --model configs/s2s_vit.py --members 8
```

---

**最后更新**: 2025年10月
**版本**: 0.0.1