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
├── xmetai/                       # 核心库（模型、数据、训练、配置等）
├── configs/                      # 训练/推理配置（LazyConfig）
├── tools/                        # 训练与导出脚本
├── scripts/                      # 辅助脚本（train.bash、run_tests.sh 等）
├── tests/                        # 单元与集成测试
├── README.md                     # 项目说明
└── CONTRIBUTING.md               # 贡献指南
```

---

## 快速开始

### 环境配置

```bash
# 安装开发依赖（含 lint/test 工具）
pip install -e .[dev]
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

#### 2. 模型评估

##### 多GPU评估
```bash
./scripts/train.bash \
    --stage eval \
    --model configs/s2s_vit.py \
    --gpu 4 \
    --members 10 \
    --save
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

## 🧪 测试和代码质量

本项目包含全面的测试套件，涵盖**50+ 个测试**，分为**单元测试**和**集成测试**两个部分。

### 📊 测试统计

```
✅ 50+ 测试用例
📦 单元测试: 37 个（5 个文件）
🔗 集成测试: 13 个（端到端工作流）
🔄 支持 CPU/GPU 双路径
📈 核心模块覆盖率: 69%-99%
```

### 🚀 快速开始

使用便捷脚本 `scripts/run_tests.sh`：

```bash
# GPU 测试（默认，包含覆盖率检查）
bash scripts/run_tests.sh gpu

# CPU 测试（跳过 GPU 标记的测试）
bash scripts/run_tests.sh cpu

# 核心单元测试子集
bash scripts/run_tests.sh unit

# 完整覆盖率报告（生成 htmlcov/）
bash scripts/run_tests.sh coverage

# 所有测试
bash scripts/run_tests.sh all
```
### 📂 测试目录结构

```
tests/                      # 测试根目录
├── unit/                   # 单元测试 (37 个测试)
│   ├── test_layers_basic.py      # 11 个测试 - 基础层
│   ├── test_layers_vision.py     # 14 个测试 - 视觉层
│   ├── test_models.py            # 13 个测试 - 模型架构
│   ├── test_loss_metrics.py     # 14 个测试 - 损失和指标
│   └── test_data.py              # 8 个测试 - 数据加载
├── integration/            # 集成测试 (13 个测试)
│   └── test_integration.py       # 端到端工作流
├── conftest.py             # 共享 fixtures 和配置
└── runner.py               # Python 测试运行器
```
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

**最后更新**: 2026年01月
**版本**: 0.1.0
