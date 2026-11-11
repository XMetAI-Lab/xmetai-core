# XMetAI Core

[English](README.md) | [简体中文](README.zh-CN.md)

XMetAI Core 是由雄安气象人工智能创新研究院（XMetAI Lab）开发的 PyTorch 天气与气候预测框架，用于模型研究、训练、评估与 ONNX 导出。

发布版本：**1.0.0**。参见[变更记录](CHANGELOG.md)。

## 核心特性

- LazyConfig 配置系统与可复用训练配方。
- 单卡、多卡训练，分布式设置以所选配方为准。
- 确定性与集合预报、概率损失及气象评估指标。
- 可选的 Lightning 训练、ONNX 导出与 ONNX Runtime 推理。

## 安装

要求 **Python 3.9+**、**PyTorch 2.7+**。GPU 功能需要兼容的设备与运行环境。这些是声明的最低版本，不表示所有版本组合均经过测试。

在源码仓库中执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# 开发工具
pip install -e '.[dev]'
# 按需安装可选功能
pip install -e '.[lightning,export,inference]'
```

`gpu` 扩展为需要 Mamba 的模型安装 `mamba-ssm`，不会安装显卡驱动。`inference` 扩展安装 `onnxruntime-gpu`；仅 CPU 推理可单独安装 `onnxruntime`。依赖统一以 `pyproject.toml` 为准。

## 数据与配置

训练需要另行准备数据和统计量；仓库不附带数据集和训练权重。现有天气配方使用 Zarr 数据及配套的 `mean.npy`、`std.npy`、`weight.npy`。运行前请核对所选数据集实现，配置数据路径、变量顺序、单位、分辨率、时间步长与静态场，不能假定所有配方的数据布局相同。

`configs/s2s_vit.py` 是教学入口；`configs/s2s_tongyun.py` 是 TongYun S2S 配方。checkpoint 的存在可能触发配置中的阶段切换，续跑前请检查配方与输出目录。

## 训练、评估与导出

配置好数据路径后，在仓库根目录运行：

```bash
bash scripts/train.bash --stage train -m configs/s2s_vit.py --gpu 1 --batch 1
bash scripts/train.bash --stage eval -m configs/s2s_vit.py --members 8
bash scripts/train.bash --stage export -m configs/s2s_vit.py
bash scripts/train.bash --help
```

导出需要安装 `export` 扩展，且模型须支持导出。checkpoint 加载和输出路径以所选配置为准。包装脚本会生成 `configs/tmp_config.py`，不要手动编辑该文件。也可直接使用 CLI：

```bash
xmetai-train --help
xmetai-train --config-file configs/s2s_vit.py
# 需要 lightning 扩展
xmetai-train-lightning --config-file configs/s2s_vit.py
```

仓库配置、脚本与文档从源码检出目录访问；wheel 安装 `xmetai` 库及 CLI 入口。

## 模型与仓库结构

| 路径 | 用途 |
| --- | --- |
| `xmetai/models/tongyun/` | [TongYun](docs/tongyun/README.md) 分组潜变量预报 |
| `xmetai/models/puyun/` | PuYun 自回归预测 |
| `xmetai/models/tianquan/` | TianQuan-S2S 气候态条件预测 |
| `xmetai/models/afno/` | AFNO 频域预测 |
| `xmetai/models/dit/` | 扩散与 Flow Matching 预测器 |
| `xmetai/models/adapter/` | 预报订正适配器 |
| `xmetai/models/infer/` | ONNX 推理与集合评测 |
| `xmetai/models/tutorials/` | ViT/ViTEns 教学模型 |
| `xmetai/core/`、`xmetai/layers/` | 公共预测契约与网络组件 |
| `xmetai/recipes/`、`configs/` | 公共配方与任务配置 |
| `scripts/`、`tools/`、`tests/` | 启动脚本、数据工具与测试 |

可用损失包括 Charbonnier、CRPS 变体与 KL 散度；指标包括 RMSE、TCC。具体可用范围和行为以模型及配方为准。

## 开发与验证

```bash
pip install -e '.[dev]'
pre-commit run --all-files
pytest tests -q
```

部分检查需要可选依赖、GPU 或外部数据。请分别报告通过、跳过及未验证的检查；测试数量和覆盖率应以当前运行结果为准。

贡献规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。Codex 贡献的 PR 标题使用 `fix: 描述修改 [codex]` 等格式。

## 许可证与联系

采用 **GPL-3.0-or-later**，见 [LICENSE](LICENSE)。数据、权重和第三方资产可能有独立条款。

XMetAI Team：douzsh@gmail.com。

## 致谢

XMetAI Core 的部分模型组件使用了以下开源库：

- [Flash Linear Attention（FLA）](https://github.com/fla-org/flash-linear-attention)：提供 Gated DeltaNet-2 层及高效线性注意力算子，用于 `xmetai/layers/gated_deltanet2.py`。
- [Mamba](https://github.com/state-spaces/mamba)：提供选择性状态空间模型与 scan 算子，用于 `xmetai/layers/mamba_vision.py`。

这些组件需要安装对应的上游库，并配置兼容的 GPU 环境。FLA 配方需要提供 `fla.layers.GatedDeltaNet2` 的版本；不同 Mamba 变体需要对应的上游实现。安装、许可证及引用要求请遵循各上游项目的说明。

## 引用

如果你的研究使用了 XMetAI Core，欢迎引用本软件：

```bibtex
@software{xmetai2026core,
  title   = {XMetAI Core: A Framework for Meteorological AI Model Training and Inference},
  author  = {Dou Zesheng, Hu jiahui, Zhengjinfang, Shenzhewen, Zhao Han, Zhou Shuang, Liu Junjie},
  url     = {https://github.com/XMetAI-Lab/xmetai-core},
  version = {1.0.0},
  year    = {2026}
}
```
