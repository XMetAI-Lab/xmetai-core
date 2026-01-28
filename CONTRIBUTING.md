# 贡献指南

感谢你对 xmetai-core 的关注与贡献！本文档旨在帮助你快速、高质量地完成改动，并顺利提交到主仓库。

## 项目速览

- 训练与库代码：
  - 可复用库在 [xmetai/](xmetai)（模型、数据、训练器、钩子、指标）。
  - 公共配置片段在 [common/](common)。
  - 标准训练脚本在 [scripts/train.bash](scripts/train.bash)。

## 开发环境

- Python：推荐 Python ≥ 3.8。
- 建议使用虚拟环境或 Conda 管理依赖。

本地开发安装（含格式化与 lint 工具）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

或使用 Conda：

```bash
conda create -n xmetai python=3.10 -y
conda activate xmetai
pip install -e .[dev]
```

## 代码风格与预提交钩子

本项目使用 Black + Ruff 统一代码风格，最大行宽 160。

首次在仓库根目录安装并启用 pre-commit：

```bash
conda run -n xmetai pre-commit install
conda run -n xmetai pre-commit run --all-files
```

注意：当前 pre-commit 钩子默认执行 Black 与 Ruff。类型检查（mypy）可选，如需启用可在 [.pre-commit-config.yaml](.pre-commit-config.yaml) 中移除 `files: ^$` 限制或在本地单独运行：

```bash
mypy xmetai
```

常见问题：

- Ruff 报告复杂度（C901）：库中个别核心函数允许较高复杂度；请优先清晰与可维护，必要时可重构拆分。
- `__init__.py` 中存在历史的通配导入；新代码请避免使用 `from ... import *`，并尽量显式导出。

## 测试

测试位于 [tests](tests)。在本地运行：

```bash
pytest tests -q
```

部分测试依赖可选组件（如数据集类）；未安装时会自动跳过。

## 训练与运行

- 标准训练入口：[scripts/train.bash](scripts/train.bash)。示例：

```bash
bash scripts/train.bash -m configs/s2s_vit.py train.max_iter=20000
```

- Slurm/HPC：使用 [scripts/slurm_train.bash](scripts/slurm_train.bash)。示例：

```bash
bash scripts/slurm_train.bash -n 4 -g 8 -p gpu --script configs/iwc_mamba.py -- --eval
```

注意：不要直接编辑自动生成的 [configs/tmp_config.py](configs/tmp_config.py)。

## 提交流程

1. Fork 主仓库到你的账户。
2. 创建功能分支：

```bash
git checkout -b feature/your-feature-name
```

3. 编码与增补测试。
4. 本地运行格式化与测试：

```bash
pre-commit run --all-files
pytest tests -q
```

5. Push 到你的 fork：

```bash
git push origin feature/your-feature-name
```

6. 提交 Pull Request 到主仓库，填写清晰的改动说明与验证方法。

## PR 要求

- 功能性改动必须包含对应的单元测试或集成测试。
- 保持公共 API 的兼容性；若需变更请在 PR 描述中明确说明迁移路径。
- 通过预提交检查（Black、Ruff）与本地测试。
- 文档与示例需同步更新（如新增脚本或配置）。
- PR 标题必须以类型前缀开头，并使用以下之一：
  - fix
  - feat
  - docs
  - test
  - tests
  - refactor
  - perf
  - style
  - build
  - ci
  - chore
  - revert

  示例：`fix: 修复数据加载崩溃`、`feat: 新增TCC指标`、`docs: 更新训练说明`。
  CI 会校验 PR 标题前缀，不符合将阻塞合并。

## 变更建议

- 配置：优先复用与扩展 [configs](configs) 现有 LazyConfig；公共超参请从 [common/](common) 引用。
- 模型：新架构需在 [xmetai/__init__.py](xmetai/__init__.py) 注册，并保持推理兼容（ONNX 导出在工具中使用）。
- 数据：遵循 Zarr 规范与伴随 `mean/std/weight.npy` 的约定；批量推理脚本依赖此布局。

## 许可证

本项目采用 GNU GPLv3 许可证，详见 [LICENSE](LICENSE)。提交代码即表示你同意你的贡献以 GPLv3 进行授权与再发布。

提示：请确保你贡献的代码不包含无法以 GPLv3 授权的第三方代码或资产；如引用第三方库，请遵循其许可证并在必要处标注。

## 致谢

感谢所有贡献者的时间与投入！如有问题或建议，欢迎在 Issues 中反馈。
