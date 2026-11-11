# XMetAI Core

[English](README.md) | [简体中文](README.zh-CN.md)

XMetAI Core is a PyTorch framework for research, training, evaluation, and ONNX export of weather and climate prediction models, developed by XMetAI Lab.

Release version: **1.0.0**. See the [changelog](CHANGELOG.md).

## Features

- LazyConfig configuration and reusable training recipes.
- Single- and multi-GPU training, with distributed settings defined by the selected recipe.
- Deterministic and ensemble forecasting, probabilistic losses, and weather evaluation metrics.
- Optional Lightning training, ONNX export, and ONNX Runtime inference.

## Installation

Requires **Python 3.9+** and **PyTorch 2.7+**. GPU features require a compatible device and runtime. These are declared minimums, not a claim that every version combination has been tested.

From a source checkout, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Development tools
pip install -e '.[dev]'
# Install only the optional features you need
pip install -e '.[lightning,export,inference]'
```

The `gpu` extra installs Mamba (`mamba-ssm`) for models that need it; it does not install GPU drivers. The `inference` extra installs `onnxruntime-gpu`; a CPU-only runtime can be installed separately as `onnxruntime`. `pyproject.toml` is the dependency source of truth.

## Data and configuration

Training requires separately prepared data and statistics; datasets and trained weights are not bundled. Existing weather recipes use Zarr data with companion `mean.npy`, `std.npy`, and `weight.npy` files. Review the selected dataset implementation and configure data paths, variable order, units, resolution, time steps, and static fields before running. Do not assume all recipes use the same layout.

`configs/s2s_vit.py` is a tutorial entry point; `configs/s2s_tongyun.py` is the TongYun S2S recipe. Configurations may switch stages when checkpoints exist: inspect the recipe and output directory before resuming.

## Training, evaluation, and export

Run from the repository root after configuring data paths:

```bash
bash scripts/train.bash --stage train -m configs/s2s_vit.py --gpu 1 --batch 1
bash scripts/train.bash --stage eval -m configs/s2s_vit.py --members 8
bash scripts/train.bash --stage export -m configs/s2s_vit.py
bash scripts/train.bash --help
```

Export requires the `export` extra and an export-capable model. Checkpoint loading and output paths follow the selected configuration. The wrapper generates `configs/tmp_config.py`; do not edit that file manually. For direct CLI usage:

```bash
xmetai-train --help
xmetai-train --config-file configs/s2s_vit.py
# Requires the lightning extra
xmetai-train-lightning --config-file configs/s2s_vit.py
```

Repository configurations, scripts, and documentation are accessed from a source checkout; the wheel installs the `xmetai` library and CLI entry points.

## Models and repository layout

| Path | Purpose |
| --- | --- |
| `xmetai/models/tongyun/` | [TongYun](docs/tongyun/README.md) grouped latent forecasting |
| `xmetai/models/puyun/` | PuYun autoregressive prediction |
| `xmetai/models/tianquan/` | TianQuan-S2S climatology-conditioned prediction |
| `xmetai/models/afno/` | AFNO spectral prediction |
| `xmetai/models/dit/` | Diffusion and flow matching predictors |
| `xmetai/models/adapter/` | Forecast correction adapters |
| `xmetai/models/infer/` | ONNX inference and ensemble evaluation |
| `xmetai/models/tutorials/` | ViT/ViTEns tutorial models |
| `xmetai/core/`, `xmetai/layers/` | Shared predictor contracts and network components |
| `xmetai/recipes/`, `configs/` | Reusable recipes and task configurations |
| `scripts/`, `tools/`, `tests/` | Launchers, data utilities, and tests |

Available losses include Charbonnier, CRPS variants, and KL divergence; metrics include RMSE and TCC. Availability and behavior depend on the model and recipe.

## Development and validation

```bash
pip install -e '.[dev]'
pre-commit run --all-files
pytest tests -q
```

Some checks require optional dependencies, GPU hardware, or external data. Report passed, skipped, and unverified checks separately; test counts and coverage should come from the current run.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules. Use a PR title such as `fix: describe the change [codex]` for Codex contributions.

## License and contact

Licensed under **GPL-3.0-or-later**; see [LICENSE](LICENSE). Data, weights, and third-party assets may have separate terms.

XMetAI Team: douzsh@gmail.com.

## Acknowledgements

XMetAI Core uses the following libraries for selected model components:

- [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention): Gated DeltaNet-2 layers and efficient linear attention kernels, used by `xmetai/layers/gated_deltanet2.py`.
- [Mamba](https://github.com/state-spaces/mamba): selective state space models and scan kernels, used by `xmetai/layers/mamba_vision.py`.

These components require the corresponding upstream libraries and compatible GPU environments. FLA-based recipes need an FLA version exposing `fla.layers.GatedDeltaNet2`; Mamba variants require their corresponding upstream implementations. Please follow each upstream project's installation instructions, license, and citation guidance.

## Citation

If you use XMetAI Core in your research, please cite this software:

```bibtex
@software{xmetai2026core,
  title   = {XMetAI Core: A Framework for Meteorological AI Model Training and Inference},
  author  = {Dou Zesheng, Hu jiahui, Zhengjinfang, Shenzhewen, Zhao Han, Zhou Shuang, Liu Junjie},
  url     = {https://github.com/XMetAI-Lab/xmetai-core},
  version = {1.0.0},
  year    = {2026}
}
```
