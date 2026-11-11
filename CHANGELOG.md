# Changelog

## 1.0.0

First major release of XMetAI Core.

- PyTorch weather and climate prediction library with LazyConfig recipes and CLI training entry points.
- Model families including TongYun, PuYun, TianQuan, AFNO, and diffusion/flow predictors.
- Distributed training, ensemble evaluation, and optional Lightning/ONNX workflows.
- Aligned English and Chinese READMEs, dependency installation instructions, software citation, and FLA/Mamba acknowledgements.
- TongYun S2S recipe explicitly selects `vt_group_norm_mode="norm"` instead of the model default `"none"`. This changes the recipe's latent normalization behavior; loading existing weights does not by itself verify equivalent forecasts. Re-evaluate existing checkpoints with the selected configuration.

This describes the release scope, not a claim of completed GPU training or export validation.
