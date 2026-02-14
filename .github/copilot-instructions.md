# Copilot Instructions

## Architecture & Layout
- `xmetai/` is the reusable library (models, data loaders, trainers, hooks, metrics); `common/` holds shared LazyConfig fragments such as training hyper-params.
- `inference/` contains standalone ONNX runners (`inference_s2s.py` for single NetCDF, `batch_infer.py` for Zarr + multi-GPU IO binding) that mirror the training data layout.
- Support folders: `scripts/` (training launchers, test runner), `docker/` (developer images).

## Config & Model Patterns
- All experiments use LazyConfig; extend an existing config in `configs/` (e.g., `s2s_vit.py`) and import shared pieces from `s2s_base.py` instead of rewriting optimizers or dataloaders.
- `common/train.py` defines the canonical `train` dict (DDP/FSDP toggles, eval cadence, teacher forcing). Override values via config or by passing `key=value` overrides to `scripts/train.bash`.
- `configs/s2s_vit.py` demonstrates staged training: it inspects `output_dir` to decide whether to enable `train.ens_train`, tweak `teacher_forcing`, and switch to `ViTEnsModel` when ensemble checkpoints already exist.
- New architectures must be registered inside `xmetai/MODELS` and can rely on optional modules like `MambaVisionLayer`; `tools/train.py` auto-injects activation checkpointing for `SwinBlock`, `TransformerEncoderLayer`, and Mamba blocks.

## Training Workflow
- Primary entrypoint is `scripts/train.bash`; it clones the chosen config to `configs/tmp_config.py`, wires `DATA_DIR`, `BASE_CONFIG`, `OUTPUT_DIR`, updates `total_batch_size`, and toggles `export_onnx`/`save_dir` based on flags.
- `--stage train|eval|export|ltrain|leval` selects between standard Torch training, evaluation (prepends `--eval` and sets `members`), export, or Lightning variants. `--save` keeps per-member outputs; omit it to have the script `sed -i` away `save_dir` entries.
- OS detection is built in: macOS switches to `train.device_type='mps'` and disables DDP/FSDP, while Linux assumes CUDA paths such as `/workspace/data`.
- Pass any extra LazyConfig overrides after the parsed flags (e.g., `scripts/train.bash -m configs/s2s_vit.py train.max_iter=20000 train.use_fsdp=True`).

## Distributed Training
- Local multi-node runs reuse `torchrun`; set `--gpu`, `--batch`, `--nnodes`, and `--master-addr` in `scripts/train.bash`. The script exports `NODE_RANK`, `LOCAL_ADDR`, and `MASTER_PORT=29500` before launching.

## Runtime Behavior
- `tools/train.py` loads configs via `LazyConfig`, instantiates models with `instantiate(cfg.model)`, applies DDP/FSDP/FSDP2 depending on `cfg.train` flags, and wires hooks such as `BestCheckpointer`, `EvalHook`, and `TorchMemoryStats`.
- Setting `train.teacher_forcing.teach_iters > 0` switches the trainer to `ExpandTrainer`, enabling staged rollout training; otherwise `SimpleTrainer` is used. `train.use_ema` turns on EMA weight tracking that the trainer updates post-`checkpointer.resume_or_load`.
- Exports call `do_export`, which binds `cfg.train.export_kwargs`, rewrites `model.forward` to the named inference method, and automatically saves external ONNX tensor data when the 2 GB limit is exceeded.
- Mixed precision is controlled through `cfg.train.float32_precision` plus `cfg.train.fsdp.precision`; the base configs default to bf16 params/reductions with fp32 buffers.

## Data & Inference Expectations
- Training datasets are normalized Zarr stores with companion `mean/std/weight.npy`; `scripts/train.bash` picks directories like `s2s.1950-2024.c76`, `land.1950-2024.c8`, or ERA5/CMA CRA variants purely by inspecting the config filename.
- When mixing datasets (e.g., land + s2s), `DATA_DIR` becomes a colon-separated list consumed by the dataloader; make sure the config’s dataset parser can split and iterate this string.
- `inference/inference_s2s.py` handles single NetCDF samples, adds optional `step`/`doy` inputs if the exported ONNX expects them, masks SST with `land_to_nan`, and writes outputs per member/day under `--save_dir/member/XX`.
- `inference/batch_infer.py` performs multi-GPU batch inference with ONNX Runtime IO binding, re-applies inverse normalization in-place (`inplace_inv_normalize`) using `mean/std` from `.nc` or `.npy`, optionally masks SST via `mask.npy`, and can dump either member-wise cubes or member averages filtered to specific `--levels`.

## Conventions & Gotchas
- Never edit `configs/tmp_config.py` directly; `scripts/train.bash` regenerates it each run and mutates it (and `BASE_CONFIG`) via `sed -i` to adjust `total_batch_size`, logging level, and evaluation members.
- Ensemble training stages rely on filesystem side effects (`model_final.pth`, `model_best.pth` in `output_dir`); deleting these files resets the progression logic in configs such as `s2s_vit.py`.
- ONNX inference expects the last channel to be log-transformed precipitation; both training and inference helpers clamp-exponentiate that channel, so keep channel ordering consistent when modifying model heads.
