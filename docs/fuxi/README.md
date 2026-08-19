# FuXi-2.1 (project-native)

A downscaled reimplementation of the **FuXi-2.1** global weather forecasting
architecture, adapted to the project's 1.5-degree ERA5 dataset. It reproduces
the architecture so it can be trained **from scratch** on any channel layout;
the official 0.25-degree / 85-channel weights are **not** loaded.

> Reference: https://huggingface.co/tpys/fuxi-2.1 (Fudan University & SAIS).

## Architecture

FuXi-2.1 replaces FuXi-1.0's U-Transformer with a **flat single-trunk
Transformer**:

```
patch/const embed → Swin window attention (1-D RoPE) + adaLN(step/hour/doy) + SwiGLU FFN
                 → variable-aware multi-head decoder (pressure / surface / derived)
```

| Component | This implementation |
|---|---|
| Backbone | flat Swin trunk (`FuxiSwinBlock` × `depth`) |
| Position encoding | 1-D RoPE inside windowed attention (`FuxiWindowAttention`) |
| Conditioning | adaLN on lead step / time-of-day / day-of-year (`FuxiTimeEmbed`) |
| FFN | SwiGLU (`SwiGLUFFN`, ratio `8/3`) |
| Decoder | `FuxiVarHead`: pressure-level deconv head + surface/derived pixel-shuffle heads |

The module layout mirrors the released `.pt2` state_dict (`decoder.patch_embed`,
`decoder.layers.N.{norm1,attn,norm2,mlp,adaln}`, `decoder.norm_layer.scale_shift`,
`decoder.head.*`) so key names stay aligned if official weights are ever adopted.

## Downscaling from the official 0.25° config

| Property | Official | This config |
|---|---|---|
| Resolution | 0.25° (721×1440) | 1.5° (120×240) |
| Channels | 85 (65 PL + 20 SFC) | 76 (data-driven) |
| `embed_dim` | 1536 | 384 |
| `depth` | 30 | 8 |
| `num_heads` | 24 | 12 |
| `patch_size` | 6 | 2 (→ 60×120 tokens) |
| `window_size` | — | 6 (shift 3; divides 60 and 120) |

`embed_dim`/`depth`/`num_heads` are a small starting point; they scale up freely
in `configs/s2s_fuxi.py`. `window_size` **must** divide `feat_size =
image_size // patch_size`; with `(120, 240)` and patch 2 that is `(60, 120)`, so
valid choices are 4, 6, 10, 12, … (6 is the default).

## Variable grouping (the extension interface)

Channel grouping is driven entirely by `in_names` via `split_var_groups`:

- **pressure** — `{z,t,u,v,q}<level>` channels (e.g. `z500`, `u850`)
- **derived** — accumulated diagnostics `ssr / ssrd / fdir / ttr / tp`
- **surface** — everything else (`msl`, `t2m`, `sst`, `u10m`, …)

Each group feeds its own decoder head; outputs are concatenated and scattered
back to the canonical `in_names` order. **Adding surface variables therefore
only requires extending the dataset** — no model-code changes. The model raises
at construction if the groups do not cover exactly `out_chans`.

## Files

| File | Purpose |
|---|---|
| `xmetai/layers/fuxi_blocks.py` | `SwiGLUFFN`, `FuxiWindowAttention`, `FuxiSwinBlock`, `FuxiSwinLayer`, `FuxiTimeEmbed`, `FuxiPatchEmbed`, `FuxiSubpixelHead`, `FuxiVarHead` |
| `xmetai/models/fuxi.py` | `Fuxi2` (`@MODELS.register()`), `FuxiBackbone`, `split_var_groups` |
| `configs/s2s_fuxi.py` | training config (reuses `s2s_base`) |
| `xmetai/layers/__init__.py`, `xmetai/models/__init__.py` | exports |

## Training

```bash
python tools/train.py --config-file configs/s2s_fuxi.py
```

Key config choices:

- `out_frames=1`, `fcst_frames=1` — single-step autoregressive (FuXi-2.1 rollout).
- `zero_accum=True`, `accumid=accumid` — derived (radiation/precip) channels are
  zeroed before feedback, matching the official rollout (diagnostic fields are
  not propagated).
- `const_chans=2` — land-sea mask + orography embedded separately and added to
  the patch tokens.

## Notes / differences from official FuXi-2.1

- No official weights, `.pt2` loading, or inference script (`tools/fuxi_infer.py`).
- Uses the project's existing ERA5 dataset (`xmetai/data/weather`) — no dedicated
  `fuxi_dataset.py`.
- Surface/derived heads use a single `pixel_shuffle(patch_size)` stage; the
  official two-stage pixel shuffle is the `patch_size=6 = 2×3` factorization and
  degenerates to one stage at patch 2.
