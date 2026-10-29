# TongYun

TongYun partitions weather variables into groups, learns a latent representation
for each group, and advances all groups through a shared dynamics predictor.
The public class is `xmetai.models.tongyun.TongYun`, registered as
`MODELS.get("TongYun")`. The S2S recipe is
[s2s_tongyun.py](../../configs/s2s_tongyun.py).

## Variable groups, shared dynamics

![TongYun variable groups and shared dynamics](../assets/tongyun_arch.png)

The supplied schematic illustrates the feature-grouped architecture. The three
rows have separate prior encoders and decoders; the central predictor couples
all groups, and the merged forecast becomes the next autoregressive input.
The illustrated GatedDeltaNet-2 mixer is selected through `vt_layer_type`;
other supported mixer choices use the same grouped model contract. The daily
feedback shown in the figure corresponds to the 24-hour S2S recipe; `freq`
controls the actual forecast cadence.

The default groups are defined by `GROUP_VAR_MAP` and `GROUP_RECON_MODES` in
[tongyun.py](../../xmetai/models/tongyun/tongyun.py).

| Group | Fields | Input representation | Output reconstruction |
| --- | --- | --- | --- |
| State-tendency (`residual_state`, zt in the figure) | Geopotential and temperature across pressure levels, plus selected surface thermodynamic fields | Previous state and state difference | Residual |
| Wind state (`abs_wind`, uv) | Pressure-level and near-surface wind | Two state frames | Absolute |
| Scalar state (`abs_scalar`, q) | Specific humidity, precipitation, radiation and cloud fields | Two state frames | Absolute |

For a two-frame input, the state-tendency group encodes
`[x_previous, x_current - x_previous]`; the other groups encode
`[x_previous, x_current]`. Forecast residuals are added to the last input frame,
while absolute outputs are used directly. All groups are reassembled into the
original channel order before autoregressive feedback. These operations use
the normalized representation; BasePredictor owns physical units, logarithmic
transforms, masks and evaluation contracts.

Grouping is an explicit model-design choice. Custom `group_var_map` and
`group_recon_modes` dictionaries must assign each channel exactly once and use
whole-group absolute/residual modes. The model has no level-group compatibility
branch or per-channel residual override.

## Training and inference

Each group has prior and posterior encoders and a decoder shared by the
reconstruction and forecasting branches. The posterior teacher window includes
the true next frame during training. Inference uses only the prior path:
encode the input window, couple the grouped latents, decode, merge, and feed
the result into the next window.

The training progression in the schematic is supported by the existing model
and configuration controls:

1. **Pretrain:** learn posterior representations through reconstruction,
   augmented views, latent metric learning and latent regularization.
2. **One-step alignment:** train the prior forecasting path against the next
   field and posterior teacher latents; freeze the teacher when configured.
3. **Long rollout:** train autoregressive trajectories, optionally freezing
   selected feature-group modules through `train.freeze`.

Group freezing is a recipe decision. The public feature-grouped S2S recipe does
not inherit the removed height-group freezing indices. The figure's reference
to per-variable SDXL-VAE validation describes the supplied design rationale;
this refactor does not reproduce those experiments or add forecast-skill results.

## Losses

The existing reconstruction, latent regularization, latent prediction and
optional field-dynamics objectives are retained. `amse_weight` enables AMSE;
`amse_image_size` specifies the native output grid when it differs from the
encoder grid, otherwise `image_size` is used. Spectral band-power loss is no
longer a TongYun training objective. Its shared spectral transform remains
available internally to AMSE and forecast-fidelity metrics.

## Migration

`FeatureGroupVAEPredictor` is renamed to `TongYun`, without an alias. The
feature-grouped parameter structure is retained; load state dictionaries using
the same architecture options and update serialized config targets and imports.

The earlier level-grouped `GroupVAEPredictor`, its S2S config and its dedicated
documentation are removed. Its mixed per-channel residual semantics are not
emulated by TongYun. Earlier code, results and checkpoint references remain in
Git history; they are not results for this feature-grouped recipe.
