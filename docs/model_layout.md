# Model layout and import migration

Each model family owns a package and exports its public classes from that
package. `xmetai.registries.import_builtin_registrations()` discovers the
packages and executes their registration decorators. `xmetai.models` no longer
re-exports individual classes. Discovery also includes the teaching models in
`xmetai.models.tutorials`.

| Previous import | Public import |
| --- | --- |
| `xmetai.models.group_vae` | `xmetai.models.tongyun` |
| `xmetai.models.puyun` | `xmetai.models.puyun` |
| `xmetai.models.tianquan` | `xmetai.models.tianquan` |
| `xmetai.models.afnonet` | `xmetai.models.afno` |
| `xmetai.models.diffusion_forecast` | `xmetai.models.dit` |
| `xmetai.models.correction.CorrectionModel` | `xmetai.models.adapter.ForecastAdapter` |
| `xmetai.models.onnx_infer` | `xmetai.models.infer` |
| `xmetai.models.vit` | `xmetai.models.tutorials.vit` |
| `xmetai.models.vit_ens` | `xmetai.models.tutorials.vit_ens` |
| `xmetai.models.base_predictor` | `xmetai.core.base_predictor` |
| `xmetai.models.meta` | `xmetai.core.meta` |
| `xmetai.models.ema` | `xmetai.core.ema` |
| `xmetai.models.utransformer` | `xmetai.layers.utransformer` |
| `xmetai.models.pred_head` | `xmetai.layers.pred_head` |

The feature-grouped public model and registry key are `TongYun` (formerly
`FeatureGroupVAEPredictor`). The level-grouped `GroupVAEPredictor`, its grouping
helpers, config and dedicated documentation are removed. The public S2S config
is `configs/s2s_tongyun.py`; it uses feature groups without legacy group-index
freezing or automatic legacy pretrained-checkpoint selection. Architecture and
training details are documented in [TongYun](tongyun/README.md).

TongYun no longer exposes spectral band-power loss options. Use `amse_weight`
and `amse_image_size` for AMSE on the native output grid. Shared spectral
transforms required by AMSE and fidelity metrics remain available.

The correction model registry key changes to `ForecastAdapter`; update custom
configs accordingly. There are no compatibility import modules.

The GraphCast predictor, its configs and dedicated tests are removed. GraphCast
data utilities, layers and training recipe entries remain available for custom
models. Configurations remain in the flat `configs/` directory.

Directory moves and the feature-grouped class rename do not rename parameter
attributes or change state-dict keys. This does not make a legacy level-grouped
GroupVAE checkpoint compatible with the feature-grouped TongYun architecture. Saved
configs containing fully qualified old Python paths must be updated according to
the table before instantiation; full pickled Python models also depend on import
paths. Load existing state dictionaries into the corresponding new model class.

## Reference architectures

These implementations adapt the architectures to XMetAI's normalization, losses
and rollout contracts; they are not claims of exact reproduction or official
pretrained weights.

- PuYun: https://arxiv.org/abs/2409.02123
- TianQuan-S2S: https://arxiv.org/abs/2504.09940
- AFNO: https://arxiv.org/abs/2111.13587
- FourCastNet weather application: https://arxiv.org/abs/2202.11214
