# GroupVAE

GroupVAE is a grouped latent state-space forecasting model for high-dimensional dynamical systems whose state variables can be organized into meaningful groups, such as pressure levels, variable families, spatial components, or other physical partitions. It learns a structured latent state, predicts the temporal evolution of that state, and decodes the evolved latent representation back to the original state space.

## 1. Model Architecture

![GroupVAE architecture](../assets/group_vae_arch.svg)

The central design choice is to decompose the full state into meaningful groups before latent encoding. Instead of compressing all variables into a single representation, GroupVAE assigns separate encoders and decoders to different groups, then learns a shared latent state transition predictor to exchange information across groups and advance the system state. In weather forecasting, these groups can be pressure levels or variable families; in other state-space forecasting problems, they can be any domain-specific partition of the state vector.

For an input history and target rollout,

$$
X_t \in \mathbb{R}^{B \times T_{\mathrm{in}} \times C \times H \times W},
\qquad
Y_t \in \mathbb{R}^{B \times K \times C \times H \times W}.
$$

The channel dimension is partitioned into groups:

$$
x_t = \{x_t^{(g)}\}_{g=1}^{G},
\qquad
\sum_{g=1}^{G} C_g = C.
$$

In the level-wise S2S setting, the groups are organized as follows.

| Group | Physical scope |
| --- | --- |
| 0 | Near-surface and 1000 hPa variables |
| 1 | Lower-to-middle pressure levels from 925 hPa to 400 hPa |
| 2 | Upper pressure levels from 300 hPa to 50 hPa |
| Optional | Slow surface variables, such as sea-surface temperature, when surface splitting is enabled |

Each group has a prior encoder, a posterior encoder, and a group-specific decoder shared by the reconstruction and forecasting branches. The prior encoder maps the current autoregressive input window into a group latent state. The posterior encoder maps a teacher-forced window, which includes the true next frame, into a target latent state. The decoder maps each latent state back to field space.

The latent state transition predictor is the module that explicitly couples groups. It receives all prior latent states and a timestamp condition, then predicts the next latent state for every group:

$$
\hat{h}_{t+1}^{(g)}
= \Phi_g(
h_{t,\mathrm{prior}}^{(1)}, \ldots, h_{t,\mathrm{prior}}^{(G)}, c_{t+1}
).
$$

Here, $\Phi_g$ denotes the group-specific output of the learned state transition operator, and the timestamp condition contains forecast step, hour, and day-of-year information. After decoding, the group outputs are reassembled into the original full-field channel order. At inference time, only the autonomous prior path is used: encode the current input, predict the next latent state, decode, reassemble, and roll the prediction into the next input window.

## 2. Training Stages

Training is organized as a staged curriculum. The goal is to first learn a stable latent state space, then align the autonomous state transition with that space, and finally train longer autoregressive rollouts without damaging selected pretrained group dynamics.

### 2.1 Pretrain

The pretraining stage focuses on posterior state-space learning. The model reconstructs both clean and augmented inputs, while a contrastive latent objective encourages stable representations under channel masking and temporal mixing.

This stage optimizes three objectives:

| Objective | Purpose |
| --- | --- |
| Reconstruction | Make the posterior encoder and decoder recover state fields accurately |
| Latent metric learning | Encourage augmented views of the same state to remain close in latent space |
| Latent regularization | Control latent magnitude and reduce collapse risk |

Conceptually, pretraining gives the posterior branch a well-conditioned target manifold. Later stages can then train the prior encoder and state transition predictor to approach this manifold without direct access to future observations.

The pretraining objective can be summarized as

$$
\begin{aligned}
\mathcal{L}_{\mathrm{pre}}
&= \rho(\hat{x}_{t}, x_t) + \lambda_{\mathrm{aug}}\rho(\hat{x}_{t,\mathrm{aug}}, x_t) \\
&\quad + \lambda_{\mathrm{metric}}
\mathcal{L}_{\mathrm{NCE}}(h_{\mathrm{clean}}, h_{\mathrm{aug}}) \\
&\quad + \lambda_{\mathrm{reg}}\frac{1}{G}\sum_{g=1}^{G}
\big[\mathrm{mean}((h_{\mathrm{clean}}^{(g)})^2) + \mathrm{mean}((h_{\mathrm{aug}}^{(g)})^2)\big].
\end{aligned}
$$

Here, $\rho$ is the robust reconstruction loss. This objective reconstructs the latest frame in the input window and regularizes both clean and augmented posterior latents.

### 2.2 One-Step

After pretraining, the posterior encoder can be frozen. This turns it into a fixed teacher that defines the target latent state for the next observed frame. The trainable prior encoder and state transition predictor must learn to reach that teacher latent state from the current input window.

The stage uses three complementary field-space branches:

| Branch | Supervision target | Role |
| --- | --- | --- |
| State-space forecasting | Next target frame | Trains the autonomous prediction path |
| Input autoencoding | Last input frame | Keeps the prior encoder-decoder path reconstructive |
| Target autoencoding | True next frame | Keeps the frozen posterior target interpretable in field space |

The posterior teacher window is formed by shifting the input window forward and inserting the true next frame:

$$
\tilde{X}_{t+1}
= [x_{t-T_{\mathrm{in}}+2}, \ldots, x_t, x_{t+1}].
$$

The key latent-space objective is state prediction consistency:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{state}}
&= \sum_{k \in \mathcal{S}} w_k
\sum_{g=1}^{G}
\rho(
\hat{h}_{t+k}^{(g)},
\mathrm{sg}(h_{t+k,\mathrm{post}}^{(g)})
).
\end{aligned}
$$

The stop-gradient operator makes the posterior branch a fixed target. This prevents the teacher latent space from moving to accommodate weak prior predictions, and instead forces the prior encoder and state transition predictor to learn meaningful one-step dynamics.

### 2.3 Rollout

The rollout stage extends training from one-step prediction to longer autoregressive forecasts. Encoders for selected groups can be frozen in the prior branch, the posterior branch, or both, which preserves their pretrained dynamics and reduces GPU memory use. In the current level-wise S2S setting, the frozen groups correspond to the middle and upper atmosphere, where preserving pretrained latent dynamics is especially important for stable long-horizon forecasts.

Only the remaining trainable components are adapted during rollout. This gives the model enough flexibility to improve autoregressive behavior while reducing the risk of overwriting the pretrained vertical latent structure.

Rollout training also uses scheduled teacher-forcing windows. The current level-wise S2S rollout starts supervised expansion at step 9 and ends at step 20, with the forecast dataset length matched to the 20-step horizon. Long rollouts may use sparse loss-step supervision: at most 10 loss steps are retained, the first and last supervised steps are preserved when applicable, and step weights are normalized.

## 3. S2S Config Settings & Final Result

This section reports one concrete [S2S forecasting setting](../../configs/s2s_group_vae.py) and result. It should be read as an experiment configuration, not as a limitation of the GroupVAE architecture.

The S2S configuration uses a level-wise GroupVAE with a 120 by 240 spatial grid, two input frames, one output frame per model step, 24-hour temporal frequency, and a maximum supervised rollout loss subset of 10 steps. When loaded from a pretrained checkpoint, the rollout curriculum trains lead steps 9 through 20, halves the base training iteration budget, and freezes the middle and upper atmosphere encoder groups. The experiment uses three level groups: near-surface, lower-to-middle atmosphere, and upper atmosphere. Residual prediction is applied selectively to slow or anomaly-like variables, such as sea-surface temperature and two-meter temperature.

The main weighted training objective combines five terms:

| Term | Relative role |
| --- | --- |
| Forecast reconstruction | Main prediction accuracy objective |
| Prior autoencoding | Regularizes the autonomous input representation |
| Posterior autoencoding | Maintains the teacher latent representation in field space |
| Latent regularization | Stabilizes latent magnitude |
| State prediction consistency | Aligns predicted prior latents with posterior teacher latents |

In the S2S setting, forecast reconstruction receives the largest direct field-space weight, while state prediction consistency is also strongly weighted. The autoencoding terms are kept as auxiliary constraints, and latent magnitude regularization is intentionally small.

### Final Result Snapshot

Increasing rollout training improves long-lead Z500 RMSE, especially after the second forecast week.

```mermaid
---
config:
    xyChart-beta:
        width: 1024
        showDataLabelOutsideBar: true
        showDataLabel: true
        showTick: true
    themeVariables:
        xyChart-beta:
            titleColor: "#ff0000"
---
xychart-beta
    title "Z500 RMSE vs Lead Time"
    x-axis "Day" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
    y-axis "RMSE"
    line "rollout=2" [58.185,110.927,181.302,261.893,350.373,440.273,526.271,606.893,677.093,734.91,785.813,829.374,864.607,899.464,931.553,952.612,962.09,975.724,987.525,997.607,1007.201,1018.602,1023.441,1024.554,1027.937,1029.199,1026.305,1026.065,1026.193,1023.706,1017.886,1017.163,1020.252,1019.469,1017.551,1021.623,1024.504,1025.818,1030.097,1033.385,1032.849,1032.13]
    line "rollout=20" [58.698,111.73,182.4,262.336,349.422,433.52,508.282,573.676,622.459,658.413,688.951,712.943,729.36,745.136,759.273,767.669,771.289,777.402,781.826,783.137,786.266,791.324,793.442,793.236,796.472,799.913,800.485,803.177,808.347,810.768,811.018,813.51,815.964,816.485,819.435,823.732,825.483,824.583,825.506,827.032,826.029,827.011]
```
Selected channel RMSE values with different lead times are listed below.

|   Times |    z500 |   u200 |   u850 |     msl |   t2m |   sst |    ttr |    tp |
|--------:|--------:|-------:|-------:|--------:|------:|------:|-------:|------:|
|      24 |  58.698 |  1.647 |  1.104 |  92.659 | 0.681 | 0.161 | 11.231 | 0.212 |
|      48 | 111.730 |  2.397 |  1.569 | 136.790 | 0.860 | 0.210 | 13.566 | 0.231 |
|      72 | 182.400 |  3.245 |  2.089 | 198.597 | 1.044 | 0.246 | 15.240 | 0.245 |
|      96 | 262.336 |  4.189 |  2.633 | 269.966 | 1.244 | 0.274 | 16.774 | 0.258 |
|     120 | 349.422 |  5.189 |  3.160 | 346.951 | 1.458 | 0.306 | 18.353 | 0.268 |
|     144 | 433.520 |  6.197 |  3.625 | 419.580 | 1.654 | 0.329 | 19.745 | 0.277 |
|     168 | 508.282 |  7.133 |  4.000 | 480.893 | 1.843 | 0.351 | 20.928 | 0.287 |
|     192 | 573.676 |  7.969 |  4.310 | 530.084 | 1.989 | 0.378 | 21.928 | 0.292 |
|     216 | 622.459 |  8.646 |  4.533 | 565.602 | 2.102 | 0.397 | 22.806 | 0.298 |
|     240 | 658.413 |  9.147 |  4.699 | 596.894 | 2.196 | 0.415 | 23.434 | 0.302 |
|     264 | 688.951 |  9.529 |  4.841 | 623.248 | 2.266 | 0.433 | 23.928 | 0.306 |
|     288 | 712.943 |  9.880 |  4.941 | 639.274 | 2.330 | 0.455 | 24.416 | 0.308 |
|     312 | 729.360 | 10.116 |  5.026 | 650.999 | 2.377 | 0.471 | 24.725 | 0.308 |
|     336 | 745.136 | 10.313 |  5.092 | 665.210 | 2.403 | 0.486 | 24.951 | 0.312 |
|     360 | 759.273 | 10.462 |  5.132 | 675.673 | 2.432 | 0.506 | 25.141 | 0.312 |
|     384 | 767.669 | 10.607 |  5.158 | 677.821 | 2.462 | 0.520 | 25.369 | 0.314 |
|     408 | 771.289 | 10.720 |  5.192 | 681.786 | 2.485 | 0.533 | 25.489 | 0.314 |
|     432 | 777.402 | 10.766 |  5.219 | 687.750 | 2.500 | 0.546 | 25.561 | 0.315 |
|     456 | 781.826 | 10.846 |  5.234 | 689.031 | 2.517 | 0.564 | 25.747 | 0.315 |
|     480 | 783.137 | 10.918 |  5.252 | 689.548 | 2.528 | 0.575 | 25.808 | 0.314 |
|     504 | 786.266 | 10.987 |  5.267 | 694.295 | 2.541 | 0.586 | 25.907 | 0.316 |
|     528 | 791.324 | 11.027 |  5.279 | 699.271 | 2.559 | 0.603 | 26.015 | 0.315 |
|     552 | 793.442 | 11.116 |  5.291 | 698.003 | 2.575 | 0.613 | 26.079 | 0.314 |
|     576 | 793.236 | 11.214 |  5.312 | 699.868 | 2.592 | 0.623 | 26.158 | 0.315 |
|     600 | 796.472 | 11.239 |  5.328 | 702.775 | 2.602 | 0.633 | 26.154 | 0.315 |
|     624 | 799.913 | 11.296 |  5.334 | 702.177 | 2.615 | 0.648 | 26.265 | 0.315 |
|     648 | 800.485 | 11.378 |  5.346 | 702.191 | 2.625 | 0.658 | 26.247 | 0.314 |
|     672 | 803.177 | 11.459 |  5.356 | 706.223 | 2.637 | 0.667 | 26.296 | 0.316 |
|     696 | 808.347 | 11.494 |  5.371 | 710.409 | 2.653 | 0.682 | 26.432 | 0.316 |
|     720 | 810.768 | 11.560 |  5.372 | 709.219 | 2.664 | 0.691 | 26.447 | 0.316 |
|     744 | 811.018 | 11.639 |  5.389 | 710.758 | 2.678 | 0.699 | 26.515 | 0.316 |
|     768 | 813.510 | 11.636 |  5.403 | 712.732 | 2.687 | 0.709 | 26.547 | 0.316 |
|     792 | 815.964 | 11.664 |  5.406 | 712.119 | 2.696 | 0.723 | 26.666 | 0.317 |
|     816 | 816.485 | 11.716 |  5.413 | 712.477 | 2.701 | 0.731 | 26.655 | 0.316 |
|     840 | 819.435 | 11.762 |  5.425 | 716.318 | 2.712 | 0.739 | 26.723 | 0.318 |
|     864 | 823.732 | 11.775 |  5.445 | 719.874 | 2.725 | 0.753 | 26.864 | 0.318 |
|     888 | 825.483 | 11.830 |  5.442 | 718.360 | 2.736 | 0.761 | 26.879 | 0.317 |
|     912 | 824.583 | 11.895 |  5.455 | 718.816 | 2.749 | 0.769 | 26.929 | 0.318 |
|     936 | 825.506 | 11.882 |  5.467 | 719.590 | 2.752 | 0.777 | 26.926 | 0.318 |
|     960 | 827.032 | 11.890 |  5.470 | 718.135 | 2.750 | 0.791 | 27.003 | 0.318 |
|     984 | 826.029 | 11.930 |  5.467 | 716.972 | 2.749 | 0.799 | 26.970 | 0.318 |
|    1008 | 827.011 | 11.952 |  5.474 | 719.771 | 2.754 | 0.806 | 26.988 | 0.318 |

The final checkpoint referenced by the s2s forecasting experiment is available from [Quark Cloud](https://pan.quark.cn/s/d8931703aee5?pwd=2pEV).
