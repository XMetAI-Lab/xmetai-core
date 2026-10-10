# GT 五档加权重建 loss

已统计的全域年度分位数保存在 `xmetai/recipes/gt_loss.py`，是普通 Python 数值配置。包含 ERA/CRA S2S、ERA/CRA 6 小时、ERA/CRA 0.25° 日数据和 ERA5-Land 七组配置。数值从 2026-09-26 的 `global_summary.json` 与归一化元数据一次性整理，运行时不依赖统计文件或数据库。

`configs/s2s_base.py` 暴露 `gt_loss_profiles`，默认选择 ERA S2S 的 `gt_loss_features`。其他数据集的子配置明确选择对应条目，例如：

```python
# CRA S2S 子配置
from .s2s_base import gt_loss_profiles

gt_loss_features = gt_loss_profiles["cra_s2s"].features
```

也可以直接导入纯 Python 配置：

```python
from xmetai.loss import GTWeightedLoss
from xmetai.recipes.gt_loss import gt_loss_profiles

features = gt_loss_profiles["cra_6h"]["features"]
criterion = GTWeightedLoss(in_names, mean, std, features)
loss_rec = criterion(output, target, weight=field_weight, mask=field_mask)
```

`output/target` 为归一化后的 `[B, T, C, H, W]`，`mean/std` 与这些通道一一对应。不需要月份、经纬度或统计目录。`weight` 和 `mask` 可省略；海温、土壤等有有限填充值的要素应传入有效区域掩码。

每个要素的配置只有方向、阈值和可选风分量名，例如：

```python
# CRA 6 小时实测 P20/P40/P60/P80 风速，单位 m/s。
wind10 = dict(mode="upper", thresholds=[2.56805205, 4.56272602, 6.47621489, 8.90651321], channels=("u10", "v10"))
```

默认降水、风速和最高温关注高值，气压和最低温关注低值，气温、海温、土壤温度和土壤湿度关注两端。修改配置中的 `mode` 和相应阈值即可改变目标；单侧四个阈值对应 P20/40/60/80，双侧八个阈值对应 P10/20/30/40/60/70/80/90。

阈值单位为反标准化后的训练变量单位：温度 K、气压 Pa、风速 m/s；降水保持 `log1p(原生降水单位)`，不额外换算毫米。loss 根据现有 mean/std 还原 GT 后查档；风分量先还原再求模，两个分量共享权重。

权重固定为 1–5，严格比较保证相同值同档，重复阈值允许出现空档。未配置的通道保持权重 1。权重乘在逐点 Charbonnier 误差外，mean 除以有效元素数，不除以 GT 权重和。无效 GT/掩码点不参与归约，全部无效时返回零及零梯度。

ESFM 可合并大气和陆面配置的 `features` 字典。阈值按数据集区分，不能把 ERA 的条目直接套到 CRA。配置记录了各自的统计区间；用于训练时需注意与独立测试期的重叠。这些是全域年度阈值，并非此前生成的逐格点月度阈值。

本次提供 loss 和配置入口，未自动替换各模型已有的重建目标。接入时可替换原重建项，避免与旧降水强度加权项重复叠加；不要直接用于加权 CRPS。
