# 1. 基本符号与数据结构

## 1.1 输入与分组

设时刻 ( t ) 的完整状态为：

$$
x_t \in \mathbb{R}^{C \times H \times W}
$$

按照变量物理属性划分为 ( G ) 个 group：

$$
x_t = {x_t^{(g)}}_{g=1}^{G}, \quad x_t^{(g)} \in \mathbb{R}^{C_g \times H \times W}
$$

每个 group 内进一步划分为两类变量：

$$
x_t^{(g)} = \big(x_t^{(g,a)}, x_t^{(g,r)}\big)
$$

其中：

* (a)：absolute variables（绝对量主导）
* (r)：residual variables（扰动主导）

---

## 1.2 显式定义三个核心算子

为避免符号歧义，定义：

### Encoder（逐 group）


$$
\mathcal{E}^{(g)}: \mathbb{R}^{C_g \times H \times W} \rightarrow \mathbb{R}^{D_g \times H' \times W'}
$$

### Decoder（逐 group，条件解码）


$$
\mathcal{D}^{(g)}: \mathbb{R}^{D_g \times H' \times W'}
\rightarrow \mathbb{R}^{C_g \times H \times W}
$$

### Latent Mixing + Predictor（全局）


$$
\mathcal{M}: \mathbb{R}^{D \times H' \times W'} \rightarrow \mathbb{R}^{D \times H' \times W'}
$$

其中：

$$
D = \sum_{g=1}^{G} D_g
$$

---

# 2. 确定性 Autoencoder（逐 group）

## 2.1 Encoder（确定性隐变量）


$$
h_t^{(g)} = \mathcal{E}^{(g)}(x_t^{(g)})
$$

$$
h_{t+1}^{(g)} = \mathcal{E}^{(g)}(x_{t+1}^{(g)})
$$

---

## 2.2 条件 Decoder 与变量类型算子

先定义 decoder 的原始输出：


$$
y^{(g)} = \mathcal{D}^{(g)}(h^{(g)})
$$

将输出拆分：

$$
y^{(g)} = \big(y^{(g,a)}, y^{(g,r)}\big)
$$

---

### 定义变量类型融合算子

定义融合算子：

$$
\mathcal{F}^{(g)}(y^{(g)}, x_t^{(g)}) \rightarrow \hat{x}_{t+1}^{(g)}
$$

其具体形式为：


$$
\hat{x}_{t+1}^{(g,a)} = y^{(g,a)}
$$


$$
\hat{x}_{t+1}^{(g,r)} = x_t^{(g,r)} + y^{(g,r)}
$$

最终：

$$
\hat{x}_{t+1}^{(g)} =
\text{concat}\big(\hat{x}_{t+1}^{(g,a)}, \hat{x}_{t+1}^{(g,r)}\big)
$$

---

# 3. Latent Mixing 与时序预测

## 3.1 全局隐变量拼接


$$
h_t = \text{concat}_g\big(h_t^{(g)}\big)
$$

---

## 3.2 时空演变预测器（Mixing Predictor）

显式定义：


$$
\hat{h}_{t+1} = \mathcal{M}(h_t)
$$

再拆回各 group：


$$
\hat{h}_{t+1}^{(g)} = \text{split}_g(\hat{h}_{t+1})
$$

---

# 4. 三条训练路径（确定性）

## 4.1 输入时刻重建（AE-input）


$$
\hat{x}_t^{(g)} =
\mathcal{F}^{(g)}\Big(
\mathcal{D}^{(g)}(h_t^{(g)}),x_{t-1}^{(g)}
\Big)
$$

损失：

$$
\mathcal{L}_{ae}^{input} =
\sum_g \left| \hat{x}_t^{(g)} - x_t^{(g)} \right|_1
$$

---

## 4.2 目标时刻重建（AE-target）


$$
\hat{x}_{t+1}^{(g)} =
\mathcal{F}^{(g)}\Big(
\mathcal{D}^{(g)}(h_{t+1}^{(g)}), x_{t}^{(g)}
\Big)
$$

损失：

$$
\mathcal{L}_{ae}^{target} =
\sum_g \left| \hat{x}_{t+1}^{(g)} - x_{t+1}^{(g)} \right|_1
$$

---

## 4.3 预测路径（核心）


$$
\hat{x}_{t+1}^{pred(g)} =
\mathcal{F}^{(g)}\Big(
\mathcal{D}^{(g)}(\hat{h}_{t+1}^{(g)}),
x_t^{(g)}
\Big)
$$

损失：

$$
\mathcal{L}_{pred} =
\sum_g \left| \hat{x}_{t+1}^{pred(g)} - x_{t+1}^{(g)} \right|_1
$$

---

# 5. 隐空间一致性约束


$$
\mathcal{L}_{latent} =
\sum_g \left| \hat{h}_{t+1}^{(g)} - h_{t+1}^{(g)} \right|_1
$$

说明：

* 等价于对 $\mathcal{M}$ 的监督学习约束
* 在确定性框架中替代 VAE 的 KL 项

---

# 6. 总体损失函数


$$
\mathcal{L} =
\lambda_1 \mathcal{L}_{pred} + \lambda_2 \mathcal{L}_{ae}^{input} +
\lambda_3 \mathcal{L}_{ae}^{target} + \lambda_4 \mathcal{L}_{latent}
$$

---

# 7. 前向计算流程（统一表达）

整体模型可写为组合算子：


$$
\hat{x}_{t+1} =
\mathcal{F} \circ \mathcal{D} \circ \mathcal{M} \circ \mathcal{E}(x_t)
$$

展开为：


$$
x_t
\xrightarrow{\mathcal{E}}
h_t
\xrightarrow{\mathcal{M}}
\hat{h}_{t+1}
\xrightarrow{\mathcal{D}}
y
\xrightarrow{\mathcal{F}}
\hat{x}_{t+1}
$$

---

# 8. 变量类型判定（统计准则）

定义判定指标：


$$
R = \frac{\mathrm{std}(x_{t+1} - x_t)}{\mathrm{std}(x_t)}, \quad
\rho = \mathrm{corr}(x_t, x_{t+1})
$$

判定规则：


$$
R < \tau_1 ;\land; \rho > \tau_2
\Rightarrow \text{residual variable}
$$

否则为 absolute variable。

典型取值：

$$
\tau_1 = 0.2, \quad \tau_2 = 0.98
$$
