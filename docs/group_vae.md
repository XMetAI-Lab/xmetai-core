# 1. 基本符号定义

### 输入与分组

- 全变量：$$ x_t \in \mathbb{R}^{C \times H \times W} $$
- 分组（当前设计为3组或4组）：$$ x_t = \{x_t^{(g)}\}_{g=1}^G $$
- 每个 group 内变量按属性分类：$$x_t^{(g)} = \big(x_t^{(g,a)}, x_t^{(g,r)}\big)$$

其中：
- $(a)$: absolute（绝对值主导，如气候态）
- $(r)$: residual（扰动主导，如异常值）

---

# 2. 确定性 Autoencoder（AE）（每个 group 独立）

## 2.1 Encoder（确定性隐变量提取）

由于是确定性模型，去除了 VAE 的先验/后验分布假设和采样过程，直接提取隐式特征：
$$ h_t^{(g)} = \text{Encoder}^{(g)}(x_t^{(g)}) $$
$$ h_{t+1}^{(g)} = \text{Encoder}^{(g)}(x_{t+1}^{(g)}) $$

---

## 2.2 Decoder（统一 decoder）

decoder 也是确定性的，并引入条件输入：
- latent feature ($h$)
- 当前状态 ($x_t$) 

### 🔴 关键：混合 residual / absolute 输出
定义 decoder 的原始输出为：
$$ \hat{y}^{(g)} = f^{(g)}(h^{(g)}, x_t^{(g)}) $$

分两类变量处理：
### residual变量（基于当前状态的残差预测）：
$$ \hat{x}_{t+1}^{(g,r)} = x_t^{(g,r)} + \hat{y}^{(g,r)} $$

### absolute变量（直接输出）：
$$ \hat{x}_{t+1}^{(g,a)} = \hat{y}^{(g,a)} $$

### 合并：
$$ \hat{x}_{t+1}^{(g)} = \text{merge}\big(\hat{x}_{t+1}^{(g,a)}, \hat{x}_{t+1}^{(g,r)}\big) $$

---

# 3. Latent Mixing + Prediction

在隐变量空间进行多组特征的混合与时序演变预测。

拼接所有 group 获得全局隐空间特征：
$$ h_t = \text{concat}_g(h_t^{(g)}) $$

## mixing predictor (时空演变模块)：
$$ \hat{h}_{t+1} = H(h_t) $$

拆回各 group 得到对应的预测隐特征：
$$ \hat{h}_{t+1}^{(g)} $$

---

# 4. 三条核心重构与预测路径

在确定性框架下，模型仍然保持三条核心的数据流，但不再做分布采样：

## 4.1 AE reconstruction（t时刻状态重建）
利用 $t$ 时刻的隐特征重建 $t$ 时刻自身：
$$ \hat{x}_t^{(g)} = f^{(g)}(h_t^{(g)}, x_t^{(g)}) $$
（注：这里如果是自编码，部分实现可能省去了 $x_t$ 作为残差项，直接重构）

损耗（Autoencoder Loss）：
$$ \mathcal{L}_{ae}^{input} = | \hat{x}_t - x_t | $$

---

## 4.2 AE reconstruction（t+1时刻目标重建）
$$ \hat{x}_{t+1}^{ae(g)} = f^{(g)}(h_{t+1}^{(g)}, x_{t+1}^{(g)}) $$

损耗：
$$ \mathcal{L}_{ae}^{target} = | \hat{x}_{t+1}^{ae} - x_{t+1} | $$

---

## 4.3 预测路径（预测未来的核心分支）
将模型预测的 $\hat{h}_{t+1}$ 送入解码器，结合当前的 $x_t$ 进行未来预测：
$$ \hat{x}_{t+1}^{pred(g)} = f^{(g)}(\hat{h}_{t+1}^{(g)}, x_t^{(g)}) $$

损耗：
$$ \mathcal{L}_{pred} = | \hat{x}_{t+1}^{pred} - x_{t+1} | $$

---

# 5. Latent prediction loss（隐空间对齐）

约束演变模块 $H()$ 预测出的隐变量，与真实 $t+1$ 时刻通过 Encoder 提取的隐变量保持一致：
$$ \mathcal{L}_{latent} = \sum_g | \hat{h}_{t+1}^{(g)} - h_{t+1}^{(g)} | $$

*(这一项取代了 VAE 中的 KL 散度约束)*

---

# 6. 总 Loss

确定性模型的损失舍弃了 KL 散度 $\mathcal{L}_{KL}$，主要由重建和预测项控制：
$$
\mathcal{L} = \lambda_1 \mathcal{L}_{pred}
+ \lambda_2 \mathcal{L}_{ae}^{input}
+ \lambda_3 \mathcal{L}_{ae}^{target}
+ \lambda_4 \mathcal{L}_{latent}
$$

---

# 7. 设计的本质（核心逻辑总结）

整体前向过程简化为：
$$
x_t \xrightarrow{\text{Encoder}} h_t \xrightarrow{H} \hat{h}_{t+1} \xrightarrow{\text{conditional decode}} \hat{x}_{t+1}
$$

并且输出结构保持：
$$ 
\hat{x}_{t+1} =  
\begin{cases}  
x_t + f(h, x_t), & \text{residual vars (扰动主导)} \\
f(h, x_t), & \text{absolute vars (绝对值主导)}  
\end{cases}  
$$

---

# 8. 变量类型判断标准

如何判定一个变量是 residual (学习演变趋势) 还是 absolute (学习全量状态)：

$$
\frac{\mathrm{std}(x_{t+1} - x_t)}{\mathrm{std}(x_t)} < 0.2  
\quad \text{且} \quad  
\text{corr}(x_t, x_{t+1}) > 0.98
$$

满足 $\rightarrow$ residual  
否则 $\rightarrow$ absolute
