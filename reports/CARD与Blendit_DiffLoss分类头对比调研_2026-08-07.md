# CARD 与 Blendit DiffLoss 分类头对比调研

日期：2026-08-07

## 1. 调研对象

- 论文：Xizewen Han, Huangjie Zheng, Mingyuan Zhou，*CARD: Classification and
  Regression Diffusion Models*，NeurIPS 2022。
- CARD 官方代码：<https://github.com/XzwHan/CARD>
- Blendit 当前仓库中的 DiffLoss 分类头、标签扩散公共实现及主实验配置。

## 2. 核心结论

Blendit DiffLoss 与 CARD 属于同一条技术路线：二者都把离散类别映射成连续的
类别原型，在标签空间加入高斯噪声，并以输入特征为条件恢复类别原型。

但 Blendit 不是 CARD 的直接复现。当前 Blendit 已经去掉 CARD 最有辨识度的
“预训练分类器条件均值”，并把主配置调整为 `x_start + 1-step DDIM + 单样本推理`。
因此，更准确的定位是：

> Blendit DiffLoss 是一个 CARD-inspired conditional label-diffusion head，
> 即通过扩散噪声训练的条件类别原型预测头。

CARD 的重点是学习完整的条件分布 $p(y\mid x)$ 和进行实例级不确定性评估；
Blendit 当前实现的重点则是利用标签噪声、时间条件和类别原型几何改进下游分类，
在实际推理形态上更接近判别式分类头。

## 3. CARD 分类方法

### 3.1 研究目标

CARD 不满足于输出单个类别，而是希望学习响应变量的完整条件分布：

$$
p(\mathbf y\mid\mathbf x).
$$

分类任务中的真实标签首先表示为普通 one-hot 向量：

$$
\mathbf y_0=\operatorname{onehot}(y)\in\{0,1\}^{C}.
$$

### 3.2 预训练分类器

CARD 先单独训练一个普通分类器 $f_\phi(\mathbf x)$，并取其 softmax 输出：

$$
\hat{\mathbf y}_0=f_\phi(\mathbf x).
$$

这个预训练分类器不是普通的 encoder 初始化，而是 CARD 概率模型的组成部分。
它同时用于：

1. 为前向扩散提供条件均值；
2. 为终点先验 $p(\mathbf y_T\mid\mathbf x)$ 提供均值；
3. 作为反向去噪网络的显式输入条件。

### 3.3 前向扩散

CARD 的前向扩散不是标准的零中心扩散，而是在真实 one-hot 与预训练分类器输出之间
进行插值：

$$
\mathbf y_t
=
\sqrt{\bar\alpha_t}\mathbf y_0
+
(1-\sqrt{\bar\alpha_t})\hat{\mathbf y}_0
+
\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,
\qquad
\boldsymbol\epsilon\sim\mathcal N(0,I).
$$

因此，扩散终点的先验近似为：

$$
p(\mathbf y_T\mid\mathbf x)
=
\mathcal N(f_\phi(\mathbf x),I),
$$

而不是标准 DDPM 中的 $\mathcal N(0,I)$。

官方实现位于：

- `classification/diffusion_utils.py::q_sample`
- <https://github.com/XzwHan/CARD/blob/main/classification/diffusion_utils.py#L39-L50>

### 3.4 条件去噪网络与损失

CARD 的噪声预测器可以概括为：

$$
\epsilon_\theta(
\mathbf x,
\mathbf y_t,
t,
f_\phi(\mathbf x)
).
$$

官方 `ConditionalModel` 会单独编码输入 $\mathbf x$，并将当前噪声标签
$\mathbf y_t$ 与基础分类器输出拼接后预测噪声：

- <https://github.com/XzwHan/CARD/blob/main/classification/model.py#L22-L94>

基础训练目标是噪声 MSE：

$$
\mathcal L_{\mathrm{CARD}}
=
\mathbb E\left[
\left\|
\boldsymbol\epsilon-
\epsilon_\theta(\mathbf x,\mathbf y_t,t,f_\phi(\mathbf x))
\right\|_2^2
\right].
$$

官方代码默认使用 `(e - output).square().mean()`。代码支持额外加入由重构
$\mathbf y_0$ 计算的交叉熵，但它不是基础 CARD 损失：

- <https://github.com/XzwHan/CARD/blob/main/classification/card_classification.py#L298-L326>

### 3.5 推理与类别决策

CARD 从条件先验中随机采样：

$$
\mathbf y_T
=
f_\phi(\mathbf x)+\mathbf z,
\qquad
\mathbf z\sim\mathcal N(0,I),
$$

然后经过随机 DDPM 反向链得到重构的 $\tilde{\mathbf y}_0$。

CARD 不直接把 $\tilde{\mathbf y}_0$ 当作 logits，而是计算每一维到 one-hot
正值 1 的距离：

$$
s_c=-(\tilde y_{0,c}-1)^2,
\qquad
p_c=\operatorname{softmax}(s_c/T).
$$

官方代码对应：

- <https://github.com/XzwHan/CARD/blob/main/classification/card_classification.py#L529-L554>

标准分类实验通常为每个输入生成 100 个重构结果。每个结果先产生一个类别，最后使用
多数投票作为预测。不同生成样本之间的变化还用于计算 prediction interval、配对
$t$-test、PAvPU 等实例级不确定性指标。

因此，CARD 的主要价值包括：

- 改进预训练基础分类器的点预测；
- 学习条件标签分布，而不仅是一个确定性 logits 向量；
- 通过重复生成度量 aleatoric uncertainty；
- 对多模态条件分布进行建模。

## 4. Blendit 当前 DiffLoss 分类头

### 4.1 B-Rep 图条件

Blendit 先使用 B-Rep 图编码器产生各个面的 embedding，再对一个 CAD 模型中的所有面
进行均值池化：

$$
\mathbf h_G
=
\frac{1}{|V_G|}
\sum_{v\in V_G}\mathbf h_v.
$$

分类模型的相关实现位于：

- `src/blendit/models/classification.py::mean_graph_pool`
- `src/blendit/models/classification.py::DiffusionClassificationModel`

每个 CAD 模型只产生一个图级标签 token；DiffLoss 去噪器的条件是图 embedding
$\mathbf h_G$。

### 4.2 双极 one-hot 标签

Blendit 使用双极 one-hot，而不是 CARD 的普通 one-hot：

$$
\mathbf z_0
=
2\operatorname{onehot}(y)-1
\in\{-1,+1\}^{C}.
$$

正确类别对应 $+1$，其他类别对应 $-1$。实现位于：

- `src/blendit/models/diffusion.py::bipolar_one_hot`

### 4.3 标准零中心前向扩散

Blendit 的标签前向过程为：

$$
\mathbf z_t
=
\sqrt{\bar\alpha_t}\mathbf z_0
+
\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon.
$$

对应实现：

- `src/blendit/models/diffusion.py::LabelDiffusionSchedule.q_sample`

这里没有 CARD 的条件均值项：

$$
(1-\sqrt{\bar\alpha_t})f_\phi(\mathbf x).
$$

所以 Blendit 扩散先验仍然以零为中心，图输入仅通过去噪器的 condition 分支进入模型。

### 4.4 没有 CARD 式基础分类器

Blendit 的去噪器可以表示为：

$$
g_\theta(\mathbf z_t,t,\mathbf h_G).
$$

它没有独立的 $f_\phi(\mathbf x)$ 分类器，也没有一个先由 CE 训练、再作为扩散均值锚点
的类别概率向量。

Blendit 的预训练 checkpoint 对分类任务只加载 `encoder.*`，不会把 coarse label head
映射到图级分类头。因此：

- CARD 预训练分类器提供类别概率先验；
- Blendit 预训练模型只提供几何表征初始化；
- Blendit 的 encoder 和 DiffLoss head 在下游训练中端到端微调。

相关实现：

- `src/blendit/training/common.py::load_pretrain_checkpoint_for_finetune`
- `src/blendit/models/downstream.py::LabelDiffusionModel`

### 4.5 当前主配置：纯 `x_start` 预测

当前联合预训练后的 TMCAD、FabWave 等主配置采用：

```yaml
target_encoding: bipolar_one_hot
train_timesteps: 1000
noise_schedule: cosine
prediction_type: x_start
x_start_loss_weight: 1.0
epsilon_loss_weight: 0.0
noise_samples_per_token: 4
sampling_method: ddim
sampling_steps: 1
ddim_eta: 0.0
sampling_temperature: 0.75
inference_samples: 1
```

因此当前训练目标是：

$$
\mathcal L_{\mathrm{Blendit}}
=
\mathbb E\left[
\|\hat{\mathbf z}_0-\mathbf z_0\|_2^2
\right].
$$

每个标签重复四次，并为每次重复独立采样 timestep 和高斯噪声。相关实现与配置：

- `src/blendit/models/downstream.py::prepare_label_diffusion_training_batch`
- `src/blendit/models/downstream.py::label_diffusion_objective`
- `configs/finetune_joint_tmcad_diffloss_200.yaml`
- `configs/finetune_joint_fabwave_min10_diffloss_acc_200.yaml`

### 4.6 早期配置与当前配置的区别

早期 DiffLoss 配置采用 `x_start_epsilon` 双输出：

$$
\mathcal L
=
\frac{
1.0\|\hat{\mathbf z}_0-\mathbf z_0\|_2^2
+
0.5\|\hat{\boldsymbol\epsilon}-\boldsymbol\epsilon\|_2^2
}{1.5},
$$

并使用 25-step DDIM。相关配置包括：

- `configs/finetune_diffloss.yaml`
- `configs/finetune_tmcad_diffloss.yaml`

当前主配置已经改成纯 `x_start`、1-step DDIM。因此分析“现在的 Blendit DiffLoss”时，
应以新的单步配置为准，而不应继续把它描述为 25 步、同时预测噪声和干净标签的模型。

### 4.7 当前推理形态

当 `sampling_steps: 1` 时，采样 timestep 只有最大噪声时刻。模型执行一次网络计算，
直接从高噪声标签状态预测 $\hat{\mathbf z}_0$：

$$
\mathbf z_{T-1}
\xrightarrow[\text{一次网络调用}]{g_\theta}
\hat{\mathbf z}_0.
$$

因此推理时不会真正走一条多步反向扩散轨迹。

初始噪声由 `seed`、`sample_index` 和 `sample_id` 的哈希稳定生成；同一个 CAD 模型在
相同配置下会得到可复现的初始噪声和预测。最后使用：

$$
p_c
=
\operatorname{softmax}
\left(
\hat z_{0,c}/T+b_c
\right).
$$

若 `inference_samples` 大于 1，Blendit 会平均不同样本的 softmax 概率；当前主配置为 1，
所以没有进行 Monte Carlo 分布估计。

相关实现：

- `src/blendit/models/downstream.py::stable_initial_noise`
- `src/blendit/models/downstream.py::predict_label_diffusion_probabilities`
- `src/blendit/models/diffusion.py::LabelDiffusionSchedule.sample`

## 5. CARD 与 Blendit 逐项对比

| 项目 | CARD | Blendit 当前 DiffLoss |
|---|---|---|
| 输入 | 图像或一般协变量 $x$ | B-Rep 图及其 pooled graph embedding |
| 标签原型 | 普通 one-hot，$\{0,1\}^C$ | 双极 one-hot，$\{-1,+1\}^C$ |
| 前向扩散中心 | $f_\phi(x)$ 条件均值 | 零中心 |
| 独立基础分类器 | 有，先用 CE 预训练 | 没有 |
| 预训练作用 | 提供分类概率及终点先验均值 | 初始化几何 encoder |
| 去噪条件 | $x,y_t,t,f_\phi(x)$ | $h_G,z_t,t$ |
| 默认预测目标 | 噪声 $\epsilon$ | 当前为干净原型 $z_0$ |
| Noise schedule | 论文主设置为 linear | cosine |
| 训练 timestep | 1000 | 1000 |
| 推理方法 | 随机 DDPM 反向链 | 当前为 1-step、`eta=0` DDIM |
| 每输入采样数 | 分类实验通常为 100 | 当前为 1 |
| 聚合规则 | 每个样本分类后多数投票 | softmax 概率；多样本时均值 |
| 类别分数 | $-(\tilde y_c-1)^2/T$ | $\hat z_c/T+b_c$ |
| 不确定性 | PIW、$t$-test、PAvPU 等 | 当前未实现 CARD 式评估 |
| 主要目标 | 条件分布和预测不确定性 | 下游准确率和标签噪声正则化 |

## 6. 两者的联系

二者共享的算法骨架是：

$$
\text{离散类别}
\rightarrow
\text{连续类别原型}
\rightarrow
\text{高斯加噪}
\rightarrow
\text{条件恢复原型}
\rightarrow
\text{类别决策}.
$$

Blendit 继承了 CARD 或同类 label-space diffusion 方法的三项基本思想：

1. 将分类视为连续标签空间中的条件生成或恢复问题；
2. 对类别原型进行 Gaussian diffusion/noising；
3. 使用 MSE 去噪目标代替必须直接优化交叉熵。

但是，Blendit 没有继承 CARD 最关键的完整机制：

1. 没有预训练分类器 $f_\phi(x)$；
2. 没有将 $f_\phi(x)$ 注入前向过程和终点先验；
3. 当前没有随机多步反向生成；
4. 当前没有通过大量重复采样估计条件标签分布；
5. 没有采用多数投票及 CARD 的 PIW、$t$-test、PAvPU 分析。

## 7. 标签编码和最终类别的关系

CARD 使用 $\{0,1\}^C$ 原型，并根据到正值 1 的平方距离计算类别分数。Blendit 使用
$\{-1,+1\}^C$ 原型，直接把重构值作为 softmax score。

Blendit 推理会把 $\hat z_0$ 裁剪到 $[-1,1]$。在这个区间内：

$$
-(z_c-1)^2
$$

是关于 $z_c$ 的单调递增函数。因此，对同一个裁剪后的向量，CARD 式距离分数与
Blendit 直接使用 $z_c$ 通常会产生相同的 top-1 类别。

不过，两者的 score 非线性不同，所以 softmax 概率、置信度和校准结果并不相同。
这意味着标签编码差异对 argmax 的影响较小，但对概率解释有明显影响。

## 8. 如何理解当前 Blendit DiffLoss 的收益

当前 Blendit DiffLoss 的主要效果更可能来自以下组合，而不是 CARD 所强调的完整
条件分布建模：

- 双极类别原型带来的类别间几何间隔；
- 在不同噪声强度下恢复标签形成的正则化；
- 每个标签四组噪声带来的训练增强；
- timestep-conditioned nonlinear head；
- AdaLN 条件调制；
- B-Rep encoder 与 DiffLoss head 的端到端联合微调。

特别是在 `x_start + 1-step + inference_samples=1` 设置下，实际预测近似为：

$$
\hat y
=
\arg\max_c
g_\theta(
\text{固定噪声},
t_{\max},
\mathbf h_G
)_c.
$$

所以从部署角度看，它是“经过扩散式标签噪声训练的类别原型回归器”。它仍然可能比
普通 MLP 有更好的泛化效果，但当前配置不能直接提供 CARD 所声称的条件分布和
实例级不确定性能力。

## 9. Blendit 当前实验结果的含义

在统一采用 validation accuracy 选择 checkpoint 的本地受控实验中：

| 分类任务 | MLP Test Acc | DiffLoss Test Acc | DiffLoss 相对变化 |
|---|---:|---:|---:|
| TMCAD | 81.8767% | 84.2686% | +2.3919 pp |
| FabWave min10 | 97.9540% | 97.9540% | +0.0000 pp |

结果来源：`reports/mlp_vs_diffloss_max_acc_results_2026-08-07.md`。

这些结果表明：

- 标签空间噪声训练在 TMCAD 上明显优于当前 MLP 基线；
- 在 FabWave min10 上，DiffLoss 与 MLP 得到完全相同的测试预测；
- DiffLoss 并非对所有数据集都会自动优于 MLP；
- 这些准确率结果只能证明当前 DiffLoss 分类头的任务效果，不能证明它已经获得 CARD
  的不确定性建模能力。

## 10. 命名与论文表述建议

在项目文档或论文中，不建议直接写“Blendit 使用 CARD 分类器”，因为两者在条件先验、
训练目标和采样协议上都有实质差异。

建议使用以下表述：

> We employ a CARD-inspired label-space diffusion head. Class labels are represented as
> bipolar one-hot prototypes and corrupted with Gaussian noise. A graph-conditioned
> denoising MLP is trained to reconstruct the clean class prototype from noisy label
> tokens at randomly sampled diffusion timesteps.

若要准确描述当前推理协议，可以继续说明：

> Unlike the original CARD classifier, our current implementation does not use a
> separately pre-trained classifier as the terminal prior mean. It performs deterministic
> one-step DDIM inference from a sample-specific seeded noise vector.

中文可表述为：

> Blendit 使用受 CARD 启发的标签空间扩散分类头。类别被编码为双极 one-hot 原型，
> 训练时在随机扩散时刻加入高斯噪声，并由 B-Rep 图表示条件化的去噪 MLP 恢复干净
> 类别原型。与原始 CARD 不同，当前实现不使用独立预训练分类器作为扩散终点均值，
> 并采用确定性的单步 DDIM 推理。

## 11. 若要进一步接近 CARD

如果后续目标是复现 CARD 的概率建模能力，而不仅是保持当前分类精度，可以考虑以下
独立消融：

1. 增加一个 CE 预训练的图级基础分类器 $f_\phi(G)$；
2. 将其 softmax 输出加入前向扩散均值；
3. 从 $\mathcal N(f_\phi(G),I)$ 而不是 $\mathcal N(0,I)$ 开始反向采样；
4. 将 $f_\phi(G)$ 作为去噪器的额外显式条件；
5. 恢复 epsilon-prediction，并与当前 `x_start` 目标做受控比较；
6. 使用多步随机采样以及至少几十个 inference samples；
7. 分别比较多数投票、mean probability 和当前单样本预测；
8. 增加 NLL、ECE、Brier score、prediction interval 和 selective accuracy 等指标。

这些改动应作为“CARD-faithful”分支单独评估，不宜直接覆盖当前效果较好的单步 DiffLoss
配置，因为完整 CARD 推理会显著增加推理成本，而且未必提高纯 top-1 accuracy。

## 12. 最终总结

CARD 是由预训练分类器引导的条件标签分布生成模型：预训练分类概率既是前向扩散的
均值锚点，也是终点先验和反向网络条件；推理通过大量随机生成、概率变换和多数投票
产生类别，并利用样本分布评估不确定性。

Blendit 当前 DiffLoss 是以 B-Rep graph embedding 为条件的标签原型恢复模型：它使用
双极 one-hot、零中心高斯扩散、纯 `x_start` MSE，以及单样本单步 DDIM 推理。两者具有
相同的标签空间扩散思想，但在条件先验、损失、采样方式和实际目标上已经明显分化。

一句话概括：

> CARD 学习“给定输入可能产生怎样的标签分布”；Blendit 当前 DiffLoss 学习“给定
> CAD 图和一个带噪标签 token，如何稳定恢复正确的类别原型”。

## 13. 参考资料

### 论文与官方代码

1. Han, X., Zheng, H., Zhou, M. *CARD: Classification and Regression Diffusion Models*.
   NeurIPS 2022：<https://arxiv.org/abs/2206.07275>
2. NeurIPS Proceedings：
   <https://papers.nips.cc/paper_files/paper/2022/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html>
3. CARD 官方仓库：<https://github.com/XzwHan/CARD>
4. CARD 前向与反向扩散实现：
   <https://github.com/XzwHan/CARD/blob/main/classification/diffusion_utils.py>
5. CARD 分类训练与不确定性评估：
   <https://github.com/XzwHan/CARD/blob/main/classification/card_classification.py>
6. CARD 条件网络：
   <https://github.com/XzwHan/CARD/blob/main/classification/model.py>

### Blendit 本地实现

1. `src/blendit/models/classification.py`
2. `src/blendit/models/downstream.py`
3. `src/blendit/models/diffusion.py`
4. `src/blendit/training/common.py`
5. `configs/finetune_joint_tmcad_diffloss_200.yaml`
6. `configs/finetune_joint_fabwave_min10_diffloss_acc_200.yaml`
7. `configs/finetune_tmcad_diffloss.yaml`
8. `reports/mlp_vs_diffloss_max_acc_results_2026-08-07.md`
