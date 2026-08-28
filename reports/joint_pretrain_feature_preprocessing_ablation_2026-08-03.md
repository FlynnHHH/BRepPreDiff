# 联合预训练连续特征预处理消融报告

实验日期：2026-08-03  
报告生成日期：2026-08-04

## 结论摘要

三组联合数据集预训练均在 4 张 TITAN RTX 上完成 150 epochs，未发现 Traceback、CUDA OOM、NaN 或 Inf。

1. **完全不标准化不适合作为默认方案。**其末 10 epochs 的平均 total loss 为 `16.69300 ± 0.92020`，主要由未缩放面特征造成的 face reconstruction loss（`30.78580 ± 1.84054`）主导；最终 face noise loss 也比逐图标准化高 15.8%。
2. **按特征类型处理并使用训练集全局统计量，在本次预训练目标上表现最好。**末 10 epochs 平均 total loss 为 `1.06356 ± 0.00687`，比逐图标准化的 `1.33130 ± 0.00206` 低 20.1%；最终 face/edge noise loss 分别低 2.1% 和 43.5%。
3. **类型化全局标准化具有明显性能成本。**150 epochs 用时约 2 小时 53 分，比逐图标准化约慢 76%，当前实现中的逐样本通道处理需要进一步向量化或缓存。
4. **暂不能仅凭预训练 loss 宣布下游模型优胜。**三种预处理改变了重建目标的数值空间，重建项及 total loss 并非完全同尺度；最终选择必须使用相同下游微调协议比较 Accuracy、Macro-F1 和 mIoU。

综合建议：将 `typewise_global` 作为下一轮下游验证的首选候选，将 `per_graph` 保留为速度更快的强基线，不再投入完全不标准化方案。

## 实验设计

三组实验除连续特征预处理方式外，其余条件保持一致：

| 项目 | 设置 |
|---|---|
| 数据配置 | `data/pretrain_joint_all_splits.yaml` |
| 模型 | 4 层、hidden dim 128 的扩散预训练模型 |
| 训练轮数 | 150 epochs |
| 随机种子 | 42 |
| 优化器 | AdamW，学习率 `1e-3`，weight decay `1e-4` |
| 批大小 | 每个 DDP rank 512，4 ranks |
| 硬件 | 4 × TITAN RTX（24 GB） |
| 连续特征维度 | 面 611，边 3 |
| checkpoint 周期 | 每 5 epochs |
| 验证集 | 未配置；本报告只分析训练指标 |

联合配置的有效自监督训练集合共有：

| 统计项 | 数量 |
|---|---:|
| 图 | 278,518 |
| 面 | 17,180,451 |
| 有向边 | 86,749,482 |

该联合配置属于 transductive 自监督协议：BRepPreDiff 使用 train；TMCAD、Fusion360Seg 和 MFCAD++ 的 train/val/test 几何均被映射到预训练 `train`，但不加载标签。因此本结果不能直接当作严格 inductive 数据隔离协议下的结论。

## 三种预处理方法

### 1. `per_graph`

对每个 CAD 图分别计算连续面特征和连续边特征各通道的均值与总体标准差，并执行 z-score：

\[
x' = \frac{x - \mu_{graph}}{\max(\sigma_{graph}, 10^{-6})}
\]

这是实验前的默认实现。

### 2. `none`

缓存中的连续特征不再做额外标准化，直接输入模型。提取阶段已有的 `log1p(area)`、`log1p(length)`、单位法向量、`angle / π` 等工程变换仍然保留。

### 3. `typewise_global`

全局统计量只从上述有效自监督训练集合拟合，运行时按特征物理类型处理：

- 面积、质心、曲率和 UV 网格采样点坐标等非方向连续通道：使用训练集全局 mean/std 做 z-score。
- 中心法向量和 UV 网格法向量：按 xyz 三元组做 L2 单位化，不做逐分量 z-score。
- 边长度：对 `log1p(length)` 使用训练集全局 mean/std。
- 边夹角：保留 `angle / π` 并限制在 `[0, 1]`。
- 法向量点积：限制在 `[-1, 1]`。

统计文件为 `runs/feature_stats/joint_all_splits_train.json`。

## 结果

### 最终 epoch

| 方法 | Total | Face noise | Edge noise | Face recon | Edge recon | Surface | Edge type | Relation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `per_graph` | 1.33092 | 0.85522 | 0.09942 | 0.38063 | 0.37164 | **0.00016** | 0.00010 | 0.00004 |
| `none` | 14.99196 | 0.99008 | 0.12179 | 27.38044 | 0.29790 | 0.08159 | 0.00018 | 0.00014 |
| `typewise_global` | **1.05835** | **0.83721** | **0.05617** | **0.13202** | **0.19760** | 0.00023 | **0.00007** | **0.00003** |

加粗表示该列数值最低。Face/edge reconstruction 以及 total loss 会受到输入表示尺度影响，跨预处理方案比较时只能作为辅助证据；noise 和离散属性预测项更适合直接横向参考。

### 末 10 epochs 稳定性

| 方法 | Total | Face noise | Edge noise | Face recon | Edge recon |
|---|---:|---:|---:|---:|---:|
| `per_graph` | 1.33130 ± 0.00206 | 0.85542 ± 0.00015 | 0.10108 ± 0.00118 | 0.37859 ± 0.00163 | 0.37074 ± 0.00183 |
| `none` | 16.69300 ± 0.92020 | 0.99022 ± 0.00029 | 0.12198 ± 0.00196 | 30.78580 ± 1.84054 | 0.29637 ± 0.00294 |
| `typewise_global` | **1.06356 ± 0.00687** | **0.83843 ± 0.00177** | **0.05808 ± 0.00184** | **0.13299 ± 0.00155** | **0.20002 ± 0.00328** |

`per_graph` 的 total loss 方差最低，表现出最平稳的末期训练；`typewise_global` 的平均损失更低，但离散 surface loss 在 epoch 142 出现一次短暂尖峰，使末 10 epochs 的波动略大。训练最终恢复正常并在 epoch 150 达到全程最低 total loss `1.05835`。

### 收敛与耗时

| 方法 | Epoch 1 total | Epoch 150 total | 全程最低 total | 150 epochs 用时 | 相对 `per_graph` |
|---|---:|---:|---:|---:|---:|
| `per_graph` | 2.15359 | 1.33092 | 1.32965（epoch 142） | 1:38:10 | 基准 |
| `none` | 36.27103 | 14.99196 | 14.99196（epoch 150） | 1:28:58 | 快 9.4% |
| `typewise_global` | 1.76561 | 1.05835 | 1.05835（epoch 150） | 2:53:10 | 慢 76.4% |

`none` 最快是因为没有运行时预处理，但节省的时间不足以抵消其明显恶化的训练目标。`typewise_global` 当前对每个样本逐通道处理 101 组面法向量，是其数据加载开销上升的主要可疑来源；需要通过 profiler 验证后再优化。

## 结果解读

### 为什么完全不标准化表现差

未标准化时，不同物理量的尺度差异直接进入同一个投影层。坐标、曲率和其他大尺度通道主导了面重建目标，使 face reconstruction loss 在末 10 epochs 仍达到 `30.78580 ± 1.84054`。同时 face noise loss 接近 1，说明噪声预测学习也明显弱于两个标准化方案。

### 为什么类型化处理可能优于逐图标准化

逐图 z-score 会对所有连续通道采用相同规则，包括已经具有方向或固定范围语义的法向量、角度和点积；它还会消除每个零件的绝对统计差异。类型化方案保留了这些物理约束，同时对无界通道应用训练集全局尺度，因此在 face noise、edge noise 和边重建目标上都取得更低结果。

### 当前证据不能回答的问题

- 是否提高 BRepPreDiff、Fusion360Seg、MFCAD++ 面分割的 Accuracy、Macro-F1 和 mIoU。
- 是否提高 TMCAD 图分类性能。
- 是否在严格排除下游 val/test 几何的 inductive 预训练协议下仍然成立。
- 类型化全局统计对不同 CAD 单位制、异常曲率及新数据域的鲁棒性。

## 建议的下一步

1. 使用 `per_graph`、`none` 和 `typewise_global` 三个 `last.pt`，在四个下游任务上采用完全相同的微调配置与随机种子；优先比较 Macro-F1/mIoU，而不是预训练 total loss。
2. 若资源有限，可先只比较 `per_graph` 与 `typewise_global`；当前证据已经足以淘汰 `none`。
3. 对 `typewise_global` 的法向量处理做批量向量化，减少约 76% 的额外训练时间。
4. 增加严格 inductive 联合配置，只使用各数据集 train 几何重新拟合统计量并预训练，用于确认 transductive 几何暴露是否影响结论。

## 实验产物

| 方法 | Checkpoint | SHA-256 |
|---|---|---|
| `per_graph` | `runs/pretrain/20260803-155054_joint_ablation_per_graph/checkpoints/last.pt` | `8078225a990af72129aeb8b14543322b112ea76b3c5910331f770036aa875899` |
| `none` | `runs/pretrain/20260803-173114_joint_ablation_none/checkpoints/last.pt` | `4d73e54417f58906208a295b3d4ca3ed11274fbd5d148c312122f8146379ca77` |
| `typewise_global` | `runs/pretrain/20260803-190220_joint_ablation_typewise_global/checkpoints/last.pt` | `7c2969b7e05306767ac7ca2ca427a1b463c72dbc1153cd3491d1d982286d7dc2` |

每个 checkpoint 约 14 MiB。完整原始日志位于：

- `runs/launch_logs/20260803-154958_joint_pretrain_per_graph.log`
- `runs/launch_logs/20260803-154958_joint_pretrain_none.log`
- `runs/launch_logs/20260803-154958_joint_pretrain_typewise_global.log`
