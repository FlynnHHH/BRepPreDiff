# MFCAD++ / TMCAD 连续特征预处理消融报告

实验日期：2026-08-04

## 结论

六组补充实验已全部完成，训练日志中未发现 Traceback、CUDA OOM、NaN/Inf 或其他 ERROR。

- **MFCAD++：完全不标准化（`none`）最优**，test Accuracy `99.17%`、Macro-F1 `98.76%`、mIoU `97.58%`。相对当前逐图标准化提高 `0.13`、`0.28`、`0.53` 个百分点。
- **TMCAD：逐图标准化（`per_graph`）最优**，test Accuracy `82.24%`、Macro-F1 `81.58%`、Macro-IoU `69.64%`。`typewise_global` 接近但仍低 `0.65` pp Macro-F1；`none` 低 `3.04` pp。
- 结合此前 Fusion360Seg 结果，**不存在跨数据集统一最优的预处理方式**：Fusion360Seg 与 MFCAD++ 偏好 `none`，TMCAD 偏好 `per_graph`；当前 `typewise_global` 在三个数据集上均未取得总体第一。

因此，现有证据支持按下游数据集选择：MFCAD++/Fusion360Seg 使用 `none`，TMCAD 保留 `per_graph`。但这些结果均为 seed 42 单次实验，尚不能替代多随机种子统计结论。

## 实验设计

三组模型使用同一种预处理完成联合数据集预训练，并在下游继续使用匹配的预处理：

| 方法 | 处理方式 | 联合预训练 checkpoint |
|---|---|---|
| `per_graph` | 每个图内逐通道 z-score | `runs/pretrain/20260803-155054_joint_ablation_per_graph/checkpoints/last.pt` |
| `none` | 不做额外标准化 | `runs/pretrain/20260803-173114_joint_ablation_none/checkpoints/last.pt` |
| `typewise_global` | 非方向连续通道用联合训练集统计量 z-score；法向量 L2 归一化；角度/点积限幅 | `runs/pretrain/20260803-190220_joint_ablation_typewise_global/checkpoints/last.pt` |

`typewise_global` 使用 `runs/feature_stats/joint_all_splits_train.json`，统计范围是联合预训练有效训练集合：278,518 个图、17,180,451 个面和 86,749,482 条有向边。

共同微调协议：

| 项目 | 设置 |
|---|---|
| 随机种子 | 42 |
| Epochs | 100 |
| Batch size | 256（test 为 512） |
| 下游头 | MLP |
| Encoder | 全量解冻 |
| 优化器 | AdamW，学习率 `3e-4`，weight decay `1e-4` |
| 模型选择 | 每个 epoch 验证，以 Val Macro-F1 最高的 `best.pt` 测试 |
| MFCAD++ | 25 类面分割；8,949 个 test 图、268,982 个 test 面 |
| TMCAD | 10 类零件分类；1,087 个 test 样本 |
| 硬件 | 4 张 TITAN RTX，按队列并行执行六组训练/测试 |

## MFCAD++ 结果

| 排名 | 方法 | Best epoch | Val Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | mIoU | Weighted-F1 | Weighted-IoU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `none` | 98 | **98.79%** | **99.17%** | **98.83%** | **98.70%** | **98.76%** | **97.58%** | **99.17%** | **98.37%** |
| 2 | `per_graph` | 86 | 98.65% | 99.04% | 98.64% | 98.33% | 98.49% | 97.05% | 99.04% | 98.12% |
| 3 | `typewise_global` | 96 | 98.32% | 98.86% | 98.56% | 98.15% | 98.35% | 96.82% | 98.85% | 97.77% |

相对 `per_graph`：

| 方法 | Accuracy | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|
| `none` | **+0.13 pp** | **+0.28 pp** | **+0.53 pp** | **+0.13 pp** |
| `typewise_global` | -0.18 pp | -0.13 pp | -0.23 pp | -0.19 pp |

`none` 在 25 个类别中的 18 个取得最高 F1，优势不是只来自 Stock 多数类。相对 `per_graph`，提升最大的类别是 Rectangular through slot（`+1.30` pp）、Rectangular blind slot（`+1.28` pp）和 Triangular through slot（`+0.94` pp）；仅 Stock 和 Circular blind step 各下降约 `0.07` pp。

`typewise_global` 在 5 类上第一，但 Rectangular through slot（`-2.85` pp）和 Rectangular through step（`-2.01` pp）的退化拉低了总体表现。

## TMCAD 结果

| 排名 | 方法 | Best epoch | Val Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Macro-IoU | Weighted-F1 | Weighted-IoU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `per_graph` | 56 | **84.37%** | **82.24%** | **81.86%** | **81.95%** | **81.58%** | **69.64%** | **82.01%** | **70.31%** |
| 2 | `typewise_global` | 97 | 81.23% | 81.42% | 81.05% | 81.04% | 80.93% | 68.73% | 81.42% | 69.48% |
| 3 | `none` | 96 | 81.91% | 79.30% | 78.49% | 78.89% | 78.54% | 65.59% | 79.02% | 66.30% |

相对 `per_graph`：

| 方法 | Accuracy | Macro-F1 | Macro-IoU | Weighted-F1 |
|---|---:|---:|---:|---:|
| `typewise_global` | -0.83 pp | -0.65 pp | -0.92 pp | -0.60 pp |
| `none` | -2.94 pp | -3.04 pp | -4.06 pp | -2.99 pp |

逐类上，`per_graph` 与 `typewise_global` 各有 4 类 F1 第一，`none` 有 2 类第一。`typewise_global` 对 pulley 提升 `3.20` pp，但对 shaft 和 screw 分别下降 `4.77`、`4.37` pp。`none` 的主要损失来自 coupling（`-8.03` pp）、bearing（`-7.85` pp）、gear（`-6.80` pp）和 screw（`-5.56` pp）。

值得注意的是，TMCAD 的 `none` 在验证集 Macro-F1（`81.91%`）略高于 `typewise_global`（`81.23%`），测试集排序却反转。这反映了 1,087 个 test 样本及单 seed 条件下的方差，不能把两者 `2.39` pp 的 test 差距直接解释为稳定的总体差异。

## 三个下游数据集的联合判断

| 数据集 | 任务 | `per_graph` Macro-F1 | `none` Macro-F1 | `typewise_global` Macro-F1 | 最优 |
|---|---|---:|---:|---:|---|
| Fusion360Seg | 8 类面分割 | 84.54% | **86.34%** | 72.98% | `none` |
| MFCAD++ | 25 类面分割 | 98.49% | **98.76%** | 98.35% | `none` |
| TMCAD | 10 类零件分类 | **81.58%** | 78.54% | 80.93% | `per_graph` |

结果验证了连续特征尺度处理会实质影响下游表现，但影响与数据域和任务有关：

1. MFCAD++ 和 Fusion360Seg 的分割任务可能从绝对尺寸、曲率或位置分布中获益，逐图 z-score 会移除其中一部分信号；`none` 在这两个数据集上更好。
2. TMCAD 是跨零件分类，逐图标准化可能降低零件坐标系、单位或整体尺度差异造成的域内噪声，因此泛化更好。
3. 当前联合全局统计量并未形成稳健优势。它在 Fusion360Seg 上严重退化，在 MFCAD++ 略低于其余两组，在 TMCAD 接近最优。可能原因是多数据源 CAD 单位与分布不一致，以及不同特征组经过不同变换后改变了输入投影层的相对尺度。

以上是由实验相关性支持的解释，不是已隔离的因果证明。

## 建议与限制

1. 当前下游默认选择：Fusion360Seg/MFCAD++ 使用 `none`，TMCAD 使用 `per_graph`。
2. 用 seeds 43、44 复现实验，并报告三随机种子的均值与标准差；MFCAD++ 的 `none` 优势仅 `0.28` pp Macro-F1，尤其需要确认稳定性。
3. 暂不将当前 `typewise_global` 设为通用默认值。若继续验证，应分别使用各下游 train split 拟合统计量，并拆分消融坐标、尺度、曲率、法向量和角度通道。
4. 联合预训练采用 transductive 几何协议：下游 val/test 几何参与无标签预训练；对外报告时必须明确这一点。
5. 三组预训练 loss 的尺度不可直接横向比较，因为预处理同时改变了重建目标的数值尺度；下游指标才是本次模型选择依据。

## 产物与可复现性

| 数据集 | 方法 | Run directory | Best checkpoint SHA-256 |
|---|---|---|---|
| MFCAD++ | `per_graph` | `runs/finetune/20260804-133605_mfcadpp_ablation_per_graph` | `21b3f92c19a087eee7a7feedb6566beccdac82e57aab356388eb3f947f6048e5` |
| MFCAD++ | `none` | `runs/finetune/20260804-133605_mfcadpp_ablation_none` | `7d510b96402b89d3f3b7088cde2e686206d6f1d615be21f57855043a37e03d73` |
| MFCAD++ | `typewise_global` | `runs/finetune/20260804-133605_mfcadpp_ablation_typewise_global` | `2dcf23b49fabf4512fc4b635adbbeaac50cfa83eed725cf70f21fb323d39c2c3` |
| TMCAD | `per_graph` | `runs/finetune/20260804-133605_tmcad_ablation_per_graph` | `6ab03b07e5b128aa13158ac10b294cb507d71d156ac5c80957fe6636be7e8056` |
| TMCAD | `none` | `runs/finetune/20260804-142015_tmcad_ablation_none` | `943684e639c586ddfb70ecedf491f6d4721f2283baa02e46811d87f97443b0bb` |
| TMCAD | `typewise_global` | `runs/finetune/20260804-141550_tmcad_ablation_typewise_global` | `6d7b1a26f077fbc04fcccce5628318230682bb22f576565f2576cd350430ed40` |

每个 run directory 均包含 `checkpoints/best.pt`、`checkpoints/last.pt`、`test_metrics.json` 和 `test_evaluate.log`。后台调度及完整日志见：

- `scripts/run_mfcadpp_tmcad_preprocessing_ablation.sh`
- `runs/launch_logs/20260804-133604_mfcadpp_tmcad_preprocessing_ablation.summary.log`
- `runs/launch_logs/20260804-133604_{dataset}_{mode}_train.log`

