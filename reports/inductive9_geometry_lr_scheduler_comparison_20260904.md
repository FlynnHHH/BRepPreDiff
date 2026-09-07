# Inductive 9-source encoder：Geometry-only、预训练 LR 与 Cosine Decay 对比

日期：2026-09-04

## 结论摘要

- 在协议严格一致的 `lr=1e-4` 四格对照中，**Full-loss + Constant** 的 14 组平均 Accuracy 最高，为 **96.3928%**；**Geometry-only + Cosine** 的平均 Macro-F1（**94.9591%**）和 mIoU（**91.3355%**）最高。
- 去掉离散预训练损失并不是稳定增益。Constant LR 下，Geometry-only 相对 Full-loss 的平均 Accuracy 为 **-0.1888 pp**；虽然 14 组中赢 8 组，但 TMCAD MLP/DiffLoss 分别下降 **1.1039/1.8399 pp**。
- Cosine 的效果依赖预训练目标。Full-loss 下，Cosine 相对 Constant 平均 **-0.1721 pp**；Geometry-only 下则平均 **+0.1280 pp**。该交互主要体现在 DiffLoss 和 TMCAD，单 seed 下不宜解释为普遍规律。
- Geometry-only + Cosine 对 DiffLoss 较有利：相对 Geometry-only + Constant，7 个 DiffLoss 任务赢 6、负 1，平均 **+0.2736 pp**；MLP 平均为 **-0.0175 pp**。
- `lr=1e-3` 只有旧的 pretrain-50/finetune-100 结果。它与当前 pretrain-100/finetune-200 协议不一致，因此本报告仅把 LR 对照列为历史参考，不作独立 LR 因果结论。

## 实验与可比性

所有实验均使用 inductive 9-source（无 FabWave）、full-data 下游划分、seed 42，分类任务使用 Mean+Max pooling。Full-loss 的 categorical/relation 权重为 `0.5/0.3`；Geometry-only 为 `0/0`。

| ID | 离散预训练损失 | Pretrain LR | Scheduler | Pretrain epochs | Finetune epochs | MLP/DiffLoss | 可比性 |
|---|---|---:|---|---:|---:|---|---|
| A | Full-loss | `1e-3` | Constant | 50 | 100 | 7/7 | 仅历史参考 |
| B | Full-loss | `1e-4` | Constant | 100 | 200 | 7/7 | 严格对照 |
| C | Geometry-only | `1e-4` | Constant | 100 | 200 | 7/7 | 严格对照 |
| D | Full-loss | `1e-4` | Cosine → 0 | 100 | 200 | 7/7 | 严格对照 |
| E | Geometry-only | `1e-4` | Cosine → 0 | 100 | 200 | 7/7 | 严格对照 |

严格对照中的 encoder 预训练数据、batch size 128、下游 task batch size 256/512、梯度累积 1、模型结构与 best-checkpoint 选择规则一致。并行方式只影响任务调度，不改变单个任务配置。

## 配置级整体结果

下表为各配置在 7 个任务 × 2 个 head 上的算术平均；不同任务样本数未加权。单位均为百分比。

| ID | 配置 | Accuracy | Macro-F1 | Weighted-F1 | mIoU |
|---|---|---:|---:|---:|---:|
| A | Full / `1e-3` / Constant（历史） | 96.1224 | 94.6220 | 96.0833 | 90.8991 |
| B | Full / `1e-4` / Constant | **96.3928** | 94.9212 | **96.3646** | 91.2891 |
| C | Geometry / `1e-4` / Constant | 96.2041 | 94.7602 | 96.1713 | 91.0570 |
| D | Full / `1e-4` / Cosine | 96.2207 | 94.7331 | 96.1962 | 91.0375 |
| E | Geometry / `1e-4` / Cosine | 96.3321 | **94.9591** | 96.3246 | **91.3355** |

按 head 拆分的平均 Accuracy：

| 配置 | MLP | DiffLoss |
|---|---:|---:|
| Full / `1e-4` / Constant | **96.3219** | 96.4638 |
| Geometry / `1e-4` / Constant | 96.1955 | 96.2127 |
| Full / `1e-4` / Cosine | 96.1729 | 96.2686 |
| Geometry / `1e-4` / Cosine | 96.1780 | **96.4862** |

## 三个消融维度

### 1. 是否使用 Geometry-only

差值为 `Geometry-only - Full-loss`，单位为 Accuracy 百分点。

| Scheduler | 全部 14 组平均 | MLP 平均 | DiffLoss 平均 | 胜/平/负 |
|---|---:|---:|---:|---:|
| Constant | -0.1888 | -0.1264 | -0.2512 | 8/0/6 |
| Cosine | +0.1114 | +0.0051 | +0.2176 | 7/0/7 |

Constant 下 Geometry-only 在多数任务上是小幅提升，但 TMCAD 的两个较大降幅使整体均值转负。Cosine 下 Geometry-only 的整体正收益主要来自 TMCAD DiffLoss 的 `+1.5639 pp`，因而不能只凭均值判断其普遍优于 Full-loss。

### 2. `lr=1e-3` 与 `lr=1e-4`

现有 LR 对照为 B−A：`1e-4/pretrain100/finetune200 - 1e-3/pretrain50/finetune100`，同时改变了 LR、预训练预算和微调预算。

| 范围 | 平均 ΔAccuracy | 胜/平/负 |
|---|---:|---:|
| 全部 14 组 | +0.2704 | 9/0/5 |
| MLP | +0.2582 | 4/0/3 |
| DiffLoss | +0.2827 | 5/0/2 |
| 排除 TMCAD 两组 | +0.0472 | 7/0/5 |

表面上 `1e-4` 组更高，但总平均提升中的 **3.2198/3.7856 pp 累计差值来自 TMCAD 两个 head**。由于训练预算不一致，该结果只能说明新版完整协议整体优于旧协议，不能证明差异由 LR 单独造成。要回答纯 LR 效应，仍需补跑 Full-loss、Constant、pretrain100/finetune200 的 `lr=1e-3` 对照。

### 3. 是否使用 Cosine Decay

差值为 `Cosine - Constant`，两侧均为 `lr=1e-4`、pretrain100、finetune200。

| 离散损失设置 | 全部 14 组平均 | MLP 平均 | DiffLoss 平均 | 胜/平/负 |
|---|---:|---:|---:|---:|
| Full-loss | -0.1721 | -0.1490 | -0.1952 | 7/0/7 |
| Geometry-only | +0.1280 | -0.0175 | +0.2736 | 8/1/5 |

Cosine 与 Geometry-only 的 Accuracy 交互量为 `(E−C)−(D−B) = +0.3002 pp`；MLP 为 `+0.1315 pp`，DiffLoss 为 `+0.4688 pp`。这说明 Cosine 不能脱离离散损失设置单独选择：它在 Full-loss encoder 上没有带来整体收益，但与 Geometry-only + DiffLoss 的组合表现较好。

## 逐任务 Accuracy 差值

- `Geom@Const`：C−B，Geometry-only 的影响。
- `Cos@Full`：D−B，Full-loss 下 Cosine 的影响。
- `Cos@Geom`：E−C，Geometry-only 下 Cosine 的影响。
- `LR 历史`：B−A，包含训练预算变化，仅供参考。

| Task | Head | Geom@Const | Cos@Full | Cos@Geom | LR 历史 |
|---|---|---:|---:|---:|---:|
| BRepPreDiff | MLP | +0.1039 | -0.0291 | -0.0563 | -0.0651 |
| BRepPreDiff | DiffLoss | +0.0077 | +0.0029 | +0.0234 | -0.0505 |
| Fusion360Seg | MLP | +0.0532 | -0.0091 | -0.0026 | +0.2894 |
| Fusion360Seg | DiffLoss | -0.0104 | +0.0143 | +0.0402 | +0.2154 |
| MFCAD++ | MLP | -0.0342 | -0.0085 | +0.0231 | +0.0126 |
| MFCAD++ | DiffLoss | +0.0171 | +0.0074 | +0.0238 | -0.0007 |
| TMCAD | MLP | -1.1039 | -1.0119 | +0.0000 | +1.5639 |
| TMCAD | DiffLoss | -1.8399 | -1.6559 | +1.7479 | +1.6559 |
| SolidLetters | MLP | +0.1186 | -0.0103 | -0.0877 | -0.0258 |
| SolidLetters | DiffLoss | +0.0773 | +0.2423 | +0.0671 | +0.0980 |
| CADSynth | MLP | +0.0075 | +0.0158 | +0.0033 | -0.0129 |
| CADSynth | DiffLoss | +0.0018 | -0.0075 | -0.0150 | +0.0207 |
| MFInstSeg | MLP | -0.0299 | +0.0100 | -0.0023 | +0.0451 |
| MFInstSeg | DiffLoss | -0.0117 | +0.0299 | +0.0275 | +0.0399 |

## `lr=1e-4` 四格逐任务最佳配置

| Task | Head | Full/Const | Geom/Const | Full/Cos | Geom/Cos | 最佳 |
|---|---|---:|---:|---:|---:|---|
| BRepPreDiff | MLP | 98.7939 | **98.8978** | 98.7648 | 98.8415 | Geom/Const |
| BRepPreDiff | DiffLoss | 98.8027 | 98.8104 | 98.8056 | **98.8338** | Geom/Cos |
| Fusion360Seg | MLP | 95.9258 | **95.9790** | 95.9167 | 95.9764 | Geom/Const |
| Fusion360Seg | DiffLoss | 95.9258 | 95.9154 | 95.9401 | **95.9556** | Geom/Cos |
| MFCAD++ | MLP | **99.4122** | 99.3780 | 99.4037 | 99.4011 | Full/Const |
| MFCAD++ | DiffLoss | 99.3784 | 99.3955 | 99.3858 | **99.4193** | Geom/Cos |
| TMCAD | MLP | **83.8086** | 82.7047 | 82.7967 | 82.7047 | Full/Const |
| TMCAD | DiffLoss | **84.9126** | 83.0727 | 83.2567 | 84.8206 | Full/Const |
| SolidLetters | MLP | 97.4629 | **97.5815** | 97.4526 | 97.4938 | Geom/Const |
| SolidLetters | DiffLoss | 97.3907 | 97.4680 | **97.6330** | 97.5351 | Full/Cos |
| CADSynth | MLP | 99.5730 | 99.5805 | **99.5888** | 99.5838 | Full/Cos |
| CADSynth | DiffLoss | 99.5973 | **99.5991** | 99.5898 | 99.5841 | Geom/Const |
| MFInstSeg | MLP | 99.2767 | 99.2468 | **99.2867** | 99.2445 | Full/Cos |
| MFInstSeg | DiffLoss | 99.2392 | 99.2275 | **99.2691** | 99.2550 | Full/Cos |

四种配置的逐行最佳次数分别为：Full/Constant 3、Geometry/Constant 4、Full/Cosine 4、Geometry/Cosine 3。不存在对所有任务和 head 都占优的单一配置。

## 建议

1. 若需要一个稳健的统一 encoder 默认配置，当前证据优先支持 **Full-loss + `lr=1e-4` + Constant**：平均 Accuracy/Weighted-F1 最佳，且 TMCAD 最稳定。
2. 若重点使用 DiffLoss，并愿意针对 head 选 encoder，**Geometry-only + `lr=1e-4` + Cosine** 值得优先：DiffLoss 平均 Accuracy 最高，但需要额外 seed 验证 TMCAD 增益。
3. 不应依据现有历史结果宣称 `1e-4` 优于 `1e-3`。应补跑同一 pretrain100/finetune200 协议的 `1e-3`，再报告纯 LR 主效应。
4. 当前全部结果只有 seed 42。对 TMCAD 至少补 2 个 seed，可判断较大的交互是否稳定。

## 数据来源

- [A：Full-loss / 1e-3 / Constant（pre50/ft100）](inductive9_no_fab_full_pre50_ft100_seed42_20260831.md)
- [B：Full-loss / 1e-4 / Constant](inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901.md)
- [C：Geometry-only / 1e-4 / Constant](geometry_only_inductive9_pre100_ft200_seed42_20260902.md)
- [D：Full-loss / 1e-4 / Cosine](inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903.md)
- [E：Geometry-only / 1e-4 / Cosine](geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902.md)

