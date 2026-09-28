# Inductive9 rotate-mix：离散属性三组对比实验

生成日期：2026-09-12。三组实验均已完成：每组 1 次 100-epoch 预训练、7 个数据集 × 2 种 head 的 200-epoch 微调与测试，共 42 份下游结果。

## 主要结论

在当前几何表示、全量监督和充分微调条件下，显式离散属性对多数任务最终性能的额外收益有限，但存在数据集和 head 依赖，不能概括为普遍无用。
- **离散输入的作用（C − B）**：Fusion360Seg 的 MLP / DiffLoss Accuracy 分别下降 0.3477 / 0.3270 个百分点，mIoU 分别下降 0.5661 / 0.9947 个百分点；其他数据集没有跨 head 一致的大幅退化。
- **离散监督的作用（B − A）**：TMCAD / MLP Accuracy 下降 1.4719 个百分点、mIoU 下降 2.3098 个百分点；同一数据集的 DiffLoss Accuracy 则提高 0.1840 个百分点，说明结论依赖 head。
- **整体移除的影响（C − A）**：多数任务变化较小；TMCAD / MLP 和 Fusion360Seg 是需要保留关注的退化项。
- 本报告只有 seed 42，不提供统计显著性或性能等价结论；没有测量不提取离散属性的预处理耗时、存储节省或收敛速度收益。

## 1. 三组定义与控制变量

| 组别 | 名称 | 预训练/微调/评估离散输入 | categorical 权重 | relation 权重 |
|---|---|---|---:|---:|
| A | 不关闭离散损失（原基线） | 使用 | 0.5 | 0.3 |
| B | 仅关闭离散损失 | 使用 | 0 | 0 |
| C | 仅几何输入 | 禁用 | 0 | 0 |

离散输入指 `face_surface_type`、`edge_type`、`edge_relation` 三类属性。C 在所有阶段设置 `model.use_discrete_attributes=false`，绕过三类 embedding，并不读取离散监督目标；原缓存字段仍保留，不参与模型计算。保留坐标、法向、曲率、边长、夹角、几何网格采样和图连接关系。保留未使用模块的初始化顺序与 checkpoint 结构，以保持其余参数初始化一致。
三组均保留 MLP 和 DiffLoss 下游 head；预训练离散属性损失与下游 DiffLoss 是不同概念。

| 项目 | 共同设置 |
|---|---|
| 预训练数据 | Inductive 9-source，仅 train splits，无 FabWave；相同数据划分 |
| 预训练 | 100 epochs，batch 128，LR 1e-4，constant scheduler；noise / recon 权重 1.0 / 0.5 |
| 微调 | 200 epochs，全参数微调；各任务优化器、学习率和 head 参数继承基线 |
| 微调 batch | 前五个数据集 128；CADSynth / MFInstSeg 为 256；梯度累积均为 2 |
| 数据增强 | 预训练和微调均为 50% canonical + 50% SO(3) rotation |
| 随机种子 | seed 42；微调 dataloader_seed 42 |
| 分类池化 | Mean+Max |
| 模型选择 | 每 epoch 验证，以 validation accuracy 选择 best checkpoint，再在 test 上评估 |

GPU 调度不同：A 使用 GPU 4；B 在 GPU 0 预训练，C 在 GPU 1 预训练，两组下游在 GPU 0–3 通过互斥锁调度。均为单进程单卡训练，不改变有效 batch。训练日志未发现 OOM 或异常退出。
已逐项核验实际保存配置：42 份微调配置的数据、模型、优化器、训练预算和 head 参数一致（仅实验路径、预训练 checkpoint、离散输入开关不同）；三份预训练配置仅损失权重、离散输入开关和实验路径不同。

## 2. 完整测试结果

各指标单位为 %。Best epoch 是验证集选中的 checkpoint epoch，并不代表提前停止；所有微调均训练至 200 epochs。Samples 为测试 CAD 样本数；分割指标按面计算，分类指标按模型计算。

| 数据集 | Head | 组别 | Best epoch | Samples | Accuracy | Macro-F1 | Weighted-F1 | mIoU |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | A | 200 | 1787 | 98.9901 | 97.3687 | 98.9887 | 94.9185 |
| BRepPreDiff | MLP | B | 135 | 1787 | 98.9144 | 97.1606 | 98.9116 | 94.5318 |
| BRepPreDiff | MLP | C | 180 | 1787 | 98.9532 | 97.3914 | 98.9510 | 94.9631 |
| BRepPreDiff | DiffLoss | A | 65 | 1787 | 98.9862 | 97.3943 | 98.9840 | 94.9674 |
| BRepPreDiff | DiffLoss | B | 167 | 1787 | 99.0095 | 97.4261 | 99.0069 | 95.0259 |
| BRepPreDiff | DiffLoss | C | 147 | 1787 | 98.9551 | 97.4254 | 98.9527 | 95.0267 |
| Fusion360Seg | MLP | A | 188 | 5366 | 96.9275 | 92.2254 | 96.9124 | 86.6362 |
| Fusion360Seg | MLP | B | 162 | 5366 | 97.0845 | 92.0885 | 97.0746 | 86.4786 |
| Fusion360Seg | MLP | C | 192 | 5366 | 96.7367 | 91.8004 | 96.7207 | 85.9125 |
| Fusion360Seg | DiffLoss | A | 190 | 5366 | 97.0183 | 90.9523 | 97.0102 | 85.2473 |
| Fusion360Seg | DiffLoss | B | 199 | 5366 | 96.9378 | 91.1295 | 96.9164 | 85.3073 |
| Fusion360Seg | DiffLoss | C | 182 | 5366 | 96.6109 | 90.5005 | 96.5892 | 84.3126 |
| MFCAD++ | MLP | A | 199 | 8949 | 99.6160 | 99.4271 | 99.6158 | 98.8667 |
| MFCAD++ | MLP | B | 198 | 8949 | 99.6119 | 99.4320 | 99.6117 | 98.8768 |
| MFCAD++ | MLP | C | 192 | 8949 | 99.5944 | 99.4065 | 99.5941 | 98.8271 |
| MFCAD++ | DiffLoss | A | 175 | 8949 | 99.5963 | 99.3842 | 99.5964 | 98.7830 |
| MFCAD++ | DiffLoss | B | 199 | 8949 | 99.5963 | 99.3937 | 99.5961 | 98.8013 |
| MFCAD++ | DiffLoss | C | 187 | 8949 | 99.6160 | 99.4206 | 99.6160 | 98.8540 |
| TMCAD | MLP | A | 50 | 1087 | 86.3845 | 85.8850 | 86.2738 | 75.9397 |
| TMCAD | MLP | B | 153 | 1087 | 84.9126 | 84.3389 | 84.7178 | 73.6299 |
| TMCAD | MLP | C | 81 | 1087 | 85.2806 | 84.9153 | 85.2027 | 74.4692 |
| TMCAD | DiffLoss | A | 110 | 1087 | 86.5685 | 86.2264 | 86.5812 | 76.4183 |
| TMCAD | DiffLoss | B | 168 | 1087 | 86.7525 | 86.4239 | 86.6753 | 76.6551 |
| TMCAD | DiffLoss | C | 155 | 1087 | 86.6605 | 86.2914 | 86.6076 | 76.5335 |
| SolidLetters | MLP | A | 165 | 19392 | 97.4629 | 97.5148 | 97.4466 | 95.3865 |
| SolidLetters | MLP | B | 179 | 19392 | 97.2463 | 97.3194 | 97.2471 | 95.0167 |
| SolidLetters | MLP | C | 120 | 19392 | 97.4216 | 97.4737 | 97.3998 | 95.3147 |
| SolidLetters | DiffLoss | A | 142 | 19392 | 97.1896 | 97.2653 | 97.1864 | 94.9468 |
| SolidLetters | DiffLoss | B | 165 | 19392 | 97.3546 | 97.4163 | 97.3441 | 95.2065 |
| SolidLetters | DiffLoss | C | 185 | 19392 | 97.2102 | 97.3006 | 97.2113 | 95.0598 |
| CADSynth | MLP | A | 99 | 9993 | 99.6488 | 99.4820 | 99.6485 | 98.9710 |
| CADSynth | MLP | B | 123 | 9993 | 99.6589 | 99.5072 | 99.6586 | 99.0206 |
| CADSynth | MLP | C | 106 | 9993 | 99.6531 | 99.4942 | 99.6528 | 98.9951 |
| CADSynth | DiffLoss | A | 157 | 9993 | 99.6617 | 99.5029 | 99.6614 | 99.0120 |
| CADSynth | DiffLoss | B | 125 | 9993 | 99.6589 | 99.5018 | 99.6585 | 99.0098 |
| CADSynth | DiffLoss | C | 122 | 9993 | 99.6403 | 99.4572 | 99.6399 | 98.9221 |
| MFInstSeg | MLP | A | 196 | 6250 | 99.4468 | 99.1326 | 99.4464 | 98.2915 |
| MFInstSeg | MLP | B | 200 | 6250 | 99.4721 | 99.1773 | 99.4717 | 98.3793 |
| MFInstSeg | MLP | C | 194 | 6250 | 99.4961 | 99.2595 | 99.4961 | 98.5400 |
| MFInstSeg | DiffLoss | A | 178 | 6250 | 99.4633 | 99.1462 | 99.4630 | 98.3191 |
| MFInstSeg | DiffLoss | B | 172 | 6250 | 99.4545 | 99.1272 | 99.4546 | 98.2823 |
| MFInstSeg | DiffLoss | C | 176 | 6250 | 99.4574 | 99.1830 | 99.4572 | 98.3907 |

## 3. 配对差值

全部差值以未四舍五入的原始 JSON 计算，单位为百分点；正值表示差值左项组别的指标高于右项组别。

### 3.1 关闭离散监督：B − A

两组都使用离散输入，差异来自移除离散预训练监督。

| 数据集 | Head | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---|---|---:|---:|---:|---:|
| BRepPreDiff | MLP | -0.0757 | -0.2081 | -0.0772 | -0.3867 |
| BRepPreDiff | DiffLoss | +0.0233 | +0.0318 | +0.0228 | +0.0584 |
| Fusion360Seg | MLP | +0.1570 | -0.1369 | +0.1621 | -0.1576 |
| Fusion360Seg | DiffLoss | -0.0804 | +0.1772 | -0.0938 | +0.0600 |
| MFCAD++ | MLP | -0.0041 | +0.0050 | -0.0041 | +0.0101 |
| MFCAD++ | DiffLoss | +0.0000 | +0.0094 | -0.0002 | +0.0183 |
| TMCAD | MLP | -1.4719 | -1.5460 | -1.5560 | -2.3098 |
| TMCAD | DiffLoss | +0.1840 | +0.1974 | +0.0941 | +0.2368 |
| SolidLetters | MLP | -0.2166 | -0.1954 | -0.1995 | -0.3698 |
| SolidLetters | DiffLoss | +0.1650 | +0.1509 | +0.1576 | +0.2597 |
| CADSynth | MLP | +0.0100 | +0.0253 | +0.0100 | +0.0496 |
| CADSynth | DiffLoss | -0.0029 | -0.0011 | -0.0029 | -0.0022 |
| MFInstSeg | MLP | +0.0252 | +0.0448 | +0.0253 | +0.0878 |
| MFInstSeg | DiffLoss | -0.0088 | -0.0190 | -0.0084 | -0.0368 |

### 3.2 移除离散输入：C − B

两组离散损失均为零，比较预训练及下游同时移除离散输入的整体影响；不能单独归因于某一个阶段。

| 数据集 | Head | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---|---|---:|---:|---:|---:|
| BRepPreDiff | MLP | +0.0388 | +0.2308 | +0.0394 | +0.4313 |
| BRepPreDiff | DiffLoss | -0.0544 | -0.0007 | -0.0542 | +0.0008 |
| Fusion360Seg | MLP | -0.3477 | -0.2882 | -0.3538 | -0.5661 |
| Fusion360Seg | DiffLoss | -0.3270 | -0.6290 | -0.3273 | -0.9947 |
| MFCAD++ | MLP | -0.0175 | -0.0255 | -0.0176 | -0.0497 |
| MFCAD++ | DiffLoss | +0.0197 | +0.0270 | +0.0198 | +0.0526 |
| TMCAD | MLP | +0.3680 | +0.5764 | +0.4848 | +0.8393 |
| TMCAD | DiffLoss | -0.0920 | -0.1325 | -0.0677 | -0.1216 |
| SolidLetters | MLP | +0.1753 | +0.1544 | +0.1527 | +0.2980 |
| SolidLetters | DiffLoss | -0.1444 | -0.1156 | -0.1328 | -0.1468 |
| CADSynth | MLP | -0.0057 | -0.0131 | -0.0058 | -0.0255 |
| CADSynth | DiffLoss | -0.0186 | -0.0445 | -0.0186 | -0.0877 |
| MFInstSeg | MLP | +0.0241 | +0.0822 | +0.0243 | +0.1607 |
| MFInstSeg | DiffLoss | +0.0029 | +0.0558 | +0.0026 | +0.1084 |

### 3.3 整体消融：C − A

同时移除离散输入与离散监督，衡量相对原完整方案的综合变化。

| 数据集 | Head | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---|---|---:|---:|---:|---:|
| BRepPreDiff | MLP | -0.0369 | +0.0227 | -0.0377 | +0.0446 |
| BRepPreDiff | DiffLoss | -0.0311 | +0.0311 | -0.0314 | +0.0592 |
| Fusion360Seg | MLP | -0.1907 | -0.4250 | -0.1917 | -0.7237 |
| Fusion360Seg | DiffLoss | -0.4074 | -0.4518 | -0.4210 | -0.9347 |
| MFCAD++ | MLP | -0.0216 | -0.0205 | -0.0217 | -0.0396 |
| MFCAD++ | DiffLoss | +0.0197 | +0.0364 | +0.0196 | +0.0709 |
| TMCAD | MLP | -1.1040 | -0.9697 | -1.0712 | -1.4705 |
| TMCAD | DiffLoss | +0.0920 | +0.0649 | +0.0264 | +0.1152 |
| SolidLetters | MLP | -0.0413 | -0.0411 | -0.0468 | -0.0718 |
| SolidLetters | DiffLoss | +0.0206 | +0.0353 | +0.0249 | +0.1130 |
| CADSynth | MLP | +0.0043 | +0.0122 | +0.0043 | +0.0241 |
| CADSynth | DiffLoss | -0.0215 | -0.0456 | -0.0215 | -0.0899 |
| MFInstSeg | MLP | +0.0493 | +0.1269 | +0.0497 | +0.2485 |
| MFInstSeg | DiffLoss | -0.0059 | +0.0367 | -0.0058 | +0.0717 |

## 4. 结果解读与边界

1. 显式类型信息可能与已有几何表示冗余。当前夹角、法向、曲率等保留信息可能让模型恢复部分类型线索；这是与结果相容的解释，尚非直接验证的机制。
2. 全量标签和 200 epochs 微调可能补偿预训练表示差异。本实验不能直接回答少标注、冻结编码器、跨域迁移或较短训练预算下离散属性是否有益。
3. 若要据此决定默认移除离散属性，应优先在 Fusion360Seg 和 TMCAD 上做多个配对随机种子复核，并保留 Accuracy 与 Macro-F1 / mIoU 的共同判断。当前未新增这些实验。
4. 三组不是完整的二因素设计：没有“禁用离散输入但保留离散监督”的第四组，因此无法完整估计输入和监督之间的交互作用。
5. 不将不同数据集的 Accuracy 混合成一个总体分数：分类与分割指标统计单位不同，且部分分割任务已接近饱和。

## 5. 来源与可追溯性

数值直接读取 42 份原始测试 JSON，而非对已有 Markdown 中的四位小数再次计算。每个配对均核验测试样本数一致；原始 JSON 包含评估 checkpoint、验证选优 epoch、指标和混淆矩阵。

- [完整结果 CSV](inductive9_rotate_mix_discrete_attributes_three_way_pre100_ft200_seed42_20260912.csv)：保留原始 0–1 指标、来源路径、JSON SHA-256、实际配置路径及预训练 checkpoint。
- A 组原报告：[inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final](inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final.md)。
- B 组原报告：[inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910](inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910.md)。
- C 组原报告：[inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910](inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910.md)。

| 数据集 | Head | 组别 | 原始测试 JSON |
|---|---|---|---|
| BRepPreDiff | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| BRepPreDiff | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/brepprediff_seg_mlp.json) |
| BRepPreDiff | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/brepprediff_seg_mlp.json) |
| BRepPreDiff | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| BRepPreDiff | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/brepprediff_seg_diffloss.json) |
| BRepPreDiff | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/brepprediff_seg_diffloss.json) |
| Fusion360Seg | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| Fusion360Seg | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/fusion360seg_mlp.json) |
| Fusion360Seg | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/fusion360seg_mlp.json) |
| Fusion360Seg | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| Fusion360Seg | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/fusion360seg_diffloss.json) |
| Fusion360Seg | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/fusion360seg_diffloss.json) |
| MFCAD++ | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| MFCAD++ | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/mfcadpp_seg_mlp.json) |
| MFCAD++ | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/mfcadpp_seg_mlp.json) |
| MFCAD++ | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| MFCAD++ | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/mfcadpp_seg_diffloss.json) |
| MFCAD++ | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/mfcadpp_seg_diffloss.json) |
| TMCAD | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| TMCAD | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/tmcad_cls_mlp.json) |
| TMCAD | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/tmcad_cls_mlp.json) |
| TMCAD | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| TMCAD | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/tmcad_cls_diffloss.json) |
| TMCAD | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/tmcad_cls_diffloss.json) |
| SolidLetters | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-083543_full_solidletters_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| SolidLetters | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/solidletters_cls_mlp.json) |
| SolidLetters | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/solidletters_cls_mlp.json) |
| SolidLetters | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-083544_full_solidletters_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| SolidLetters | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/solidletters_cls_diffloss.json) |
| SolidLetters | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/solidletters_cls_diffloss.json) |
| CADSynth | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| CADSynth | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/cadsynth_seg_mlp.json) |
| CADSynth | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/cadsynth_seg_mlp.json) |
| CADSynth | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| CADSynth | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/cadsynth_seg_diffloss.json) |
| CADSynth | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/cadsynth_seg_diffloss.json) |
| MFInstSeg | MLP | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| MFInstSeg | MLP | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/mfinstseg_seg_mlp.json) |
| MFInstSeg | MLP | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/mfinstseg_seg_mlp.json) |
| MFInstSeg | DiffLoss | A | [JSON](../runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json) |
| MFInstSeg | DiffLoss | B | [JSON](../runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910/results/mfinstseg_seg_diffloss.json) |
| MFInstSeg | DiffLoss | C | [JSON](../runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910/results/mfinstseg_seg_diffloss.json) |

代码与实验报告基于已提交的 `1ba15ba`；本次仅汇总结果，没有重新训练或修改既有测试指标。
