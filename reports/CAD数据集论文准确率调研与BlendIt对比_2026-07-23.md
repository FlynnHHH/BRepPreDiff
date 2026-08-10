# Fusion360Seg、MFCAD++、TMCAD、FabWave 论文准确率调研与 BlendIt 对比

> 检索截止：2026-07-23  
> BlendIt 实验结果更新：2026-08-07  
> 排序指标：Accuracy（Acc，%），同一论文的多个版本优先采用正式发表版或最新版。  
> `NR` 表示论文明确使用了该数据集，但可访问正文、主表或摘要没有披露可核验的该数据集 Accuracy。

## 结论摘要

- **MFCAD++ 数值上位列第 8/21，但不是严格的测试隔离对比。** 最新最佳结果来自 **Edge Update Attention + MLP 微调头**：Acc **99.3632%**、Macro-F1 **98.9913%**、Macro-IoU **98.0172%**。它比 Hierarchical CADNet 高 **1.99 pp**，比 BrepMFR 低 **0.40 pp**。
- **TMCAD 只能在注明版本与预训练协议后比较。** 最新 Edge Update Attention + DiffLoss 结果为 Acc **83.9926%**、Macro-F1 **83.4995%**、Macro-IoU **72.3311%**；数值上比 BRT 的 83.45% 高 **0.54 pp**，但低于 Brep2Shape 的 84.72%。有效样本、划分和预训练协议不同，不能把差值解释为纯模型收益。
- **Fusion360Seg 现在有两套不同任务结果。** 标准 8 类任务上的最新最佳结果来自 **Edge Update Attention + MLP 微调头**：Acc **93.0596%**、Macro-F1 **87.3927%**、Macro-IoU **78.6993%**，数值上排第 **9/15**；本地 3 类 `NonTransition / VBF / EBF` 任务仍须与标准 8 类榜单分开报告。
- **FabWave 没有统一的官方划分或稳定的标签版本。** 清洗后 40 类上，最新 **Edge Update Attention + DiffLoss** 与先前 MLP/DiffLoss 实验得到相同测试结果：Acc **97.9540%**、Macro-F1 **99.4929%**、Macro-IoU **99.0794%**。该版本删除了 Rotary Shaft、302 个 Washers/O-Rings 重叠模型及有效样本少于 10 的类别；论文主榜通常采用 45 类且划分不同，因此只作数值参考。
- 本次共整理：Fusion360Seg **14** 篇论文结果加 **1** 条 BlendIt 标准 8 类结果、MFCAD++ **20** 篇论文结果加 **1** 条 BlendIt 本地结果、TMCAD **4** 篇论文结果加 **1** 条 BlendIt 本地结果；FabWave 包含 **5** 条 45 类论文主榜结果、**1** 条 BlendIt 40 类结果、**6** 条派生/特殊协议结果和 **1** 条仅报告 F1/mAP 的结果。

最新 BlendIt 结果来自 **333,476** 个可解析 STEP 的七来源联合无标签预训练，使用
**Edge Update Attention** encoder 预训练 150 epochs（micro-batch 32、梯度累积 8、有效 batch 256）。来源包括
Blendit、TMCAD、Fusion360Seg s2.0.1、MFCAD++、Fusion360Rec、Fusion360Ass 和
FabWave（清洗后 40 类版本）。各下游数据集的 train、validation、test 几何均参与无标签预训练，但微调、
模型选择和最终评估仍分别只使用 train、validation 和 test 标签。因此，这些结果属于
**transductive self-supervised pretraining**，下表中的排名和差值
只表示数值位置，不能解释为与测试几何完全隔离方法之间的纯模型收益。

### 最新 Edge Update Attention 实验汇总

| 下游任务 | 任务 / 微调头 | Best epoch | Test Acc | Macro-F1 | Macro-IoU |
|---|---|---:|---:|---:|---:|
| BlendIt | Seg / MLP | 18 | 98.1210% | 96.2606% | 92.8949% |
| Fusion360Seg | Seg / MLP | 97 | 93.0596% | 87.3927% | 78.6993% |
| MFCAD++ | Seg / MLP | 88 | 99.3632% | 98.9913% | 98.0172% |
| TMCAD | Cls / DiffLoss | 173 | 83.9926% | 83.4995% | 72.3311% |
| FabWave 40 类清洗版 | Cls / DiffLoss | 87 | 97.9540% | 99.4929% | 99.0794% |

## 1. 口径与可比性

### 1.1 数据集版本

| 数据集 | 标准/常见版本 | BlendIt 本地版本 | 是否可直接排序 |
|---|---|---|---|
| Fusion360Seg | 35,858 个模型，8 类 operation face labels | 标准 8 类 cache 为 35,680 个模型，24,964/5,350/5,366；另有 7,659 个模型的本地 3 类 transition task | **有限可比**：8 类任务标签一致，但无标签 test 几何参与预训练；3 类任务不可混排 |
| MFCAD++ | 59,665 个有效模型，24 类加工特征 + Stock；官方 41,766/8,950/8,949 划分 | 同 59,665 个模型、同官方划分、完整 268,982 个测试面 | **有限可比**：标签与监督划分一致，但无标签 test 几何参与预训练 |
| TMCAD / MechCAD | 原始 10 类、约 10,897 个模型；不同论文清洗结果不一致 | 10,897 个源模型、10,886 个有效模型；8,709/1,090/1,087 | **有限可比**：必须同时看有效样本、标签、划分与预训练几何 |
| [FabWave / FabSearch](https://doi.org/10.1115/1.4043211) | 原始仓库超过 10 万个模型；常见有 45 类约 4.5k、UV-Net Standard 52 类及多个派生子集；无官方划分 | 清洗后 40 类、3,989 个有效模型；3,191/407/391；删除 Rotary Shaft、302 个 Washers/O-Rings 重叠模型及 3 个低于 10 条有效数据的类别；完整几何参与无标签预训练 | **不可直接混排**：BlendIt 类别数、模型数、去重、划分和预训练重叠均与 45 类论文主榜不同 |

### 1.2 证据等级

| 等级 | 含义 |
|---|---|
| A | 同一公开数据集，Accuracy 来自原论文正文、主表或正式摘要，协议基本可核对 |
| A− | 同一数据集家族，但存在小幅版本、复现或训练设置差异 |
| B | 自定义划分、额外监督、预训练重叠，或数值需由后续论文同协议对比表核验 |
| C | 派生标签体系、显著不同的数据量/任务，或包含人工复核；仅供参考 |

> Accuracy 排序只回答“论文报告了多高的数值”，不自动代表公平的模型优劣排序。跨版本、跨划分和跨标签体系的结果必须结合“等级/口径”列阅读。

## 2. Fusion360Seg

### 2.1 标准 8 类任务可核验结果（Acc 降序）

| 排名 | 论文 / 方法 | 年份 | Acc | mIoU | 等级 | 实验口径 |
|---:|---|---:|---:|---:|:---:|---|
| 1 | [TopoGNN](https://doi.org/10.2139/ssrn.6604901) | 2026 | **97.21** | 87.57 | B | SSRN 预印本；公开摘要给出 Fusion360 逐面 Acc/mIoU，完整划分细节未披露 |
| 2 | [Masked BRep Autoencoder via Hierarchical Graph Transformer](https://arxiv.org/abs/2603.14927) | 2026 | **97.02** | 86.75 | B | 35,858；70/15/15；训练划分参与自监督预训练，测试集隔离 |
| 3 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | **96.88** | 83.77 | B | 35,858、8 类；预训练语料包含 Fusion360Seg；取主表最佳模型规模 |
| 4 | [BRep-BERT*](https://doi.org/10.1145/3583780.3615237) | 2023 | **96.02** | 85.54 | B | `*` 版本额外使用 temporal supervision；无该监督版本为 95.14/82.88 |
| 5 | [Self-Supervised Representation Learning for CAD / SSL4CAD](https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html) | 2023 | **96.00** | — | B | Construction-based segmentation；训练集约 23,266 个模型 |
| 6 | [CADOps-Net](https://arxiv.org/abs/2208.10555) | 2022 | **95.90** | 84.20 | A | Fusion360 8 类；联合学习 operation type 与 operation step；主表 `w/ JL+` |
| 7 | [Two-level feature reconstruction network](https://www.sciencedirect.com/science/article/pii/S095219762601050X) | 2026 | **94.51** | 78.13 | B | 35,680 个模型并增加 85,511 个实例标注；取 face classifier 结果 |
| 8 | [Bringing Attention to CAD / BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | **94.48** | 79.23 | A | 期刊最终版/v2；35,858、8 类；arXiv v1 曾报告 90.47 |
| **9** | **BlendIt（Edge Update Attention + MLP；全量联合预训练）** | 2026 | **93.0596** | **78.6993** | B | 8 类；35,680 个有效 cache 模型；24,964/5,350/5,366；77,070 个测试面；无标签 test 几何参与七来源联合预训练；100 epochs；按 validation Acc 选择 epoch 97 `best.pt` |
| 10 | [BRepNet](https://openaccess.thecvf.com/content/CVPR2021/html/Lambourne_BRepNet_A_Topological_Message_Passing_System_for_Solid_Models_CVPR_2021_paper.html) | 2021 | **92.52** | 77.10 | A | 数据集原始论文；官方划分；10 次运行均值 |
| 11 | [UV-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | 2021 | **92.30** | 72.40 | A− | 8 类；数值由 CADOps-Net 的同协议对比主表核验 |
| 12 | [FoV-Net](https://arxiv.org/abs/2602.24084) | 2026 | **91.72** | 73.81 | A | 原始与随机旋转测试结果一致；5 次运行均值 |
| 13 | [Uncertainty estimation review](https://doi.org/10.1007/978-3-031-96196-0_4) | 2025 | **≈84.00** | — | C | 由约 16% error 折算；人工复核 10% 后有效 Acc 约 89%，不属于纯模型结果 |
| 14 | [Modified PointNet++](https://doi.org/10.1016/j.cad.2023.103629) | 2024 | **80.46** | — | A | 分层采样 + B-rep face loss + 额外点属性的最终配置 |
| 15 | [BRepGAT](https://doi.org/10.1093/jcde/qwad100) | 2023 | **80.32** | — | A | 标准 8 类；2-layer 主结果；5-layer 为 80.29 |

BlendIt 在该标准 8 类任务上的最佳完整聚合指标来自
**Edge Update Attention + MLP 微调头**：**93.0596% Acc / 87.3927% Macro-F1 /
78.6993% Macro-IoU**。相比先前联合预训练的 baseline encoder + MLP，分别提高
**0.0844 / 1.8658 / 2.2407 pp**。

### 2.2 BlendIt 本地三分类结果

| 方法 | 类别数 | 模型数 | Test Acc | Macro-F1 | mIoU |
|---|---:|---:|---:|---:|---:|
| BlendIt `baseline_default` | 3 | 7,659 | 97.8509% | 95.4612% | 91.4494% |
| BlendIt `ablation_head_mlp_full`（最佳 F1/mIoU） | 3 | 7,659 | 98.1234% | **96.4880%** | **93.3230%** |
| BlendIt `ablation_mlp_encoder_partial` | 3 | 7,659 | 98.1283% | 96.4395% | 93.2313% |
| **BlendIt `joint_all_splits_no_coarse`（先前最高 Acc）** | 3 | 7,659 | **98.1403%** | 96.3957% | 93.1435% |
| BlendIt Edge Update Attention + MLP（新联合预训练） | 3 | 7,659 | 98.1210% | 96.2606% | 92.8949% |

**比较结论：不可直接排位。** BlendIt 的 `NonTransition / VBF / EBF` 与标准数据集的 ExtrudeSide、CutEnd、Fillet 等 8 类操作标签没有一一对应关系，而且本地训练规模仅约为标准数据集的 21%。新 Edge Update 结果为 **98.1210% Acc / 96.2606% Macro-F1 / 92.8949% Macro-IoU**；当前该任务最高 Acc 仍是 `joint_all_splits_no_coarse` 的 **98.1403%**。建议将该实验命名为 **“Fusion-derived 3-class transition task”**，与第 2.1 节的标准 Fusion360Seg 8 类结果分开报告。

### 2.3 使用了 Fusion 数据但未进入 Accuracy 排名

| 论文 | 年份 | 原因 |
|---|---:|---|
| [Segmentation of CAD models using hybrid representation](https://doi.org/10.1016/j.vrih.2025.01.001) | 2025 | 明确使用 Fusion 360 Gallery；可访问摘要未公开数值表。SSRN 题名与期刊版按作者、内容和 DOI 去重为一篇 |
| [BRepMAE](https://arxiv.org/abs/2602.22701) | 2026 | Fusion 360 Gallery 仅用于自监督预训练，未报告 Fusion360Seg 独立测试 Accuracy |

## 3. MFCAD++

### 3.1 全部可核验结果（Acc 降序）

以最新 BlendIt **Edge Update Attention + MLP** 的 **99.3632%** 为本地基准；`Δ` 为“论文 Acc − BlendIt Acc”，单位为百分点（pp）。

| 排名 | 论文 / 方法 | 年份 | Acc | mIoU / Macro-IoU | Δ vs BlendIt | 等级 | 实验口径 |
|---:|---|---:|---:|---:|---:|:---:|---|
| 1 | [BrepMFR](https://doi.org/10.1016/j.cagd.2024.102318) | 2024 | **99.76** | 99.30 | +0.40 | A | MFCAD++ target-domain supervised test；另报跨域 DA 90.32，不作为主榜 |
| 2 | [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | 2026 | **99.66** | 99.32 | +0.30 | B | 清洗后数据；5-fold CV；论文称 validation sets |
| 3 | [EMD-GNN](https://doi.org/10.1016/j.aei.2026.104609) | 2026 | **99.62** | 98.79 | +0.26 | A | 官方 70/15/15 测试；多次运行均值 |
| 4 | [Masked HGT](https://arxiv.org/abs/2603.14927) | 2026 | **99.61** | 99.03 | +0.25 | B | 59,665；自定义 80/10/10；测试隔离 |
| 5 | [BRepMAE / MFCAD2](https://arxiv.org/abs/2602.22701) | 2026 | **99.59** | 98.82 | +0.23 | B | MFCAD2 清洗别名/派生版，59,450 个有效模型；80/10/10 |
| 6 | [SCUT B-Rep GNN](https://zrb.bjb.scut.edu.cn/EN/10.12141/j.issn.1000-565X.230497) | 2025 | **99.53** | 99.15 | +0.17 | A | 期刊摘要直接给出 MFCAD++ Acc/mIoU |
| 7 | [MMNet](https://doi.org/10.32604/cmes.2026.078073) | 2026 | **99.38** | 98.86 | +0.02 | A | 59,655；70/15/15；测试集逐面语义分割 |
| **8** | **BlendIt（Edge Update Attention + MLP；全量联合预训练）** | 2026 | **99.3632** | **98.0172** | **基准** | B | 59,665；官方 41,766/8,950/8,949；完整 268,982 个测试面；无标签 test 几何参与七来源联合预训练；100 epochs；按 validation Acc 选择 epoch 88 `best.pt` |
| 9 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | **99.35** | 98.02 | −0.01 | B | 预训练语料包含 MFCAD++；取主表最佳模型规模 |
| 10 | [FoV-Net](https://arxiv.org/abs/2602.24084) | 2026 | **99.33** | 97.81 | −0.03 | A | 原始与旋转测试一致；5 次运行均值 |
| 11 | [TEGNet](https://www.researchgate.net/publication/401656636) | 2026 | **99.31** | 98.71 | −0.05 | A | 59,655；70/15/15；0.49M 参数 |
| 12 | [MFTReNet](https://doi.org/10.1016/j.aei.2024.102721) | 2024 | **99.30** | 98.63 | −0.06 | A | 同时执行实例识别与拓扑关系任务；结果由后续同协议主表交叉核对 |
| 13 | [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | **99.27** | 97.98 | −0.09 | A | 期刊最终版/v2；70/15/15；v1 为 99.26/97.94 |
| 14 | [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | 2024 | **99.26** | 98.66 | −0.10 | A− | 不同论文复现约 99.24–99.33；此处采用多篇主表一致值 |
| 15 | [MSFRNet](https://doi.org/10.1016/j.aei.2026.104365) | 2026 | **99.19** | — | −0.17 | B | 数据集特定 Acc 由 Topo-Geom DualGNN 对比表核验；不把摘要的跨数据集最大值反推给 MFCAD++ |
| 16 | [BRepGAT / MFCAD18++](https://doi.org/10.1093/jcde/qwad100) | 2023 | **99.10** | — | −0.26 | C | 作者重标注的 MFCAD18++ 派生版；不是原始 MFCAD++ 标签体系 |
| 17 | [Sheet-metalNet](https://doi.org/10.1038/s41598-024-61443-2) | 2024 | **98.86** | — | −0.50 | A− | 480 epoch 的性能上限；标准 100-epoch 主表为 98.43 Acc / 97.44 F1 |
| 18 | [Hierarchical CADNet](https://doi.org/10.1016/j.cad.2022.103226) | 2022 | **97.37** | — | −1.99 | A | MFCAD++ 数据集原始论文；Edge 版本逐面准确率 |
| 19 | [Uncertainty estimation review](https://doi.org/10.1007/978-3-031-96196-0_4) | 2025 | **≈96.20** | — | −3.16 | C | 由约 3.8% error 折算；人工复核 10% 后有效 Acc 约 99.3%，不是纯模型结果 |
| 20 | [Modified PointNet++](https://doi.org/10.1016/j.cad.2023.103629) | 2024 | **95.85** | — | −3.51 | A | 最终 PointNet++ 配置；同文 Point Transformer 最终同为 95.85 |
| 21 | [Shaft process-planning model](https://doi.org/10.3390/app16020828) | 2026 | **90.87** | — | −8.49 | C | 轴类工艺规划应用中的迁移/外部验证；F1=89.85 |

### 3.2 BlendIt 本地结果对照

| 方法 | Test Acc | Macro-F1 | Macro-IoU |
|---|---:|---:|---:|
| BlendIt DiffLoss（全量联合预训练，200 epochs） | 99.1248% | 98.6323% | 97.3280% |
| BlendIt baseline encoder + MLP（全量联合预训练，200 epochs） | 99.1557% | 98.6718% | 97.4071% |
| **BlendIt Edge Update Attention + MLP（新联合预训练，100 epochs）** | **99.3632%** | **98.9913%** | **98.0172%** |

新 Edge Update Attention 结果相对先前 baseline encoder + MLP 提高
**0.2075 pp Acc**、**0.3195 pp Macro-F1** 和 **0.6101 pp Macro-IoU**。两次实验的
预训练 encoder 和联合语料版本同时变化，因此该差值不是单一架构消融结论。

### 3.3 使用了 MFCAD++ 但未进入 Accuracy 排名

| 论文 | 年份 | 原因 |
|---|---:|---|
| [SFRGNN-DA](https://doi.org/10.1016/j.jmsy.2025.05.005) | 2025 | 明确在 MFCAD++ 上做语义分割，但可访问摘要/索引页未披露数据集特定 Accuracy |
| [Semantic Direct Modeling](https://www.researchgate.net/publication/390989999_Semantic_Direct_Modeling) | 2025 | 清洗后 57,992 个模型；报告生成模块 IoU=98.50%、ME=96.73%，未报告传统逐面 Accuracy |
| [AgentsCAD](https://arxiv.org/abs/2607.02448) | 2026 | GraphSAGE 在 MFCAD++ 上训练并作为可选语义模块，但没有独立 MFCAD++ test Accuracy |

## 4. TMCAD / MechCAD

### 4.1 全部可核验结果（Acc 降序）

以最新 BlendIt **Edge Update Attention + DiffLoss** 的 **83.9926%** 为本地基准。由于各论文的数据清洗、标签、划分与预训练协议差异明显，以下 `Δ` 只作数值参考。

| 排名 | 论文 / 方法 | 年份 | Acc | Δ vs BlendIt | 等级 | 数据版本与协议 |
|---:|---|---:|---:|---:|:---:|---|
| 1 | [KDH-CAD](https://arxiv.org/abs/2606.01702) | 2026 | **95.82** | +11.83 | C | 清洗重标注 9,799；Bolt/Screw 合并并增加 Spring；仅 1,000 train shots，固定 500 val/500 test；Macro-F1=94.47 |
| 2 | [TopoGNN](https://doi.org/10.2139/ssrn.6604901) | 2026 | **88.50** | +4.51 | B | SSRN 预印本；摘要报告 TMCAD shape-level Acc，具体有效样本与划分未完整披露 |
| 3 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | **84.72** | +0.73 | B | 仅保留 7,599 个有效文件；取最佳模型规模；默认 100 epoch 为 82.64，350 epoch 为 84.03 |
| **4** | **BlendIt（Edge Update Attention + DiffLoss；全量联合预训练）** | 2026 | **83.9926** | **基准** | B | 原始 MechCAD 10 类；10,897 个源模型、10,886 个有效模型；8,709/1,090/1,087；无标签 test 几何参与七来源联合预训练；200 epochs；按 validation Acc 选择 epoch 173 `best.pt` |
| 5 | [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | **83.45** | −0.54 | A− | 原始 TMCAD 家族、10 类、70/15/15；期刊最终版；v1/MechCAD 曾报告 82.01 |

### 4.2 BlendIt 本地结果对照

| 方法 | Test Acc | Macro-F1 | Macro-IoU |
|---|---:|---:|---:|
| BlendIt baseline encoder + DiffLoss（先前联合预训练，200 epochs） | 84.2686% | 83.9380% | 73.2085% |
| BlendIt MLP（全量联合预训练，200 epochs） | 81.8767% | 81.3106% | 69.4729% |
| **BlendIt Edge Update Attention + DiffLoss（新联合预训练，200 epochs）** | **83.9926%** | **83.4995%** | **72.3311%** |

新 Edge Update Attention 结果相对先前 baseline encoder + DiffLoss 低
**0.2760 pp Acc**、**0.4385 pp Macro-F1** 和 **0.8774 pp Macro-IoU**。两次实验的
联合语料版本也不同，不应将差值解释为单一 encoder 效应。

**最接近的数据版本参照是 BRT。** 它仍使用原始 TMCAD 10 类家族，但采用 70/15/15 划分，而 BlendIt 约为 80/10/10；此外 BlendIt 的无标签 test 几何参与了联合预训练。BlendIt 数值高 **0.54 pp**；该差距可用于定位，但不应写成严格同协议优越性。

## 5. FabWave

### 5.1 常见约 45 类全量监督口径（Acc 降序）

FabWave 没有官方 train/test split，公开论文对损坏文件、稀有类别和重复模型的
过滤也不一致。下表以 **45 类、约 4.5k 个模型**的全量监督分类结果为主，并加入
BlendIt 的清洗后 40 类本地结果作为数值参考；排名表示已报告数值的位置，不代表严格同协议比较。

| 排名 | 论文 / 方法 | 年份 | Acc | 等级 | 数据版本与协议 |
|---:|---|---:|---:|:---:|---|
| 1 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | **99.99** | B | 4,572 个模型、45 类；6-layer 模型在 Brep2Shape-250k 上预训练后微调 100 epochs；预训练语料本身含 3,270 个 FabWave 模型；99.99 为主表/最佳种子值，附录三种子均值为 99.74±0.43 |
| 2 | [Bringing Attention to CAD / BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | **98.95** | A− | 45 类 FabWave；期刊最终版/v2；数据集无官方划分，具体有效文件与随机划分需随实现记录 |
| 3 | [VGNet](https://doi.org/10.1109/TMM.2024.3521706) | 2025 | **98.00** | A− | 4,475 个模型、45 类；融合多视图与 B-rep attributed graph；同文 retrieval mAP=92.90 |
| **4** | **BlendIt（Edge Update Attention + DiffLoss；全量联合预训练）** | 2026 | **97.9540** | C | 清洗后 40 类、3,989 个有效模型；3,191/407/391；按最高 validation Acc 选择 epoch 87 `best.pt`；无标签 test 几何参与七来源联合预训练；200 epochs；类别与划分不同，仅作数值参考 |
| 5 | [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | 2024 | **96.33** | B | AAGNet 原本面向分割；此处采用 Brep2Shape 按分类任务适配并训练 350 epochs 的同表基线，不是 AAGNet 原论文主任务结果 |
| 6 | [UV-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | 2021 | **92.68** | B | 采用 Brep2Shape 的 45 类同表复现值；UV-Net 原论文的 52 类 Standard 子集结果为 94.51%，见第 5.2 节 |

BlendIt 最新 **Edge Update Attention + DiffLoss** 与先前 MLP/DiffLoss 得到完全相同的测试结果：**97.9540% Acc /
99.4929% Macro-F1 / 99.0794% Macro-IoU / 97.9461% Weighted-F1**，391 个 test
样本中正确 383 个。新 checkpoint 位于 epoch 87，validation Acc 为
**96.8059%**。本地数据处理依次删除
250 个 `Rotary_Shaft`、302 个与 O-Rings 重叠的 Washers，以及有效样本少于 10 的
`Webbing Guide`（0）、`Miter Gears`（1）和 `Sleeve Washers`（7）；最终保留 40 类。
由于论文主榜通常采用 45 类，不能将表中第 4 的数值位置解释为严格方法排名。

### 5.2 派生标签、少样本与零样本协议

| 论文 / 方法 | 年份 | FabWave 版本 | Acc / 主要结果 | 等级 | 为什么不进入 45 类主榜 |
|---|---:|---|---:|:---:|---|
| [Self-Supervised Representation Learning for CAD / SSL4CAD](https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html) | 2023 | 26 类、最高 3,125 个训练样本 | **100.00%**（四舍五入） | C | 只保留至少有 3 个兼容样本的类别，并移除 2 个仅朝向不同的类别；完整监督点为 10 次运行均值，表中只保留两位小数；face codes 在 Fusion360 Gallery 上预训练 |
| [KDH-CAD](https://arxiv.org/abs/2606.01702) | 2026 | 7 类、2,662 个模型 | **99.84%** | C | 从 45 类中只选样本数不少于 200 的类别，并将重复率约 83% 的 Rotary Shaft/Keyway Shaft 合并；KDH-CAD 仅用 350 个训练样本，另使用工程知识和冻结的 Qwen3-VL；同划分 BRT 为 99.68% |
| [MVCNN++](https://doi.org/10.1115/1.4047486) | 2021 | FW10C、10 类 | **95.45±1.00%** | C | 10 类多视图派生集，ImageNet 预训练并融合尺寸元数据；5-fold validation，不是 45 类 B-rep 分类协议 |
| [PP-Brep](https://openaccess.thecvf.com/content/CVPR2026/html/Hao_PP-Brep_Few-Shot_B-rep_Classification_with_Hybrid_Graph_Representation_CVPR_2026_paper.html) | 2026 | FabWave-31、2,775 个模型 | **72.98±3.84%**（1-shot）；**87.55%**（5-shot） | C | 31 类派生集；在 DeepCAD 上预训练后做 1/3/5-shot graph prompt，不能与全量监督混排 |
| [BRepCLIP](https://arxiv.org/abs/2606.05515) | 2026 | 清洗后 4,378 个模型、39 类 | **38.62% Top-1** | C | 在 ABC 上训练，FabWave 完全不微调；按类别文本描述做严格 zero-shot，另报 Top-5=70.28、Top-10=86.71 |
| [UV-Net（原论文）](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | 2021 | Standard 子集、52 类 | **94.51±0.10%** | C | 从原数据提供的 56 类中移除极少或无有效模型的 4 类，并按类随机 80/20；类别数不同于当前常见 45 类版本 |

### 5.3 使用了 FabWave 但未进入 Accuracy 排名

| 论文 | 年份 | 原因 |
|---|---:|---|
| [CADGCL](https://doi.org/10.1007/s00371-025-03949-y) | 2025 | 4,475 个模型、45 类、80/20；报告 F1=98.84%、retrieval mAP@50=89.35%、mAP@10=98.58%，但未给出 Accuracy，故不由 F1 反推 Acc |

### 5.4 FabWave 比较结论

- **最接近当前 45 类全量基准的是 BRT、VGNet 与 Brep2Shape 主表。** 即使类别数相同，4,475、4,504、4,572 等有效模型统计和随机划分仍不一致。
- **Brep2Shape 的 99.99% 不能直接视为严格 test-isolated SOTA。** 其 250k 预训练集包含 3,270 个 FabWave 模型；论文未证明这些模型与下游 test split 完全去重隔离。更稳妥的复现实验参照是附录三种子均值 **99.74±0.43%**。
- **BlendIt baseline MLP/DiffLoss 与新 Edge Update Attention + DiffLoss 在清洗后 40 类协议上均为 97.9540% Acc。** 数值上比 Brep2Shape/BRT 的 45 类结果低 2.04/1.00 pp，比 VGNet 低 0.05 pp；类别、有效模型和划分不同，差值只作定位。
- **FabWave 很容易受近重复、参数变体和类别过滤影响。** 后续实验应固定文件 manifest、哈希去重、类别表和随机种子，并同时报告 Macro-F1。

## 6. 建议用于论文/报告的表述

### Fusion360Seg 与 Fusion-derived 三分类任务

> With seven-source transductive self-supervised pretraining, an Edge Update Attention encoder, and an MLP fine-tuning head, BlendIt achieved 93.06% face-level accuracy, 87.39% macro-F1, and 78.70% macro-IoU on the eight-class Fusion360Seg task. Because unlabeled test geometry was included during pretraining, this result is reported separately from test-isolated inductive comparisons. On the local three-class transition task, the same encoder achieved 98.12% accuracy and 92.89% macro-IoU.

### MFCAD++

> On the official MFCAD++ split, BlendIt with an Edge Update Attention encoder and an MLP fine-tuning head achieved 99.36% face-level accuracy, 98.99% macro-F1, and 98.02% macro-IoU. Numerically, its accuracy is 1.99 percentage points above the original Hierarchical CADNet result and 0.40 percentage points below the highest verified result identified in this survey. Because unlabeled test geometry was included during transductive self-supervised pretraining, these differences should not be interpreted as gains under a strictly test-isolated protocol.

### TMCAD

> On the original ten-class TMCAD/MechCAD taxonomy, BlendIt with an Edge Update Attention encoder and a DiffLoss fine-tuning head achieved 83.99% accuracy and 83.50% macro-F1. BRT reports 83.45% on the original dataset family and Brep2Shape reports 84.72% on a more heavily filtered version; the data splits and pretraining protocols differ.

### FabWave

> FabWave results are reported separately by label taxonomy and supervision protocol because the dataset has no official split and published variants contain different numbers of categories and valid models. After removing Rotary Shaft, 302 overlapping Washer/O-Ring models, and classes with fewer than ten valid samples, BlendIt with an Edge Update Attention encoder and a DiffLoss head achieved 97.95% accuracy, 99.49% macro-F1, and 99.08% macro-IoU on a local 40-class split containing 3,989 valid models. On commonly used 45-class variants, Brep2Shape reports 99.99% accuracy (99.74% mean over three seeds) and BRT reports 98.95%; no strict direct comparison is claimed.

## 7. 检索边界与注意事项

- 检索截止 2026-07-23；之后发表或更新的论文不在本表内。
- 本表优先使用论文原文、正式出版页面、arXiv/SSRN 原稿和可核验的同协议主表；无法核验的搜索摘要数字不录入。
- BRT 的 arXiv v1 与期刊最终版数值不同，统一采用期刊最终版/v2。
- AAGNet 等方法在不同论文复现中有约 0.01–0.07 pp 波动，表中采用原论文或多篇主表一致值。
- FabWave 没有官方划分，且公开版本至少包含 52/45/39/31/26/10/7 类口径；同名数据集结果必须连同类别数、有效模型数、划分和监督协议一起引用。
- Brep2Shape 主表的 FabWave 99.99% 为最佳种子/主结果，附录三种子均值为 99.74±0.43%；其预训练语料包含 3,270 个 FabWave 模型，因此标为 B 级而非严格隔离的 A 级证据。
- 不同论文对 `mIoU`、`IoU`、`Macro-IoU` 的平均方式可能不同，不能仅凭名称相同就视为严格同定义。
- “所有论文”按公开可检索、明确说明使用目标数据集且能去重识别的论文理解；受限全文或仅在正文中隐含数据集名称的工作可能仍有漏检。

## 8. 本地证据文件

- `finetune_baseline_ablation_results.md`：Fusion-derived 三分类任务与消融结果
- `mfcad_finetune_results.md`：MFCAD++ 官方划分与完整测试集结果
- `tmcad_finetune_results.md`：TMCAD/MechCAD 原始 10 类结果
- `joint_all_splits_no_coarse_report.md`：2026-07-24 更新的联合无 coarse-label 预训练、四项微调与完整测试结果
- `joint_fusion_gallery_all_unlabeled_mlp_diffloss_2026-08-05.md`：334,036 个 STEP 的七来源联合预训练与 DiffLoss 四项测试结果
- `joint_fusion_gallery_mlp_head_benchmarks_2026-08-05.md`：同一预训练 checkpoint 下的 MLP 四项测试结果及与 DiffLoss 的受控对比
- `fabwave_min10_diffloss_acc_results_2026-08-06.md`：FabWave 40 类清洗口径、按最高 validation Acc 选取的 DiffLoss checkpoint、完整测试指标与错误构成
- `fabwave_min10_misclassified_ids_2026-08-06.txt`：FabWave 最新 test split 的 8 个误分类样本 ID
- `max_acc_best_checkpoint_retest_2026-08-06.md`：三项主数据集在 max-accuracy `best.pt` 策略下的 DiffLoss 重训与测试结果
- `mlp_vs_diffloss_max_acc_results_2026-08-07.md`：四项任务共 4 组 MLP 与 4 组 DiffLoss 的统一 max-accuracy 对比、run 路径和 checkpoint 哈希
- `runs/edge_update_new_joint/`：333,476 图七来源 Edge Update Attention 预训练 checkpoint，以及 BlendIt/Fusion360Seg/MFCAD++ MLP 和 TMCAD/FabWave DiffLoss 的完整测试指标
