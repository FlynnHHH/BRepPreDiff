# CAD 数据集论文准确率调研与 BRepPreDiff inductive9 对比

> 检索截止：2026-09-02
>
> BRepPreDiff 结果来源：`inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901.md`
>
> 排序指标：Accuracy（Acc，%）；`NR` 表示无法从可访问原文核验该项。

## 结论摘要

- 本报告中的 BRepPreDiff 数字已全部统一为指定实验：四层 Edge Update Attention encoder 在九来源的 **train geometry** 上无标签预训练 100 epochs（lr=`1e-4`），下游以 seed 42 微调 200 epochs，按 validation Acc 选 `best.pt`；分类统一用 Mean+Max pooling。
- 这是 **inductive** 协议：所有下游 validation/test geometry 均未参与预训练。预训练共 488,099 个图，checkpoint SHA-256 为 `c4c838c54199f592847a66456dc38e44194e44198092556a469a7ad2b6782a0e`。
- Fusion360Seg：MLP 与 DiffLoss Acc 同为 **95.9258%**；MLP 的 Macro-F1/mIoU 更高，为 **89.5940% / 82.7926%**，数值列第 **6/15**。
- MFCAD++：MLP 为 **99.4122% Acc / 99.0716% Macro-F1 / 98.1722% mIoU**，数值列第 **7/21**；比 Hierarchical CADNet 高 2.04 pp，比 BrepMFR 低 0.35 pp。
- TMCAD：DiffLoss 为 **84.9126% / 84.3891% / 73.7466%**，数值列第 **3/5**；比 Brep2Shape 高 0.19 pp、比 BRT 高 1.46 pp，但数据清洗和划分不同。
- CADSynth：DiffLoss 为 **99.5973% / 99.3385% / 98.6904%**，数值列第 **6/7**；比 Hierarchical CADNet 高 0.07 pp，比 BrepMFR 低 0.36 pp。
- MFInstSeg：MLP 为 **99.2767% / 98.7694% / 97.5963%**，在可核验语义结果中数值列第 **6/9**。本地是 80/10/10，而主要论文采用约 70/15/15，只作有限比较。
- 指定实验明确排除 FabWave，因此本版只保留其论文调研，不再混入旧的 transductive 本地结果。

## 1. BRepPreDiff 统一实验结果

粗体为同任务 Accuracy 更高的头；Fusion360Seg Acc 相同，按 Macro-F1/mIoU 选择 MLP 为正文主结果。

| Task | Head | Best epoch | Test samples | Acc | Macro-F1 | Weighted-F1 | mIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 154 | 1,787 | 98.7939% | **96.9602%** | 98.7902% | **94.1623%** |
| BRepPreDiff | **DiffLoss** | 109 | 1,787 | **98.8027%** | 96.8796% | **98.7978%** | 94.0120% |
| Fusion360Seg | **MLP** | 143 | 5,366 | **95.9258%** | **89.5940%** | 95.8991% | **82.7926%** |
| Fusion360Seg | DiffLoss | 194 | 5,366 | **95.9258%** | 88.5822% | **95.8994%** | 81.8155% |
| MFCAD++ | **MLP** | 185 | 8,949 | **99.4122%** | **99.0716%** | **99.4120%** | **98.1722%** |
| MFCAD++ | DiffLoss | 197 | 8,949 | 99.3784% | 99.0143% | 99.3781% | 98.0617% |
| TMCAD | MLP | 103 | 1,087 | 83.8086% | 83.2922% | 83.6475% | 72.2066% |
| TMCAD | **DiffLoss** | 136 | 1,087 | **84.9126%** | **84.3891%** | **84.7557%** | **73.7466%** |
| SolidLetters | **MLP** | 170 | 19,392 | **97.4629%** | **97.5259%** | **97.4565%** | **95.3956%** |
| SolidLetters | DiffLoss | 139 | 19,392 | 97.3907% | 97.4568% | 97.3840% | 95.2732% |
| CADSynth | MLP | 62 | 9,993 | 99.5730% | 99.2922% | 99.5726% | 98.5993% |
| CADSynth | **DiffLoss** | 78 | 9,993 | **99.5973%** | **99.3385%** | **99.5968%** | **98.6904%** |
| MFInstSeg | **MLP** | 194 | 6,250 | **99.2767%** | **98.7694%** | **99.2758%** | **97.5963%** |
| MFInstSeg | DiffLoss | 186 | 6,250 | 99.2392% | 98.7303% | 99.2384% | 97.5235% |

| Task | ΔAcc | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Acc winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | +0.0087 | −0.0806 | +0.0076 | −0.1503 | DiffLoss |
| Fusion360Seg | +0.0000 | −1.0119 | +0.0004 | −0.9771 | Tie |
| MFCAD++ | −0.0338 | −0.0573 | −0.0338 | −0.1105 | MLP |
| TMCAD | +1.1040 | +1.0969 | +1.1082 | +1.5400 | DiffLoss |
| SolidLetters | −0.0722 | −0.0690 | −0.0725 | −0.1224 | MLP |
| CADSynth | +0.0243 | +0.0463 | +0.0242 | +0.0912 | DiffLoss |
| MFInstSeg | −0.0375 | −0.0391 | −0.0373 | −0.0728 | MLP |

> 项目对新下游实验默认 100 epochs；上表保留 200 epochs，是因为本报告按指定的既有实验原协议汇总。

## 2. 数据与证据口径

| 数据集 | 论文常见版本 | 本地版本 | 可比性 |
|---|---|---|---|
| Fusion360Seg | 35,858、8 类 operation face labels | 35,680；24,964/5,350/5,366 | 较强：标签一致，模型数略少，test geometry 未预训练 |
| MFCAD++ | 59,665、24 类 + Stock；41,766/8,950/8,949 | 同官方划分 | 较强：监督划分一致，test geometry 未预训练 |
| TMCAD | 原始 10 类、约 10,897；清洗差异明显 | 10,886；8,709/1,090/1,087 | 有限：须核对有效样本和划分 |
| [CADSynth](https://doi.org/10.57760/sciencedb.17011) | 100,000、24 类 + Stock；80/10/10 | OCC-clean 79,941/9,994/9,993 | 有限：沿用官方 train/test，但过滤 72 个异常模型 |
| [MFInstSeg](https://doi.org/10.1016/j.rcim.2023.102661) | 62,495、24 类 + Stock；常见 70/15/15；三种任务标签 | 49,996/6,249/6,250，seed 42；仅语义标签 | 有限：任务一致但划分不同 |
| FabWave | 45 类约 4.5k，另有多种派生版 | 指定实验排除 | 不比较本地结果 |

证据等级：`A` 为原论文可核验且协议基本一致；`A−` 为同一数据集但有小幅版本差异；`B` 为自定义划分、预训练或后续论文复现表；`C` 为派生标签/少样本/显著不同任务。排名只是已报告数值位置。

## 3. Fusion360Seg 标准 8 类

| 排名 | 方法 | 年份 | Acc | mIoU | 等级 |
|---:|---|---:|---:|---:|:---:|
| 1 | [TopoGNN](https://doi.org/10.2139/ssrn.6604901) | 2026 | 97.21 | 87.57 | B |
| 2 | [Masked HGT](https://arxiv.org/abs/2603.14927) | 2026 | 97.02 | 86.75 | B |
| 3 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | 96.88 | 83.77 | B |
| 4 | [BRep-BERT*](https://doi.org/10.1145/3583780.3615237) | 2023 | 96.02 | 85.54 | B |
| 5 | [SSL4CAD](https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html) | 2023 | 96.00 | — | B |
| **6** | **BRepPreDiff inductive9 + MLP** | 2026 | **95.9258** | **82.7926** | A− |
| 7 | [CADOps-Net](https://arxiv.org/abs/2208.10555) | 2022 | 95.90 | 84.20 | A |
| 8 | [Two-level feature reconstruction](https://www.sciencedirect.com/science/article/pii/S095219762601050X) | 2026 | 94.51 | 78.13 | B |
| 9 | [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | 94.48 | 79.23 | A |
| 10 | [BRepNet](https://openaccess.thecvf.com/content/CVPR2021/html/Lambourne_BRepNet_A_Topological_Message_Passing_System_for_Solid_Models_CVPR_2021_paper.html) | 2021 | 92.52 | 77.10 | A |
| 11 | [UV-Net](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | 2021 | 92.30 | 72.40 | A− |
| 12 | [FoV-Net](https://arxiv.org/abs/2602.24084) | 2026 | 91.72 | 73.81 | A |
| 13 | [Uncertainty review](https://doi.org/10.1007/978-3-031-96196-0_4) | 2025 | ≈84.00 | — | C |
| 14 | [Modified PointNet++](https://doi.org/10.1016/j.cad.2023.103629) | 2024 | 80.46 | — | A |
| 15 | [BRepGAT](https://doi.org/10.1093/jcde/qwad100) | 2023 | 80.32 | — | A |

两个头 Acc 相同；MLP 的 Macro-F1 和 mIoU 分别高 1.0119/0.9771 pp。其 Acc 比 CADOps-Net 高 0.0258 pp、比 SSL4CAD 低 0.0742 pp。

## 4. MFCAD++

`Δ = 论文 Acc − 99.4122`。

| 排名 | 方法 | 年份 | Acc | mIoU | Δ | 等级 |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | [BrepMFR](https://doi.org/10.1016/j.cagd.2024.102318) | 2024 | 99.76 | 99.30 | +0.35 | A |
| 2 | [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | 2026 | 99.66 | 99.32 | +0.25 | B |
| 3 | [EMD-GNN](https://doi.org/10.1016/j.aei.2026.104609) | 2026 | 99.62 | 98.79 | +0.21 | A |
| 4 | [Masked HGT](https://arxiv.org/abs/2603.14927) | 2026 | 99.61 | 99.03 | +0.20 | B |
| 5 | [BRepMAE / MFCAD2](https://arxiv.org/abs/2602.22701) | 2026 | 99.59 | 98.82 | +0.18 | B |
| 6 | [SCUT B-Rep GNN](https://zrb.bjb.scut.edu.cn/EN/10.12141/j.issn.1000-565X.230497) | 2025 | 99.53 | 99.15 | +0.12 | A |
| **7** | **BRepPreDiff inductive9 + MLP** | 2026 | **99.4122** | **98.1722** | 基准 | A |
| 8 | [MMNet](https://doi.org/10.32604/cmes.2026.078073) | 2026 | 99.38 | 98.86 | −0.03 | A |
| 9 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | 99.35 | 98.02 | −0.06 | B |
| 10 | [FoV-Net](https://arxiv.org/abs/2602.24084) | 2026 | 99.33 | 97.81 | −0.08 | A |
| 11 | [TEGNet](https://www.researchgate.net/publication/401656636) | 2026 | 99.31 | 98.71 | −0.10 | A |
| 12 | [MFTReNet](https://doi.org/10.1016/j.aei.2024.102721) | 2024 | 99.30 | 98.63 | −0.11 | A |
| 13 | [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | 99.27 | 97.98 | −0.14 | A |
| 14 | [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | 2024 | 99.26 | 98.66 | −0.15 | A− |
| 15 | [MSFRNet](https://doi.org/10.1016/j.aei.2026.104365) | 2026 | 99.19 | — | −0.22 | B |
| 16 | [BRepGAT / MFCAD18++](https://doi.org/10.1093/jcde/qwad100) | 2023 | 99.10 | — | −0.31 | C |
| 17 | [Sheet-metalNet](https://doi.org/10.1038/s41598-024-61443-2) | 2024 | 98.86 | — | −0.55 | A− |
| 18 | [Hierarchical CADNet](https://doi.org/10.1016/j.cad.2022.103226) | 2022 | 97.37 | — | −2.04 | A |
| 19 | [Uncertainty review](https://doi.org/10.1007/978-3-031-96196-0_4) | 2025 | ≈96.20 | — | −3.21 | C |
| 20 | [Modified PointNet++](https://doi.org/10.1016/j.cad.2023.103629) | 2024 | 95.85 | — | −3.56 | A |
| 21 | [Shaft process-planning model](https://doi.org/10.3390/app16020828) | 2026 | 90.87 | — | −8.54 | C |

MLP 相对 DiffLoss 高 0.0338 pp Acc、0.0573 pp Macro-F1 和 0.1105 pp mIoU。官方监督划分和 test-isolated 预训练使本版比旧 transductive 结果更适合论文比较。

## 5. TMCAD / MechCAD

| 排名 | 方法 | 年份 | Acc | Δ | 等级 | 协议 |
|---:|---|---:|---:|---:|:---:|---|
| 1 | [KDH-CAD](https://arxiv.org/abs/2606.01702) | 2026 | 95.82 | +10.91 | C | 9,799、重标注、少样本 |
| 2 | [TopoGNN](https://doi.org/10.2139/ssrn.6604901) | 2026 | 88.50 | +3.59 | B | 完整划分未披露 |
| **3** | **BRepPreDiff inductive9 + Mean+Max + DiffLoss** | 2026 | **84.9126** | 基准 | A− | 10,886；8,709/1,090/1,087 |
| 4 | [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | 84.72 | −0.19 | B | 仅 7,599 个有效文件 |
| 5 | [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | 83.45 | −1.46 | A− | 原始 10 类；70/15/15 |

DiffLoss 相对 MLP 提高 1.1040 pp Acc、1.0969 pp Macro-F1 和 1.5400 pp mIoU。与 BRT 的 +1.46 pp 只作定位，不是严格同协议收益。

## 6. CADSynth

CADSynth 由 BrepMFR 工作提出，含 100,000 个合成 B-rep、24 种加工特征及 Stock，原论文为 80/10/10。下表统一采用逐面**总体 Accuracy**。

| 排名 | 方法 | 年份 | Acc | mIoU | Δ | 等级 |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | [BrepMFR](https://doi.org/10.1016/j.cagd.2024.102318) | 2024 | **99.96** | 99.83 | +0.36 | A |
| 2 | [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | 2026 | 99.91±0.01 | 99.83±0.01 | +0.31 | B |
| 3 | AAGNet（[BrepMFR Table 2](https://doi.org/10.1016/j.cagd.2024.102318) 复现） | 2024 | 99.80 | 99.25 | +0.20 | B |
| 4 | UV-Net（BrepMFR Table 2 复现） | 2024 | 99.74 | 99.02 | +0.14 | B |
| 5 | BRepNet（BrepMFR Table 2 复现） | 2024 | 99.67 | 98.77 | +0.07 | B |
| **6** | **BRepPreDiff inductive9 + DiffLoss** | 2026 | **99.5973** | **98.6904** | 基准 | A− |
| 7 | Hierarchical CADNet（BrepMFR Table 2 复现） | 2024 | 99.53 | 97.96 | −0.07 | B |

**表头错位核验：** 2026 Topo-Geom DualGNN 的对比表把 BrepMFR 写成 99.92% Acc；BrepMFR 原论文 Table 2 显示 99.96% 才是总体 Acc，99.92% 是 per-class accuracy，本表采用原始来源。Topo-Geom 的 99.91±0.01 来自清洗后 5-fold validation；本地是官方 test 的 OCC-clean 子集，故不作严格排名。DiffLoss 相对 MLP 提高 0.0243/0.0463/0.0912 pp（Acc/Macro-F1/mIoU）。

[SFRGNN-DA](https://doi.org/10.1016/j.jmsy.2025.05.005) 和 [BrepHGNet](https://www.sciencedirect.com/science/article/pii/S073658452600030X) 明确使用 CADSynth，但当前可访问摘要未披露可核验的数据集特定 Accuracy，因此记为 `NR`，不进入排序。

## 7. MFInstSeg

MFInstSeg 由 AAGNet 发布，通常称含 62,495 个 B-rep、24 种加工特征及 Stock，并提供语义分割、实例分组、底面识别三套标签。本地只评估逐面语义分割。

### 7.1 语义分割 Accuracy

| 排名 | 方法 | 年份 | Acc | mIoU | Δ | 等级 |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | [BRepFormer](https://arxiv.org/abs/2504.07378) | 2025 | 99.62±0.03 | 98.74±0.09 | +0.34 | A− |
| 2 | [EMD-GNN](https://doi.org/10.1016/j.aei.2026.104609) | 2026 | 99.58±0.01 | 98.60±0.02 | +0.30 | A |
| 3 | [MFTReNet](https://doi.org/10.1016/j.aei.2024.102721) | 2024 | 99.56±0.02 | 98.43±0.03 | +0.28 | A |
| 4 | [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | 2026 | 99.54±0.04 | 99.14±0.06 | +0.26 | B |
| 5 | [Masked HGT](https://arxiv.org/abs/2603.14927) | 2026 | 99.53 | NR | +0.25 | B |
| **6** | **BRepPreDiff inductive9 + MLP** | 2026 | **99.2767** | **97.5963** | 基准 | B |
| 7 | [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | 2024 | 99.15±0.03 | 98.45±0.04 | −0.13 | A |
| 8 | DeeperGCN（AAGNet 主表） | 2024 | 99.03±0.02 | 98.31±0.01 | −0.25 | B |
| 9 | [MSFRNet](https://doi.org/10.1016/j.aei.2026.104365) | 2026 | 98.95 | 98.23 | −0.33 | B |

本地 MLP 相对 DiffLoss 高 0.0375 pp Acc、0.0391 pp Macro-F1 和 0.0728 pp mIoU。本地 Acc 高于 AAGNet，但 mIoU 低 0.8537 pp，且 80/10/10 与论文约 70/15/15 不同，不能宣称整体优于 AAGNet。

### 7.2 不与语义 Accuracy 混排

| 方法 | 任务/协议 | 主要结果 | 原因 |
|---|---|---|---|
| [AAGNet](https://doi.org/10.1016/j.rcim.2023.102661) | 实例 / 底面 | Instance Acc 99.94、F1 98.84；Bottom Acc 99.75、mIoU 98.47 | 不同任务 |
| [EMD-GNN](https://doi.org/10.1016/j.aei.2026.104609) | 实例 / 底面 | Instance Acc 99.90、F1 98.55；Bottom Acc 99.83、mIoU 99.37 | 不同任务 |
| [Topo-Geom DualGNN](https://doi.org/10.3390/machines14040362) | 实例 / 底面 | Instance Acc 99.90、F1 99.54；Bottom Acc 99.92、mIoU 99.51 | 不同任务且为 5-fold validation |
| [EAGIS](https://assets-eu.researchsquare.com/files/rs-4908235/v1_covered_753b1394-2eae-4437-98ae-dbfc4da2214f.pdf) | 仅实例分组 | Instance Acc 99.57、F1 98.04 | 不预测语义；预印本 |
| [BRepMAE](https://arxiv.org/abs/2602.22701) | 0.1% 标注语义 | Acc 81.49、mIoU 63.27 | 极低标签协议 |
| [Masked HGT](https://arxiv.org/abs/2603.14927) | 0.1% 标注语义 | Acc 88.75、mIoU 66.38 | 极低标签协议 |

[FeatureFox](https://arxiv.org/abs/2604.26770) 在 MFInstSeg 上采用联合衡量实例分离与语义正确性的 Panoptic Quality；[AAGATNet](https://doi.org/10.1016/j.cad.2026.104041) 的新增主基准是 MFInstSeg++。二者不与上面的原始 MFInstSeg 逐面语义 Accuracy 主表混排。

## 8. FabWave（指定实验无本地结果）

| 方法 | 年份 | 版本 | Acc / 主要结果 |
|---|---:|---|---:|
| [Brep2Shape](https://arxiv.org/abs/2602.07429) | 2026 | 4,572、45 类 | 99.99%；三种子 99.74±0.43 |
| [BRT](https://doi.org/10.1016/j.cad.2025.103940) | 2025 | 45 类 | 98.95% |
| [VGNet](https://doi.org/10.1109/TMM.2024.3521706) | 2025 | 4,475、45 类 | 98.00% |
| AAGNet（Brep2Shape 复现） | 2026 | 45 类 | 96.33% |
| UV-Net（Brep2Shape 复现） | 2026 | 45 类 | 92.68% |
| [UV-Net 原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.html) | 2021 | 52 类 | 94.51±0.10% |
| [CADGCL](https://doi.org/10.1007/s00371-025-03949-y) | 2025 | 4,475、45 类 | F1 98.84%；未报 Acc |

指定九来源实验排除 FabWave，本节仅保留文献背景，旧版 40 类 transductive BRepPreDiff 数字已移除。

## 9. 建议表述

> Using a four-layer Edge Update Attention encoder pretrained for 100 epochs on an inductive nine-source corpus, BRepPreDiff achieved 95.93% accuracy and 82.79% mIoU on Fusion360Seg, 99.41% and 98.17% on the official MFCAD++ split, and 84.91% and 73.75% on TMCAD. All validation and test geometries were held out from self-supervised pretraining.

> On the OCC-cleaned official CADSynth test subset, BRepPreDiff achieved 99.60% face accuracy, 99.34% macro-F1, and 98.69% mIoU with DiffLoss. On a deterministic seed-42 80/10/10 MFInstSeg split, the MLP head achieved 99.28% semantic face accuracy, 98.77% macro-F1, and 97.60% mIoU. MFInstSeg literature commonly uses a different split and also evaluates instance and bottom-face tasks, so only numerical references are claimed.

## 10. 检索边界与本地证据

- 优先使用原论文、正式出版页、arXiv 原稿和作者公开 PDF；无法核验的摘要数字不进入主表。
- CADSynth 的历史方法采用 BrepMFR 原论文 Table 2；Topo-Geom 采用其清洗后 5-fold validation 结果。
- MFInstSeg 的语义、实例、底面任务严格分表；同名 F1/Accuracy 只有任务定义一致时才比较。
- BRT 统一采用期刊最终版/v2。不同论文的 mIoU 平均方式仍可能不同。
- 本地 run、checkpoint 哈希详见 `reports/inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901.md`。
- 九来源为 BRepPreDiff、TMCAD、Fusion360Seg、MFCAD++、Fusion360Rec、Fusion360Ass、SolidLetters、CADSynth、MFInstSeg；均只取 train split，排除 FabWave。配置见 `data/pretrain_joint_inductive.yaml`。
