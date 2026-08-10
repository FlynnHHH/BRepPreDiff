# Edge Update Attention：全部 Benchmark 的 MLP 与 DiffLoss 对照

> 生成日期：2026-08-08  
> Encoder：Edge Update Attention（4 heads）  
> 预训练：七来源、全部 split 无标签联合预训练  
> 随机种子：42；按 validation accuracy 选择 `best.pt`；下表均为正式 test split

## 1. 实验协议

每个 benchmark 使用同一份预训练 checkpoint、相同数据划分、优化器设置和有效 batch size。分割任务的两种 head 均训练 100 epochs；分类任务的两种 head 均训练 200 epochs。DiffLoss 使用 bipolar one-hot、cosine 1000-step 训练噪声日程、每 token 4 个噪声样本以及 1-step DDIM 推理。

该预训练语料包含下游 validation/test 的无标签几何，因此属于 transductive self-supervised 协议，不能视作严格 inductive benchmark。所有配置只有单个随机种子。

## 2. 完整测试结果

| Benchmark | Task | Head | Best epoch | Samples | Accuracy (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Blendit | 3-class face segmentation | MLP | 18 | 766 | 98.1210 | 95.3927 | 97.1606 | 96.2606 | 98.1351 | 92.8949 |
| Blendit | 3-class face segmentation | DiffLoss | 36 | 766 | 98.1234 | 95.3413 | 97.2025 | 96.2541 | 98.1381 | 92.8842 |
| Fusion360Seg s2.0.0 | 8-class face segmentation | MLP | 97 | 5,366 | 93.0596 | 90.7196 | 85.0325 | 87.3927 | 93.0200 | 78.6993 |
| Fusion360Seg s2.0.0 | 8-class face segmentation | DiffLoss | 58 | 5,366 | 92.8182 | 89.3465 | 83.5582 | 85.9072 | 92.7495 | 76.8181 |
| MFCAD++ | 25-class face segmentation | MLP | 88 | 8,949 | 99.3632 | 99.0484 | 98.9362 | 98.9913 | 99.3627 | 98.0172 |
| MFCAD++ | 25-class face segmentation | DiffLoss | 94 | 8,949 | 99.3825 | 99.0638 | 99.0127 | 99.0376 | 99.3826 | 98.1083 |
| TMCAD | 10-class graph classification | MLP | 192 | 1,087 | 84.9126 | 84.7475 | 84.7205 | 84.6192 | 84.9482 | 73.9883 |
| TMCAD | 10-class graph classification | DiffLoss | 173 | 1,087 | 83.9926 | 83.5004 | 83.7245 | 83.4995 | 83.9063 | 72.3311 |
| FabWave min10 | 40-class graph classification | MLP | 6 | 391 | 97.9540 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10 | 40-class graph classification | DiffLoss | 87 | 391 | 97.9540 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |

## 3. Head 差异

差值定义为 `DiffLoss - MLP`，单位为百分点；正值表示 DiffLoss 更高。

| Benchmark | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| Blendit | +0.0024 | -0.0065 | +0.0030 | -0.0106 | DiffLoss |
| Fusion360Seg s2.0.0 | -0.2413 | -1.4855 | -0.2705 | -1.8812 | MLP |
| MFCAD++ | +0.0193 | +0.0463 | +0.0199 | +0.0911 | DiffLoss |
| TMCAD | -0.9200 | -1.1197 | -1.0419 | -1.6571 | MLP |
| FabWave min10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Tie |

## 4. 结论

- **Blendit**：DiffLoss，Accuracy 差值 +0.0024 pp。
- **Fusion360Seg s2.0.0**：MLP，Accuracy 差值 -0.2413 pp。
- **MFCAD++**：DiffLoss，Accuracy 差值 +0.0193 pp。
- **TMCAD**：MLP，Accuracy 差值 -0.9200 pp。
- **FabWave min10**：Tie，Accuracy 差值 +0.0000 pp。

低于 1 pp 的单 seed 差异不应解释为稳定优势，建议对重点组合补做至少 3 个随机种子。

## 5. 运行产物

- Blendit / MLP：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_blendit_seg_20260807-123259`；`best.pt` SHA-256 `10f8bb47a845206701a5c44272c5bb0c29e90ed473cb051baff5d8c7dc474c9d`
- Blendit / DiffLoss：`runs/edge_update_new_joint/finetune/20260808-163205_edge_update_blendit_seg_diffloss_20260808-head-complements-titan`；`best.pt` SHA-256 `1f55bc738dd5628a1f2caa134141f49cb2931d7ae769bb36c49c184ebbddfc9b`
- Fusion360Seg s2.0.0 / MLP：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_fusion360seg_20260807-123259`；`best.pt` SHA-256 `ec85f26bc567b8ca15bc0d8b757c151cc9febe13ff5af0baed9b6b6955572237`
- Fusion360Seg s2.0.0 / DiffLoss：`runs/edge_update_new_joint/finetune/20260808-163205_edge_update_fusion360seg_diffloss_20260808-head-complements-titan`；`best.pt` SHA-256 `77c174f8378a1dd88ec062f84e370c6ca1fe0624244abc7ceff0118091feee13`
- MFCAD++ / MLP：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_mfcadpp_seg_20260807-123259`；`best.pt` SHA-256 `ce23b90d702bec5fe986d917e6866d3d15e8981284079ca9b4d051fd2f948efd`
- MFCAD++ / DiffLoss：`runs/edge_update_new_joint/finetune/20260808-163205_edge_update_mfcadpp_seg_diffloss_20260808-head-complements-titan`；`best.pt` SHA-256 `09a8bc2192ae7444e9565470945b8c35241725cce94b568e18fdd889567e0ea5`
- TMCAD / MLP：`runs/edge_update_new_joint/finetune/20260808-163205_edge_update_tmcad_cls_mlp_20260808-head-complements-titan`；`best.pt` SHA-256 `e11e379ecd925bc1881c652667dd52eccfa67eca47841ce9b850f36e07cc2871`
- TMCAD / DiffLoss：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_tmcad_cls_20260807-123259`；`best.pt` SHA-256 `0abb8b936a4dd76f79296980db1d8ed510b6fe94924d481b22ff20941bf7ccfa`
- FabWave min10 / MLP：`runs/edge_update_new_joint/finetune/20260808-174658_edge_update_fabwave_cls_mlp_20260808-head-complements-titan`；`best.pt` SHA-256 `d2776a349a44a78531211ebd9d75333d69b6dd9135e90c6f49db13c4760c08e8`
- FabWave min10 / DiffLoss：`runs/edge_update_new_joint/finetune/20260807-211612_edge_update_fabwave_cls_20260807-123259`；`best.pt` SHA-256 `95c439f3ca66a7e490d7cb6838ca4127e6c8626573c5407bda1612eca15fdff8`
