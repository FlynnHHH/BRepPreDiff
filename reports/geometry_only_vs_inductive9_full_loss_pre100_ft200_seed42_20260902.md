# Geometry-only 与 full-loss inductive 9-source encoder：200-epoch 下游对照

候选模型为 **Geometry-only encoder（categorical/relation loss weight = 0）**，基线为 **Full-loss encoder（lr=1e-4）**。两侧均使用相同的 9-source（无 FabWave）数据、
随机种子 42、下游划分、head 配置和 200-epoch 预算；分类任务使用 Mean+Max pooling。候选模型从 epoch 100 checkpoint
恢复并训练至 epoch 200，最终 checkpoint 按 epoch 1–200 的最高 validation accuracy 选择。

基线来源：`/home/nvme03/hhfeng/BRepPreDiff/reports/inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901.md`。差值定义为 `候选 - 基线`，单位为百分点。

| Task | Head | Candidate best | Baseline best | Candidate Acc. | Baseline Acc. | ΔAcc. | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 23 | 154 | 98.8978 | 98.7939 | +0.1039 | +0.0950 | +0.1049 | +0.1730 |
| BRepPreDiff | DiffLoss | 92 | 109 | 98.8104 | 98.8027 | +0.0077 | -0.0548 | +0.0091 | -0.1022 |
| Fusion360Seg | MLP | 196 | 143 | 95.9790 | 95.9258 | +0.0532 | -0.4560 | +0.0519 | -0.4447 |
| Fusion360Seg | DiffLoss | 157 | 194 | 95.9154 | 95.9258 | -0.0104 | +1.1486 | -0.0053 | +1.0887 |
| MFCAD++ | MLP | 185 | 185 | 99.3780 | 99.4122 | -0.0342 | -0.0449 | -0.0349 | -0.0873 |
| MFCAD++ | DiffLoss | 185 | 197 | 99.3955 | 99.3784 | +0.0171 | +0.0252 | +0.0176 | +0.0494 |
| TMCAD | MLP | 199 | 103 | 82.7047 | 83.8086 | -1.1039 | -1.3962 | -1.2885 | -1.7934 |
| TMCAD | DiffLoss | 89 | 136 | 83.0727 | 84.9126 | -1.8399 | -1.7322 | -1.7233 | -2.4166 |
| SolidLetters | MLP | 184 | 170 | 97.5815 | 97.4629 | +0.1186 | +0.1172 | +0.1182 | +0.2069 |
| SolidLetters | DiffLoss | 199 | 139 | 97.4680 | 97.3907 | +0.0773 | +0.0755 | +0.0777 | +0.1313 |
| CADSynth | MLP | 64 | 62 | 99.5805 | 99.5730 | +0.0075 | +0.0327 | +0.0075 | +0.0646 |
| CADSynth | DiffLoss | 85 | 78 | 99.5991 | 99.5973 | +0.0018 | +0.0080 | +0.0018 | +0.0154 |
| MFInstSeg | MLP | 199 | 194 | 99.2468 | 99.2767 | -0.0299 | -0.0072 | -0.0300 | -0.0102 |
| MFInstSeg | DiffLoss | 194 | 186 | 99.2275 | 99.2392 | -0.0117 | -0.0645 | -0.0118 | -0.1242 |

## 汇总

- Overall: Accuracy 胜/平/负 = 8/0/6；平均 ΔAccuracy -0.1888，ΔMacro-F1 -0.1610，ΔWeighted-F1 -0.1932，ΔmIoU -0.2321
- MLP: Accuracy 胜/平/负 = 4/0/3；平均 ΔAccuracy -0.1264，ΔMacro-F1 -0.2371，ΔWeighted-F1 -0.1530，ΔmIoU -0.2702
- DiffLoss: Accuracy 胜/平/负 = 4/0/3；平均 ΔAccuracy -0.2512，ΔMacro-F1 -0.0849，ΔWeighted-F1 -0.2335，ΔmIoU -0.1940

## Candidate run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_brepprediff_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_brepprediff_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_fusion360seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-140846_geometry_only_fusion360seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-141140_geometry_only_mfcadpp_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-140720_geometry_only_mfcadpp_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-143001_geometry_only_tmcad_cls_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-144929_geometry_only_tmcad_cls_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-145015_geometry_only_solidletters_cls_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-150400_geometry_only_solidletters_cls_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-152527_geometry_only_cadsynth_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-155609_geometry_only_cadsynth_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-161343_geometry_only_mfinstseg_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-161816_geometry_only_mfinstseg_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`
