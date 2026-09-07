# Inductive 9-source encoder：全量下游 MLP 与 DiffLoss 对照

所有实验使用同一 100-epoch inductive 9-source lr=1e-4（无 FabWave）encoder、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 154 | 1787 | 98.7939 | 96.9602 | 98.7902 | 94.1623 |
| BRepPreDiff | DiffLoss | 109 | 1787 | 98.8027 | 96.8796 | 98.7978 | 94.0120 |
| Fusion360Seg | MLP | 143 | 5366 | 95.9258 | 89.5940 | 95.8991 | 82.7926 |
| Fusion360Seg | DiffLoss | 194 | 5366 | 95.9258 | 88.5822 | 95.8994 | 81.8155 |
| MFCAD++ | MLP | 185 | 8949 | 99.4122 | 99.0716 | 99.4120 | 98.1722 |
| MFCAD++ | DiffLoss | 197 | 8949 | 99.3784 | 99.0143 | 99.3781 | 98.0617 |
| TMCAD | MLP | 103 | 1087 | 83.8086 | 83.2922 | 83.6475 | 72.2066 |
| TMCAD | DiffLoss | 136 | 1087 | 84.9126 | 84.3891 | 84.7557 | 73.7466 |
| SolidLetters | MLP | 170 | 19392 | 97.4629 | 97.5259 | 97.4565 | 95.3956 |
| SolidLetters | DiffLoss | 139 | 19392 | 97.3907 | 97.4568 | 97.3840 | 95.2732 |
| CADSynth | MLP | 62 | 9993 | 99.5730 | 99.2922 | 99.5726 | 98.5993 |
| CADSynth | DiffLoss | 78 | 9993 | 99.5973 | 99.3385 | 99.5968 | 98.6904 |
| MFInstSeg | MLP | 194 | 6250 | 99.2767 | 98.7694 | 99.2758 | 97.5963 |
| MFInstSeg | DiffLoss | 186 | 6250 | 99.2392 | 98.7303 | 99.2384 | 97.5235 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | +0.0087 | -0.0806 | +0.0076 | -0.1503 | DiffLoss |
| Fusion360Seg | +0.0000 | -1.0119 | +0.0004 | -0.9771 | Tie |
| MFCAD++ | -0.0338 | -0.0573 | -0.0338 | -0.1105 | MLP |
| TMCAD | +1.1040 | +1.0969 | +1.1082 | +1.5400 | DiffLoss |
| SolidLetters | -0.0722 | -0.0690 | -0.0725 | -0.1224 | MLP |
| CADSynth | +0.0243 | +0.0463 | +0.0242 | +0.0912 | DiffLoss |
| MFInstSeg | -0.0375 | -0.0391 | -0.0373 | -0.0728 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-200155_full_brepprediff_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `dcfb2591b465e2f4b028078ea5a0b3b095c3c4e16c69ccc717e3bfd9479315e9`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-202953_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `ed2528e9e66f6ee5e41d69c4bcb9a41d7a0e3543c40a1ef11947977bf02b1b01`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-211317_full_fusion360seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `f1ccef9462d39bc5730b3d7807259cc94c0c01e705dbe4b5c94140c563a2d504`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-214421_full_fusion360seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `92b81f42119434728e7f12b44ee463affe51f0f19d258c1cf1c7aeeb46cbbb6f`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-221937_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `354333861a4ef16f82fba75682300575773e17043996e456fd03b9c186ec0bb4`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-232240_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `bb8aa29156b70ef03be07cbe74d1bcc505c45c6ba4774f5b8829833cba72841c`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-003838_full_tmcad_cls_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `0488b77fbe9886fb2c46b92c095f51be856bd5ae506ca0edd73b32c369bf6131`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-013844_full_tmcad_cls_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `c3cfe696aeb76028eae3d558585356423582835d0555dc460401a2592d2020c4`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-024034_full_solidletters_cls_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `b27f5453e8fc24e0893568312fbcfef347e208fa5aacd800b07ab92bd0fb2479`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-042701_full_solidletters_cls_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `96d9aff29c98ce7adfc9c9f6b3e3b68d6bc25a7ecfea22e4c7006d3ad95e2e67`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-080006_full_cadsynth_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `fb5b758d516dd08734d2799e68af1c1e08d2c47e81014c9a51700833c38778ac`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-092556_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `db3c5ba0c5161902db2b181569096189e50b19420b98a7b2f4c1a8ca51478225`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-103203_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `be383a1556f275561449778feef6318dfa02b3462beb5822e4b0bdca4d4128d4`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-112521_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901`; SHA-256 `bb126bc9f1bab996be5dc3dedc82dbedca46a5c2064970aa2b74f54a9a66e8ca`
