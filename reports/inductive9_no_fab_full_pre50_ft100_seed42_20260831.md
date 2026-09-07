# Inductive 9-source encoder：全量下游 MLP 与 DiffLoss 对照

所有实验使用同一 50-epoch inductive 9-source（无 FabWave）encoder、相同数据划分、
随机种子 42、100 epochs、task batch size 256/512、梯度累积 1，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 76 | 1787 | 98.8590 | 97.1953 | 98.8537 | 94.5988 |
| BRepPreDiff | DiffLoss | 92 | 1787 | 98.8532 | 97.1639 | 98.8482 | 94.5395 |
| Fusion360Seg | MLP | 94 | 5366 | 95.6364 | 88.1436 | 95.5946 | 81.1643 |
| Fusion360Seg | DiffLoss | 91 | 5366 | 95.7104 | 88.9072 | 95.6894 | 81.9816 |
| MFCAD++ | MLP | 88 | 8949 | 99.3996 | 99.0666 | 99.3996 | 98.1630 |
| MFCAD++ | DiffLoss | 96 | 8949 | 99.3791 | 99.0154 | 99.3793 | 98.0635 |
| TMCAD | MLP | 23 | 1087 | 82.2447 | 81.4891 | 81.8035 | 69.6056 |
| TMCAD | DiffLoss | 57 | 1087 | 83.2567 | 82.8090 | 83.2259 | 71.7709 |
| SolidLetters | MLP | 99 | 19392 | 97.4887 | 97.5464 | 97.4750 | 95.4277 |
| SolidLetters | DiffLoss | 87 | 19392 | 97.2927 | 97.3810 | 97.3064 | 95.1357 |
| CADSynth | MLP | 68 | 9993 | 99.5859 | 99.3069 | 99.5855 | 98.6285 |
| CADSynth | DiffLoss | 96 | 9993 | 99.5766 | 99.2733 | 99.5760 | 98.5641 |
| MFInstSeg | MLP | 97 | 6250 | 99.2316 | 98.7268 | 99.2308 | 97.5136 |
| MFInstSeg | DiffLoss | 99 | 6250 | 99.1993 | 98.6838 | 99.1989 | 97.4308 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.0058 | -0.0314 | -0.0055 | -0.0592 | MLP |
| Fusion360Seg | +0.0740 | +0.7636 | +0.0948 | +0.8173 | DiffLoss |
| MFCAD++ | -0.0204 | -0.0512 | -0.0203 | -0.0994 | MLP |
| TMCAD | +1.0120 | +1.3199 | +1.4225 | +2.1653 | DiffLoss |
| SolidLetters | -0.1960 | -0.1654 | -0.1686 | -0.2920 | MLP |
| CADSynth | -0.0093 | -0.0336 | -0.0095 | -0.0644 | MLP |
| MFInstSeg | -0.0323 | -0.0429 | -0.0319 | -0.0828 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-010305_full_brepprediff_seg_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `7d0ba068035b1c2623b21395a29138b1c8e9a8cb352aa238ecde364ec19fe073`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-011534_full_brepprediff_seg_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `2917e0842d580ff317ed9c811445fb138a55bfc5373ed51a8e185fedb4a5f063`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-012950_full_fusion360seg_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `acac5227c32bb1dd4a8609aff01a7dc2517a1b5030e6a3127a0bac7d70e99180`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-014022_full_fusion360seg_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `a74554d64f17c4bdc11d71ee3e5e3e27c66c5d583d267fbb88ae1ccab636e3da`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-015154_full_mfcadpp_seg_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `bf20a8f651be613ccc8a675106e3451a72726ac6a66e3191c35eb14fb74218a6`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-021219_full_mfcadpp_seg_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `bcd57b5c4ea0ed3a1198385bee79d9b510e72265e83a0336323bef1074ee4fcb`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-023517_full_tmcad_cls_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `abe4b313d85bf74814a96d960b7fc46874cfd6a8550158913ce10c68b1f84719`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-025621_full_tmcad_cls_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `73e1775cb30926ebcbb99758dcab028579769cc8998938a8da5feae92e894baf`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-031801_full_solidletters_cls_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `50eeacdb358ad77ce2bad793bd99d01f387e3a5b77d1c06e9d0b573c7913b34f`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-035209_full_solidletters_cls_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `2676b476547aa65f57c38db51769db6f786a18e20341f632373e6e7616dcde7b`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-042914_full_cadsynth_seg_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `45cc785f972695d035f0720bd5c31b15868d25f1169624b64734a45b9350dc46`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-045735_full_cadsynth_seg_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `5b28ae75e5511a2f9125debe98849a07a78e9793b5a3061ba43e5d4179fe9b49`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-053014_full_mfinstseg_seg_mlp_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `f20483cc4b1e11b7e1b9fece4638515a2fd5aa00fd57b6acc54455cb5ace1b4e`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full/finetune/20260901-054844_full_mfinstseg_seg_diffloss_ft100_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831`; SHA-256 `8af747324ad2f7fd03cc1e6f23ee9c053fdec9d6d40cd271356115f210b6acf5`
