# Inductive 9-source encoder：全量下游 MLP 与 DiffLoss 对照

所有实验使用同一 100-epoch full-loss inductive 9-source（无 FabWave）encoder；LR=1.0e-3、scheduler=constant、min_lr=0.0、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1、最多 3 路并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 137 | 1787 | 98.8211 | 97.1017 | 98.8190 | 94.4246 |
| BRepPreDiff | DiffLoss | 116 | 1787 | 98.8522 | 97.0389 | 98.8476 | 94.3063 |
| Fusion360Seg | MLP | 191 | 5366 | 95.8609 | 89.9946 | 95.8331 | 83.3339 |
| Fusion360Seg | DiffLoss | 198 | 5366 | 95.8012 | 88.6997 | 95.7705 | 81.8114 |
| MFCAD++ | MLP | 185 | 8949 | 99.4003 | 99.0602 | 99.4001 | 98.1502 |
| MFCAD++ | DiffLoss | 191 | 8949 | 99.3866 | 99.0465 | 99.3862 | 98.1240 |
| TMCAD | MLP | 152 | 1087 | 83.9006 | 83.5167 | 83.8505 | 72.3958 |
| TMCAD | DiffLoss | 71 | 1087 | 83.7167 | 83.1429 | 83.5570 | 72.0176 |
| SolidLetters | MLP | 150 | 19392 | 97.6176 | 97.6856 | 97.6136 | 95.7037 |
| SolidLetters | DiffLoss | 166 | 19392 | 97.5866 | 97.6490 | 97.5784 | 95.6409 |
| CADSynth | MLP | 71 | 9993 | 99.5870 | 99.3219 | 99.5866 | 98.6577 |
| CADSynth | DiffLoss | 144 | 9993 | 99.6063 | 99.3618 | 99.6058 | 98.7355 |
| MFInstSeg | MLP | 166 | 6250 | 99.2580 | 98.7840 | 99.2574 | 97.6236 |
| MFInstSeg | DiffLoss | 200 | 6250 | 99.2345 | 98.7404 | 99.2344 | 97.5414 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | +0.0311 | -0.0629 | +0.0286 | -0.1184 | DiffLoss |
| Fusion360Seg | -0.0597 | -1.2949 | -0.0626 | -1.5225 | MLP |
| MFCAD++ | -0.0138 | -0.0137 | -0.0139 | -0.0261 | MLP |
| TMCAD | -0.1840 | -0.3738 | -0.2935 | -0.3781 | MLP |
| SolidLetters | -0.0309 | -0.0366 | -0.0352 | -0.0628 | MLP |
| CADSynth | +0.0193 | +0.0399 | +0.0192 | +0.0777 | DiffLoss |
| MFInstSeg | -0.0235 | -0.0436 | -0.0230 | -0.0822 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-173933_full_brepprediff_seg_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `135f5e7ceaa96c9a12ebc993d557dd69aa2539a33d22aebbfe6437d7bd86f321`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-173933_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `13a0235eb7a9731b57cb59915422ba5231ca75802b475e800e5fd1766d334cd9`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-173933_full_fusion360seg_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `43d6ea338d8971b509eda0c0faa044f3cce1d56b20b65a17ead02ed044d24384`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_fusion360seg_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `190b33c34351958895865a8dc8e3f2abbc145b99a112ef918354a2526c2a56df`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `fe85158883c5816c34ab9a51d6e9f191069c0f0eca2147735041a129d43e7fe2`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `607a92c92992ef0e73850071e3d9f5771a35b7b73755c7ae92c1a6b2771dab4f`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-194207_full_tmcad_cls_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `2f61fb4234b0db8ed512a8b4dc9bb5a653a8104991841ad83cebd6513c326e9d`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-195204_full_tmcad_cls_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `f5f1050d671d40a4ac252bc9f419792a5fbebbd0f92290424921c9e463bb7bcf`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-191501_full_solidletters_cls_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `6bfc029add63e8e5b6bd9df93d2f14023f0e5fa1f25bf5c800a6dd6177024ac1`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-205024_full_solidletters_cls_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `a352df3ae42f648d7dc8e2da39229797f61e3298bd6dd6632dbeab19d2a86f0b`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_cadsynth_seg_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `1a690a97bb7ac2aec1132ae334e1d0980c040359eef0cd09d831b52820ef95d5`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `51b32c38d351b572e31554ee35378ab701b0da68d896890715a9ee9f741a4b32`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `947107ff3b936bccf5a2406367883ae9961af0e6ee30d62c7a706e92c772397c`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_lr1e3_constant_pre100_ft200/finetune/20260904-182816_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904`; SHA-256 `f80f90d867a9a07d6eebc1a220c12f17bed79bc84a596fea99dcf1b195b20912`
