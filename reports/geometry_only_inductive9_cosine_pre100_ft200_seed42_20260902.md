# Geometry-only inductive 9-source encoder：7 个下游 MLP 与 DiffLoss 对照

所有实验使用100-epoch inductive 9-source（无 FabWave）encoder；离散属性 loss 权重为 0；预训练 LR scheduler=cosine、min_lr=0.0、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1、GPU 4 并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 136 | 1787 | 98.8415 | 96.9633 | 98.8391 | 94.1649 |
| BRepPreDiff | DiffLoss | 16 | 1787 | 98.8338 | 96.8571 | 98.8315 | 93.9702 |
| Fusion360Seg | MLP | 196 | 5366 | 95.9764 | 90.2960 | 95.9483 | 83.6819 |
| Fusion360Seg | DiffLoss | 156 | 5366 | 95.9556 | 89.1821 | 95.9391 | 82.2927 |
| MFCAD++ | MLP | 189 | 8949 | 99.4011 | 99.0593 | 99.4004 | 98.1496 |
| MFCAD++ | DiffLoss | 188 | 8949 | 99.4193 | 99.0627 | 99.4192 | 98.1551 |
| TMCAD | MLP | 188 | 1087 | 82.7047 | 82.2682 | 82.6695 | 70.9888 |
| TMCAD | DiffLoss | 178 | 1087 | 84.8206 | 84.4206 | 84.7984 | 73.8701 |
| SolidLetters | MLP | 129 | 19392 | 97.4938 | 97.5575 | 97.4951 | 95.4171 |
| SolidLetters | DiffLoss | 174 | 19392 | 97.5351 | 97.6073 | 97.5385 | 95.5511 |
| CADSynth | MLP | 64 | 9993 | 99.5838 | 99.3204 | 99.5832 | 98.6551 |
| CADSynth | DiffLoss | 82 | 9993 | 99.5841 | 99.3123 | 99.5835 | 98.6406 |
| MFInstSeg | MLP | 168 | 6250 | 99.2445 | 98.7579 | 99.2447 | 97.5775 |
| MFInstSeg | DiffLoss | 191 | 6250 | 99.2550 | 98.7629 | 99.2536 | 97.5830 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.0078 | -0.1062 | -0.0076 | -0.1947 | MLP |
| Fusion360Seg | -0.0208 | -1.1139 | -0.0093 | -1.3891 | MLP |
| MFCAD++ | +0.0182 | +0.0034 | +0.0187 | +0.0055 | DiffLoss |
| TMCAD | +2.1159 | +2.1524 | +2.1289 | +2.8813 | DiffLoss |
| SolidLetters | +0.0413 | +0.0498 | +0.0434 | +0.1339 | DiffLoss |
| CADSynth | +0.0004 | -0.0081 | +0.0003 | -0.0145 | DiffLoss |
| MFInstSeg | +0.0106 | +0.0051 | +0.0089 | +0.0055 | DiffLoss |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_brepprediff_seg_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `32802704d003b817a7745e6b49d41165b28479e9b4cd8f1581470b4d093ad4c0`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_brepprediff_seg_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `aaada892c5dff938a836ad9d8300bc01bf2171d3f9522443837512445672a87a`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_fusion360seg_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `39a8f99e4bdc3afeebb7bf86c33652f05e5539d838d9ab1feca596c2185b9a9d`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_fusion360seg_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `08a7772b0938b33a89a74deade16d143ddc093b4b0b51b40de12348a5b0afd91`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-115022_geometry_only_mfcadpp_seg_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `4acd4871ec1b4a0ccef0e672199783ed546308eb8b75e09bd237b715ef02268c`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_mfcadpp_seg_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `d62b12a6aadccfd260f55e333f9aa075ec26e6f955d630167146d68456acbe3b`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_tmcad_cls_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `0d6adeb41f0c277b442b65a804c65a46234ac8c1d89fd151ed1cd313c6bb943f`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-115022_geometry_only_tmcad_cls_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `c2e6cf3520c3f4894847e62a91ad990142bf6fdfb501050cc1310f402b76d452`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_solidletters_cls_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `331550696651fa68a732ad9a4ecfa074e669c809b85e5f1d53012cd619178995`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-115022_geometry_only_solidletters_cls_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `08784aae7509ce8e15b123900dee055a778db8e3f383909bc1c01574cdb37a6b`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_cadsynth_seg_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `53e5ff27b8229e12b4efa58aa764fc7bd297aec095645301c0b6c85e62eb8e7a`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_cadsynth_seg_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `1acdb501528147be60c796aebe94527d08420f442e67527752124fd7c7c33ebd`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042548_geometry_only_mfinstseg_seg_mlp_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `4d151c38b55ddd438c0e116c5580c51118ee0bf013d6820a717afa4a619f32aa`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_cosine/finetune/20260903-042549_geometry_only_mfinstseg_seg_diffloss_ft200_seed42_geometry_only_inductive9_cosine_pre100_ft200_seed42_20260902`; SHA-256 `caa43f724bc0a64f3eee2867d93c7bb0efe41889e61f827edfc4fad088a48c43`
