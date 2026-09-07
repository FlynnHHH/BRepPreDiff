# Inductive 9-source encoder：全量下游 MLP 与 DiffLoss 对照

所有实验使用同一 100-epoch full-loss inductive 9-source（无 FabWave）encoder；LR scheduler=cosine、min_lr=0.0、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1、最多 3 路并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 185 | 1787 | 98.7648 | 96.8588 | 98.7627 | 93.9737 |
| BRepPreDiff | DiffLoss | 160 | 1787 | 98.8056 | 96.8879 | 98.8013 | 94.0273 |
| Fusion360Seg | MLP | 153 | 5366 | 95.9167 | 88.9856 | 95.8881 | 82.2166 |
| Fusion360Seg | DiffLoss | 198 | 5366 | 95.9401 | 88.9535 | 95.9088 | 82.1120 |
| MFCAD++ | MLP | 188 | 8949 | 99.4037 | 99.0574 | 99.4037 | 98.1466 |
| MFCAD++ | DiffLoss | 189 | 8949 | 99.3858 | 99.0148 | 99.3858 | 98.0633 |
| TMCAD | MLP | 84 | 1087 | 82.7967 | 82.3395 | 82.7322 | 70.8260 |
| TMCAD | DiffLoss | 142 | 1087 | 83.2567 | 82.6908 | 83.0637 | 71.4387 |
| SolidLetters | MLP | 126 | 19392 | 97.4526 | 97.5096 | 97.4377 | 95.3735 |
| SolidLetters | DiffLoss | 194 | 19392 | 97.6330 | 97.6914 | 97.6300 | 95.6671 |
| CADSynth | MLP | 49 | 9993 | 99.5888 | 99.3179 | 99.5883 | 98.6499 |
| CADSynth | DiffLoss | 82 | 9993 | 99.5898 | 99.3172 | 99.5892 | 98.6484 |
| MFInstSeg | MLP | 187 | 6250 | 99.2867 | 98.8143 | 99.2865 | 97.6822 |
| MFInstSeg | DiffLoss | 195 | 6250 | 99.2691 | 98.8241 | 99.2687 | 97.6992 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | +0.0408 | +0.0291 | +0.0386 | +0.0537 | DiffLoss |
| Fusion360Seg | +0.0234 | -0.0322 | +0.0208 | -0.1046 | DiffLoss |
| MFCAD++ | -0.0178 | -0.0427 | -0.0179 | -0.0833 | MLP |
| TMCAD | +0.4600 | +0.3512 | +0.3316 | +0.6127 | DiffLoss |
| SolidLetters | +0.1805 | +0.1818 | +0.1924 | +0.2937 | DiffLoss |
| CADSynth | +0.0011 | -0.0007 | +0.0009 | -0.0015 | DiffLoss |
| MFInstSeg | -0.0176 | +0.0098 | -0.0178 | +0.0170 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-210624_full_brepprediff_seg_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `2e2b49eaca043c3d54b5615d683056e8a8bd6fb9815abc469c1a56b20e67fcfa`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-210624_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `fc504ffbce02f9af05ba83abf61f0fec7f4735ae1aa8ca721ecd1da58abf4385`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-210624_full_fusion360seg_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `20ed83976fd5c0bcff2b9d39d3c293738fa7ca4deb119833b3920c9dc8400bee`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-215002_full_fusion360seg_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `e8596736951818b32430f2f6f8d88c8f637156f4037d387ac9cbcb75f0484b77`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-215002_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `0949faccb811e3b7cbcaf7de262f2df14f4e1a6b736fa18bda71c9f340bf9b6a`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-215002_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `da73e6c3e051fe327ba91a0bec5e376699f144efad31165bd5c2728950a729bb`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-225946_full_tmcad_cls_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `9e01e753f71b49d3d7c8711159f8efee365a4390ba4dc3bf5770d3d576ee7337`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-225946_full_tmcad_cls_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `a7ec16c34e80fc3ff19a58f2515560c51199399a9d64aabaf93657ad21b6fa62`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260903-225946_full_solidletters_cls_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `1eaa199ab29c253e2ee8d0c500fd0508503bbcdb9ccb23873090f2d7b58dec82`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260904-004157_full_solidletters_cls_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `00edb7fa9570eee27a5e99bf6362149e56faae5405190893016da14e8eff5ec5`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260904-004157_full_cadsynth_seg_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `d313117458a954de975a159e608cd15a15d02465b7988e93b17b2d5b12077399`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260904-004157_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `ef130abd4f3bac0ff3254a98a7d1b776d2497f95b0265c823670df15c337ec92`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260904-025054_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `b0ebf926a2edb99f2982b08bfc94a6b1f60236eb7e15479b1c86cc1813e9b97f`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive_full_cosine_batched3/finetune/20260904-025054_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_full_cosine_pre100_ft200_batched3_seed42_20260903`; SHA-256 `09395b3e9b7f363e7004ad0c712c211042eb12e5f1502f73df81bc52c056b443`
