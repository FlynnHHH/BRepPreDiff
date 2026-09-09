# Inductive9 rotate-mix encoder：全量下游 MLP 与 DiffLoss 对照

所有实验使用同一 100-epoch full-loss inductive 9-source（无 FabWave）encoder，LR=1e-4、constant scheduler、相同数据划分、
随机种子 42、200 epochs、per-process batch 128/256、梯度累积 2、预训练与微调均为 50% canonical + 50% SO(3) rotation、GPU 4 最多四路并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 200 | 1787 | 98.9901 | 97.3687 | 98.9887 | 94.9185 |
| BRepPreDiff | DiffLoss | 65 | 1787 | 98.9862 | 97.3943 | 98.9840 | 94.9674 |
| Fusion360Seg | MLP | 188 | 5366 | 96.9275 | 92.2254 | 96.9124 | 86.6362 |
| Fusion360Seg | DiffLoss | 190 | 5366 | 97.0183 | 90.9523 | 97.0102 | 85.2473 |
| MFCAD++ | MLP | 199 | 8949 | 99.6160 | 99.4271 | 99.6158 | 98.8667 |
| MFCAD++ | DiffLoss | 175 | 8949 | 99.5963 | 99.3842 | 99.5964 | 98.7830 |
| TMCAD | MLP | 50 | 1087 | 86.3845 | 85.8850 | 86.2738 | 75.9397 |
| TMCAD | DiffLoss | 110 | 1087 | 86.5685 | 86.2264 | 86.5812 | 76.4183 |
| SolidLetters | MLP | 165 | 19392 | 97.4629 | 97.5148 | 97.4466 | 95.3865 |
| SolidLetters | DiffLoss | 142 | 19392 | 97.1896 | 97.2653 | 97.1864 | 94.9468 |
| CADSynth | MLP | 99 | 9993 | 99.6488 | 99.4820 | 99.6485 | 98.9710 |
| CADSynth | DiffLoss | 157 | 9993 | 99.6617 | 99.5029 | 99.6614 | 99.0120 |
| MFInstSeg | MLP | 196 | 6250 | 99.4468 | 99.1326 | 99.4464 | 98.2915 |
| MFInstSeg | DiffLoss | 178 | 6250 | 99.4633 | 99.1462 | 99.4630 | 98.3191 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.0039 | +0.0256 | -0.0047 | +0.0490 | MLP |
| Fusion360Seg | +0.0908 | -1.2731 | +0.0978 | -1.3890 | DiffLoss |
| MFCAD++ | -0.0197 | -0.0428 | -0.0195 | -0.0837 | MLP |
| TMCAD | +0.1840 | +0.3415 | +0.3074 | +0.4786 | DiffLoss |
| SolidLetters | -0.2733 | -0.2495 | -0.2602 | -0.4397 | MLP |
| CADSynth | +0.0129 | +0.0209 | +0.0129 | +0.0410 | DiffLoss |
| MFInstSeg | +0.0164 | +0.0137 | +0.0166 | +0.0276 | DiffLoss |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `fa51a5408cfb77ac6c4d0d76c694843f70e190819613fe8c08dfdc01161ce0cb`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `5f1c953e90c03461cfe8286f3540912b340d750a7c6a1acc94e5e7ee5070333b`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `2a78b5b63be1cdfb5fe623ba5522d21d7a6171db4b5a91aca80118219f1ba77f`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `00280e690af132747d55f49d459e0be8b222e4b8f6f34069686760e3947cc3fe`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `b1df5b9983106e432319f9a0e7f1d3a5a82a180923727aa090579968fdad2a39`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `567e2b07abbb2b8548a7f7f533776cba874f8a463bc34b66cac2e7d4096c325a`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `8ff59dd2f397ab369c2f3d2d14ec474970855fd1d20a4f3e9d41ce279a06a8ee`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `deb63c3f0a75bcd5151cd299b83732e5d31cf635bf5d1f4ce2c25933dd2dc99b`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-083543_full_solidletters_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `61d6fa011a48b258c86354ff513865295b1afa38d297c5bae6ba43b85f186e68`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-083544_full_solidletters_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `2dca6cde4a92ea550c6569fa1361d0caaf6aad48a8bb1ae059cfeb0027c99015`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `19ba273f5044a6622e9533778e3475003ed131bbd1e3e7230eedf3d7e25a04d6`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `a2bbea825d413ec86243c58e81503e5e3a17b0e19a8cb3d3461d34664a8fac7b`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `8abd02884ec80bd8e65bb0fbaaba35390707ace2b200194e1107031ca1654df8`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final`; SHA-256 `c4517c1539744d0bddffea90e3c879aab53e8476ed8de5759239797361a65a37`
