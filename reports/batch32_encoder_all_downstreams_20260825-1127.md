# 新 OCC 特征：全部下游任务 MLP 与 DiffLoss 对照

所有实验使用同一 Edge Update Attention 新特征预训练 encoder、相同数据划分、
随机种子 42、200 epochs、batch size 64、梯度累积 4，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 54 | 1787 | 98.7502 | 96.7458 | 98.7457 | 93.7644 |
| BRepPreDiff | DiffLoss | 111 | 1787 | 98.7590 | 96.9315 | 98.7544 | 94.1108 |
| Fusion360Seg | MLP | 190 | 5366 | 96.0322 | 89.5175 | 96.0181 | 82.7660 |
| Fusion360Seg | DiffLoss | 193 | 5366 | 95.8596 | 89.6147 | 95.8434 | 82.8296 |
| MFCAD++ | MLP | 190 | 8949 | 99.4345 | 99.1151 | 99.4346 | 98.2574 |
| MFCAD++ | DiffLoss | 183 | 8949 | 99.4494 | 99.1249 | 99.4490 | 98.2767 |
| TMCAD | MLP | 143 | 1087 | 82.7047 | 82.2850 | 82.6702 | 70.8423 |
| TMCAD | DiffLoss | 185 | 1087 | 82.6127 | 81.9815 | 82.3892 | 70.3167 |
| FabWave | MLP | 11 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave | DiffLoss | 94 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | +0.0087 | +0.1858 | +0.0087 | +0.3464 | DiffLoss |
| Fusion360Seg | -0.1726 | +0.0972 | -0.1747 | +0.0637 | MLP |
| MFCAD++ | +0.0149 | +0.0099 | +0.0144 | +0.0192 | DiffLoss |
| TMCAD | -0.0920 | -0.3035 | -0.2811 | -0.5256 | MLP |
| FabWave | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Tie |

## Run artifacts

- BRepPreDiff / MLP: `runs/batchsize_encoder_downstreams/finetune/20260825-112655_seven_source_711_brepprediff_seg_mlp_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `5d4476d494d892a078402469a1a066f717894b2bb110d3b38ea3a4a90ad7d92f`
- BRepPreDiff / DiffLoss: `runs/batchsize_encoder_downstreams/finetune/20260825-133313_seven_source_711_brepprediff_seg_diffloss_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `4c2414392e6c95dfc3ca91ab559d3e6fcc0277170994b1fb2df29958e46a4c58`
- Fusion360Seg / MLP: `runs/batchsize_encoder_downstreams/finetune/20260825-112655_seven_source_711_fusion360seg_mlp_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `2f5f65bed9bab336b60cf57b556a5325fc3f29cd88065fade14fcc8a8fc17f81`
- Fusion360Seg / DiffLoss: `runs/batchsize_encoder_downstreams/finetune/20260825-151001_seven_source_711_fusion360seg_diffloss_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `fc4f8fde1a3c6707176f2ec025ea58940d27e6aab3fdcef4eb8471d2bc75980c`
- MFCAD++ / MLP: `runs/batchsize_encoder_downstreams/finetune/20260825-112655_seven_source_711_mfcadpp_seg_mlp_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `4a317197c56187a2b78c128a43b4de54e28edd58728c33ae7d2dd3a982c7c1fa`
- MFCAD++ / DiffLoss: `runs/batchsize_encoder_downstreams/finetune/20260825-173100_seven_source_711_mfcadpp_seg_diffloss_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `1ea013dd0a516b2b8c06a394fc7c3ab0f4e62a4b00d368f4f6a83f2fa7fe2967`
- TMCAD / MLP: `runs/batchsize_encoder_downstreams/finetune/20260825-112655_seven_source_711_tmcad_cls_mlp_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `7b9e1d3716efc45ffd07c9d37aebede6b2781b5cc09c0b634c3766d711ecdccb`
- TMCAD / DiffLoss: `runs/batchsize_encoder_downstreams/finetune/20260825-235753_seven_source_711_tmcad_cls_diffloss_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `0492580a502f37e1ff46f2186d7488cc5d73eff9777dfd4a3b864a7ad1a36258`
- FabWave / MLP: `runs/batchsize_encoder_downstreams/finetune/20260825-155357_seven_source_711_fabwave_cls_mlp_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `565f52b8fb62273861ef4b09b4305328024df756e8551ed2cff05b7e945fa5d2`
- FabWave / DiffLoss: `runs/batchsize_encoder_downstreams/finetune/20260825-191253_seven_source_711_fabwave_cls_diffloss_200_batch32_encoder_all_downstreams_20260825-1127`; SHA-256 `08fdb26d3e1b1ee323a3b02d2dfeffe156030521c46db45c110ef6dd3d5da2a9`
