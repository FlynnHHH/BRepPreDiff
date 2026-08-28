# 新 OCC 特征：全部下游任务 MLP 与 DiffLoss 对照

所有实验使用同一 Edge Update Attention 新特征预训练 encoder、相同数据划分、
随机种子 42、200 epochs、batch size 64、梯度累积 4，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BlendIt | MLP | 194 | 1787 | 98.8745 | 97.1210 | 98.8713 | 94.4582 |
| BlendIt | DiffLoss | 177 | 1787 | 98.8677 | 97.0653 | 98.8656 | 94.3549 |
| Fusion360Seg | MLP | 199 | 5366 | 96.0439 | 89.8028 | 96.0135 | 83.1263 |
| Fusion360Seg | DiffLoss | 164 | 5366 | 96.0179 | 89.3881 | 95.9988 | 82.6592 |
| MFCAD++ | MLP | 190 | 8949 | 99.4420 | 99.1314 | 99.4417 | 98.2892 |
| MFCAD++ | DiffLoss | 174 | 8949 | 99.4405 | 99.1101 | 99.4406 | 98.2467 |
| TMCAD | MLP | 25 | 1087 | 83.9006 | 83.1678 | 83.5633 | 72.0196 |
| TMCAD | DiffLoss | 155 | 1087 | 83.7167 | 83.3542 | 83.6900 | 72.3751 |
| FabWave | MLP | 7 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave | DiffLoss | 94 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BlendIt | -0.0068 | -0.0556 | -0.0056 | -0.1034 | MLP |
| Fusion360Seg | -0.0260 | -0.4147 | -0.0148 | -0.4671 | MLP |
| MFCAD++ | -0.0015 | -0.0213 | -0.0012 | -0.0425 | MLP |
| TMCAD | -0.1840 | +0.1864 | +0.1267 | +0.3555 | MLP |
| FabWave | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Tie |

## Run artifacts

- BlendIt / MLP: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-120317_seven_source_711_blendit_seg_mlp_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `b47953def0ffe340c31cc58bbc9117012b75c6a777010f9d1c6cc5e9d7c05733`
- BlendIt / DiffLoss: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-140123_seven_source_711_blendit_seg_diffloss_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `38018dcbc98fd091940fa31491ab57d78ecad527e5f47d360dca44af48912607`
- Fusion360Seg / MLP: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-120317_seven_source_711_fusion360seg_mlp_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `8d13bca1aea1ebab36ace79c974bfaa74923b553aa8c846131f0b88b8b088401`
- Fusion360Seg / DiffLoss: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-154651_seven_source_711_fusion360seg_diffloss_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `d6b82bf8dbd176de9e864466743c3dd03099266b6f4c6f6e4446ed65819711e3`
- MFCAD++ / MLP: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-120317_seven_source_711_mfcadpp_seg_mlp_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `64ab71c2b20bbc8aa480269239818a4cb03af49c095e4d846ae975e564e168d1`
- MFCAD++ / DiffLoss: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-180527_seven_source_711_mfcadpp_seg_diffloss_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `6db5f8fe187715bd12baba16e8fd218b25bae8edb81a11c6ea349f59b04e1994`
- TMCAD / MLP: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-120317_seven_source_711_tmcad_cls_mlp_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `ae6671b90d2a87c31b4ca56c83f4c5a06ea3dbf2fff6bf454dc5fee24e1718fc`
- TMCAD / DiffLoss: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260827-003051_seven_source_711_tmcad_cls_diffloss_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `cfe9684eaadca6cdf95a86bc3c47198dfe13a65e77ca088c0399f5fb4f50e375`
- FabWave / MLP: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-162043_seven_source_711_fabwave_cls_mlp_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `f586a85e4bb94fecbb552a4d953da3fefb29421342762633b403118721fbfabb`
- FabWave / DiffLoss: `/home/hhfeng/Blendit/runs/batchsize_encoder_downstreams/finetune/20260826-193325_seven_source_711_fabwave_cls_diffloss_200_lr5e4_encoder_all_downstreams_20260826-1150`; SHA-256 `788755e478bbdf3e2f2728e45eb2e01d36030cab77849899a978d49ac951a9f2`
