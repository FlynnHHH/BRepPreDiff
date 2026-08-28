# 新 OCC 特征：全部下游任务 MLP 与 DiffLoss 对照

所有实验使用同一 Edge Update Attention 新特征预训练 encoder、相同数据划分、
随机种子 42、200 epochs、batch size 64、梯度累积 4，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BlendIt | MLP | 192 | 1787 | 98.8571 | 97.1710 | 98.8541 | 94.5522 |
| BlendIt | DiffLoss | 75 | 1787 | 98.8338 | 97.0511 | 98.8296 | 94.3293 |
| Fusion360Seg | MLP | 197 | 5366 | 95.8492 | 90.3379 | 95.8147 | 83.5425 |
| Fusion360Seg | DiffLoss | 181 | 5366 | 95.9971 | 89.5362 | 95.9556 | 82.8839 |
| MFCAD++ | MLP | 196 | 8949 | 99.4089 | 99.0792 | 99.4084 | 98.1886 |
| MFCAD++ | DiffLoss | 189 | 8949 | 99.4059 | 99.0533 | 99.4059 | 98.1384 |
| TMCAD | MLP | 36 | 1087 | 82.8887 | 82.4809 | 82.8318 | 71.0397 |
| TMCAD | DiffLoss | 183 | 1087 | 83.0727 | 82.4183 | 82.8417 | 70.8140 |
| FabWave | MLP | 8 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave | DiffLoss | 101 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BlendIt | -0.0233 | -0.1199 | -0.0245 | -0.2229 | MLP |
| Fusion360Seg | +0.1479 | -0.8017 | +0.1409 | -0.6586 | DiffLoss |
| MFCAD++ | -0.0030 | -0.0259 | -0.0025 | -0.0502 | MLP |
| TMCAD | +0.1840 | -0.0627 | +0.0100 | -0.2257 | DiffLoss |
| FabWave | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Tie |

## Run artifacts

- BlendIt / MLP: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260820-225428_seven_source_711_blendit_seg_mlp_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `bf9bcfe7ff0190b82c354d644fa0988ad8b8e291c69f77587c0ffa5aca5bfe11`
- BlendIt / DiffLoss: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260820-235132_seven_source_711_blendit_seg_diffloss_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `8d519e14b89249d63555c5eb50bb75b0d8fed1475c828931a97dd63b2ea718f3`
- Fusion360Seg / MLP: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260820-225428_seven_source_711_fusion360seg_mlp_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `640eb5065fb40322776b20e2661ffe8ca32adf98d93ef9ab571ae419e94e6df2`
- Fusion360Seg / DiffLoss: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260821-001918_seven_source_711_fusion360seg_diffloss_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `0f3498e720232f1a4ddd376f6a22c65ecae57653fd86744a822f144c0a4d6d9d`
- MFCAD++ / MLP: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260820-225428_seven_source_711_mfcadpp_seg_mlp_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `f776639bc73d0cefff9e1ddccac3d0ea52fd78716f30731e1222e7f7f9729a9d`
- MFCAD++ / DiffLoss: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260821-011822_seven_source_711_mfcadpp_seg_diffloss_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `eacff2cc41e824e1b461427a520d6313f26a2ffe69402fc837167dc324579964`
- TMCAD / MLP: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260820-225428_seven_source_711_tmcad_cls_mlp_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `d82d22c78ac73acba85be4ea1d8032668633a1aafab6e284febf7135f6c08d1a`
- TMCAD / DiffLoss: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260821-000037_seven_source_711_tmcad_cls_diffloss_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `c62215289419d65145b1bd429b6277790afb82e031b8468677b090a0fdd7ce79`
- FabWave / MLP: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260821-005437_seven_source_711_fabwave_cls_mlp_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `955e50af36b31b03f70f5961a1833dc23abf828d87e66ebc20b7233ce5a59c82`
- FabWave / DiffLoss: `/home/hhfeng/Blendit/runs/seven_source_711_downstreams/finetune/20260821-011058_seven_source_711_fabwave_cls_diffloss_200_seven_source_711_all_downstreams_20260820-204011`; SHA-256 `c4eebb3484d6f21f6f5aeb45f92f1614cadc252080936ac736612b138271f25c`
