# Geometry-only inductive 9-source encoder：7 个下游 MLP 与 DiffLoss 对照

所有实验使用100-epoch inductive 9-source（无 FabWave）encoder；离散属性 loss 权重为 0、相同数据划分、
随机种子 42、100 epochs、task batch size 256/512、梯度累积 1、GPU 4 并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 23 | 1787 | 98.8978 | 97.0552 | 98.8951 | 94.3353 |
| BRepPreDiff | DiffLoss | 92 | 1787 | 98.8104 | 96.8248 | 98.8069 | 93.9098 |
| Fusion360Seg | MLP | 87 | 5366 | 95.7882 | 88.8898 | 95.7620 | 82.0201 |
| Fusion360Seg | DiffLoss | 77 | 5366 | 95.9323 | 89.3897 | 95.9135 | 82.6174 |
| MFCAD++ | MLP | 95 | 8949 | 99.3479 | 98.9830 | 99.3474 | 98.0015 |
| MFCAD++ | DiffLoss | 93 | 8949 | 99.3141 | 98.9230 | 99.3142 | 97.8852 |
| TMCAD | MLP | 81 | 1087 | 82.4287 | 81.7453 | 82.1877 | 70.2145 |
| TMCAD | DiffLoss | 89 | 1087 | 83.0727 | 82.6569 | 83.0324 | 71.3300 |
| SolidLetters | MLP | 81 | 19392 | 97.4938 | 97.5546 | 97.4850 | 95.4349 |
| SolidLetters | DiffLoss | 94 | 19392 | 97.4216 | 97.4811 | 97.4084 | 95.3077 |
| CADSynth | MLP | 64 | 9993 | 99.5805 | 99.3249 | 99.5801 | 98.6639 |
| CADSynth | DiffLoss | 85 | 9993 | 99.5991 | 99.3465 | 99.5986 | 98.7058 |
| MFInstSeg | MLP | 98 | 6250 | 99.2087 | 98.6659 | 99.2081 | 97.4009 |
| MFInstSeg | DiffLoss | 97 | 6250 | 99.1840 | 98.6154 | 99.1835 | 97.3007 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.0874 | -0.2304 | -0.0882 | -0.4255 | MLP |
| Fusion360Seg | +0.1440 | +0.4999 | +0.1515 | +0.5973 | DiffLoss |
| MFCAD++ | -0.0338 | -0.0600 | -0.0332 | -0.1164 | MLP |
| TMCAD | +0.6440 | +0.9116 | +0.8447 | +1.1155 | DiffLoss |
| SolidLetters | -0.0722 | -0.0735 | -0.0767 | -0.1273 | MLP |
| CADSynth | +0.0186 | +0.0216 | +0.0185 | +0.0419 | DiffLoss |
| MFInstSeg | -0.0246 | -0.0505 | -0.0246 | -0.1002 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_brepprediff_seg_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `3afe7a9071e9beb7c4a695f843ac8adabf8c7fcc4511274cc957154a2b1e87d7`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_brepprediff_seg_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `fa5e82d3d729a4a4e362b42f3c4cf0f4a06707eaefd806a4ec038850769e6fa2`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_fusion360seg_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `09266f431b40a839af8d13f765c7f216d6235ec93b3e03772e82c3f5f7f84709`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-105200_geometry_only_fusion360seg_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `288e1ef3a86d4767dc5aea59bea1873dca158398efb325f188d119175216144f`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-122432_geometry_only_mfcadpp_seg_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `e0e57b0b472d3e45b12539f09cc01b843a7db09ab265e12e0ead52326ecf15d4`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_mfcadpp_seg_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `9a4a85411fbe2757e309e09084563c0195db47de0a413d8fd498950099899212`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-122432_geometry_only_tmcad_cls_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `07c8a0d14ffbd0a874c9840511bab7950ce9652006e8311b5b331aee352f02da`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_tmcad_cls_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `2a1e1ca2cd77bbc9e2fbbbd09d4c02220bc41e12597cb340359b97730087cf71`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_solidletters_cls_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `e6ae39ae7ed822af3c2d904dadf4e8d3b9dfb6ea8dde14a1b38bfbf9f9f9c693`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_solidletters_cls_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `240c7d7975846773bd0de1968d91ef1fcc72f420e004214237c48f97c8f3911b`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_cadsynth_seg_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `eceb919ee65da868e950e4edca5cb72944285a433498f13b95a14720a653fa70`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_cadsynth_seg_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `f85d2e9955cff1300c860a6549085bc39033f55167742e236e2e9616f6aa997c`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_mfinstseg_seg_mlp_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `324be696194898fcafcaaeec08ccf20cd8f5c22dbc1744855c5d1a0806a06eb1`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9/finetune/20260902-054811_geometry_only_mfinstseg_seg_diffloss_ft100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901`; SHA-256 `6a760075b995c8ea2a0455cebe766b0ba60cee8a113f8769d33ca1e5fcb4aad7`
