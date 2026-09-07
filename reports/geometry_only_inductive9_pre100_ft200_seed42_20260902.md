# Geometry-only inductive 9-source encoder：200-epoch MLP 与 DiffLoss 对照

所有实验使用同一 100-epoch geometry-only inductive 9-source（无 FabWave）encoder、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1、GPU 4 三路并发，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 23 | 1787 | 98.8978 | 97.0552 | 98.8951 | 94.3353 |
| BRepPreDiff | DiffLoss | 92 | 1787 | 98.8104 | 96.8248 | 98.8069 | 93.9098 |
| Fusion360Seg | MLP | 196 | 5366 | 95.9790 | 89.1380 | 95.9510 | 82.3479 |
| Fusion360Seg | DiffLoss | 157 | 5366 | 95.9154 | 89.7308 | 95.8941 | 82.9042 |
| MFCAD++ | MLP | 185 | 8949 | 99.3780 | 99.0267 | 99.3771 | 98.0849 |
| MFCAD++ | DiffLoss | 185 | 8949 | 99.3955 | 99.0395 | 99.3957 | 98.1111 |
| TMCAD | MLP | 199 | 1087 | 82.7047 | 81.8960 | 82.3590 | 70.4132 |
| TMCAD | DiffLoss | 89 | 1087 | 83.0727 | 82.6569 | 83.0324 | 71.3300 |
| SolidLetters | MLP | 184 | 19392 | 97.5815 | 97.6431 | 97.5747 | 95.6025 |
| SolidLetters | DiffLoss | 199 | 19392 | 97.4680 | 97.5323 | 97.4617 | 95.4045 |
| CADSynth | MLP | 64 | 9993 | 99.5805 | 99.3249 | 99.5801 | 98.6639 |
| CADSynth | DiffLoss | 85 | 9993 | 99.5991 | 99.3465 | 99.5986 | 98.7058 |
| MFInstSeg | MLP | 199 | 6250 | 99.2468 | 98.7622 | 99.2458 | 97.5861 |
| MFInstSeg | DiffLoss | 194 | 6250 | 99.2275 | 98.6658 | 99.2266 | 97.3993 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.0874 | -0.2304 | -0.0882 | -0.4255 | MLP |
| Fusion360Seg | -0.0636 | +0.5928 | -0.0570 | +0.5564 | MLP |
| MFCAD++ | +0.0175 | +0.0128 | +0.0186 | +0.0262 | DiffLoss |
| TMCAD | +0.3680 | +0.7609 | +0.6734 | +0.9168 | DiffLoss |
| SolidLetters | -0.1134 | -0.1108 | -0.1130 | -0.1980 | MLP |
| CADSynth | +0.0186 | +0.0216 | +0.0185 | +0.0419 | DiffLoss |
| MFInstSeg | -0.0194 | -0.0964 | -0.0192 | -0.1869 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_brepprediff_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `3afe7a9071e9beb7c4a695f843ac8adabf8c7fcc4511274cc957154a2b1e87d7`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_brepprediff_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `fa5e82d3d729a4a4e362b42f3c4cf0f4a06707eaefd806a4ec038850769e6fa2`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-134814_geometry_only_fusion360seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `afec760896810941b386b693f62fb3ddbc11140884a42eb9a488b1ef8345116c`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-140846_geometry_only_fusion360seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `19a04af095c258c3079806278184296279e13d78a301fd82da4ccd31838a7fdf`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-141140_geometry_only_mfcadpp_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `8ee4afbcc9299e03ec0996c243b6a21fe1bc9a8e92ff3eeac6cf72df90911b4f`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-140720_geometry_only_mfcadpp_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `da40248310e4da76df372907c4e94e26d723f6cf919ec1c570af0921d8517fdb`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-143001_geometry_only_tmcad_cls_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `ddda1188aa387f0c5c352c7ac74a2946e4929270a95f03d5253f1b3be4535843`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-144929_geometry_only_tmcad_cls_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `2a1e1ca2cd77bbc9e2fbbbd09d4c02220bc41e12597cb340359b97730087cf71`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-145015_geometry_only_solidletters_cls_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `5479063acb2af1c506c8b5b3bd09c6568f48719374c257b5dbe7eeed25861c88`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-150400_geometry_only_solidletters_cls_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `09c63435d1877d39ffb1cc660483abb401cc9cc3c7fa2d3c57265ad90e1361f0`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-152527_geometry_only_cadsynth_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `eceb919ee65da868e950e4edca5cb72944285a433498f13b95a14720a653fa70`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-155609_geometry_only_cadsynth_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `f85d2e9955cff1300c860a6549085bc39033f55167742e236e2e9616f6aa997c`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-161343_geometry_only_mfinstseg_seg_mlp_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `4b3263addee3639c3e61a179deb73057e44f22ee7bd3b4c72264a0ac105f6b6e`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/geometry_only_inductive9_ft200/finetune/20260902-161816_geometry_only_mfinstseg_seg_diffloss_ft200_seed42_geometry_only_inductive9_pre100_ft200_seed42_20260902`; SHA-256 `fddad63daf546c15df86b26daecbf09fde5e083192f2083521d98a0dc07891c3`
