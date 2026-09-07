# 单数据集 train-only 预训练：7 个下游任务 MLP 与 DiffLoss 对照

所有实验使用每个下游任务各自数据集 train split 上独立训练的 100-epoch encoder、相同数据划分、
随机种子 42、200 epochs、task batch size 256/512、梯度累积 1、GPU 4 5 6 7 队列并行，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 165 | 1787 | 98.7920 | 96.7916 | 98.7888 | 93.8494 |
| BRepPreDiff | DiffLoss | 143 | 1787 | 98.6920 | 96.5669 | 98.6899 | 93.4391 |
| Fusion360Seg | MLP | 186 | 5366 | 95.9907 | 89.5833 | 95.9520 | 82.9123 |
| Fusion360Seg | DiffLoss | 195 | 5366 | 95.9362 | 88.7000 | 95.9053 | 81.8820 |
| MFCAD++ | MLP | 185 | 8949 | 99.4193 | 99.0658 | 99.4189 | 98.1618 |
| MFCAD++ | DiffLoss | 180 | 8949 | 99.4081 | 99.0674 | 99.4080 | 98.1644 |
| TMCAD | MLP | 198 | 1087 | 83.6247 | 83.0324 | 83.4575 | 71.9098 |
| TMCAD | DiffLoss | 186 | 1087 | 83.9926 | 83.5942 | 83.8994 | 72.5329 |
| SolidLetters | MLP | 187 | 19392 | 97.5454 | 97.6122 | 97.5384 | 95.5844 |
| SolidLetters | DiffLoss | 186 | 19392 | 97.2618 | 97.3423 | 97.2636 | 95.0888 |
| CADSynth | MLP | 57 | 9993 | 99.5562 | 99.2635 | 99.5557 | 98.5443 |
| CADSynth | DiffLoss | 70 | 9993 | 99.5723 | 99.2764 | 99.5719 | 98.5685 |
| MFInstSeg | MLP | 200 | 6250 | 99.2756 | 98.7786 | 99.2749 | 97.6123 |
| MFInstSeg | DiffLoss | 180 | 6250 | 99.2280 | 98.6895 | 99.2276 | 97.4447 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BRepPreDiff | -0.1000 | -0.2247 | -0.0990 | -0.4103 | MLP |
| Fusion360Seg | -0.0545 | -0.8833 | -0.0468 | -1.0304 | MLP |
| MFCAD++ | -0.0112 | +0.0016 | -0.0109 | +0.0026 | MLP |
| TMCAD | +0.3680 | +0.5617 | +0.4419 | +0.6231 | DiffLoss |
| SolidLetters | -0.2836 | -0.2699 | -0.2749 | -0.4957 | MLP |
| CADSynth | +0.0161 | +0.0129 | +0.0162 | +0.0242 | DiffLoss |
| MFInstSeg | -0.0475 | -0.0891 | -0.0473 | -0.1675 | MLP |

## Run artifacts

- BRepPreDiff / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-003711_single_brepprediff_seg_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `25b28619b7018cf7edaa1b8bc25bff6b0ed8c3c3a7d35d237b98c04dedc653bb`
- BRepPreDiff / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-012114_single_brepprediff_seg_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `3dfca9e4d971b84bd010844ff715fd1cd5c779769d8cc81978b5757300f202a4`
- Fusion360Seg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-004033_single_fusion360seg_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `95aa98729aaea7a4c9c3023d92111ab9f765de0774400699a3429b9f018dde52`
- Fusion360Seg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-011906_single_fusion360seg_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `d7baf12411ee11b3302e2904733f9aa8d004f11239bd1454e9f892cf8712b7fe`
- MFCAD++ / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-221041_single_mfcadpp_seg_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `65c207ad7904e25e7df45a442afd7d92bdac0a72d6c25fe19a4ed28425d31d51`
- MFCAD++ / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-230559_single_mfcadpp_seg_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `63d3e3564b28177eeefab9281b015e22a4dee8c4a63cef0a3d724937a23a2526`
- TMCAD / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-022136_single_tmcad_cls_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `49b7fd66b59ea92d50e17eebb1a1f87bc1e440eb42eec219c1eab7599ce76bdb`
- TMCAD / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-030658_single_tmcad_cls_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `d47da297966d350c330f97495cd960b9f401e527171421d100a3f267c1151463`
- SolidLetters / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-223826_single_solidletters_cls_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `772f14533e4492cfa77ec708305fa383d9f90e3fb9622949a7d0ffb857dc745d`
- SolidLetters / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260905-000521_single_solidletters_cls_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `6bf3b7a7c3cbd3144fb47994224823ca357d7756ab3c6c46fc366f32627474e8`
- CADSynth / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-224222_single_cadsynth_seg_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `e4c850a1fcaa13c7ed51b5d9294bd31cc315fbf6c679ae52f011865d10ccfcab`
- CADSynth / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-235955_single_cadsynth_seg_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `385a7beec3835649d33f15246ad89a8a9d16c3d601d42f63dc36d06b3119101c`
- MFInstSeg / MLP: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-221747_single_mfinstseg_seg_mlp_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `5ce337426d6955e8563694dfd9a67a7f869753d0374aced7b86cf8f04b136f43`
- MFInstSeg / DiffLoss: `/home/nvme03/hhfeng/BRepPreDiff/runs/single_dataset_pretrain/finetune/20260904-231017_single_mfinstseg_seg_diffloss_ft200_seed42_single_dataset_pretrain_pre100_ft200_seed42_20260904-213106`; SHA-256 `2f1db88f42646f7f7414ba08f0633511db44aa4784e4dfad7f342c375814b567`
