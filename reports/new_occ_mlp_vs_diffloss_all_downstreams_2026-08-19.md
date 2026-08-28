# 新 OCC 特征：全部下游任务 MLP 与 DiffLoss 对照

所有实验使用同一 Edge Update Attention 新特征预训练 encoder、相同数据划分、
随机种子 42、200 epochs、batch size 64、梯度累积 4，并按 validation accuracy
选择 best checkpoint。分类任务统一使用 Mean+Max pooling。

| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BlendIt | MLP | 69 | 1787 | 98.8561 | 97.0293 | 98.8541 | 94.2883 |
| BlendIt | DiffLoss | 157 | 1787 | 98.8231 | 97.0099 | 98.8190 | 94.2534 |
| Fusion360Seg | MLP | 172 | 5366 | 95.9530 | 89.3850 | 95.9273 | 82.6929 |
| Fusion360Seg | DiffLoss | 192 | 5366 | 95.9868 | 89.4871 | 95.9612 | 82.7699 |
| MFCAD++ | MLP | 162 | 8949 | 99.4052 | 99.0675 | 99.4050 | 98.1650 |
| MFCAD++ | DiffLoss | 198 | 8949 | 99.4156 | 99.0989 | 99.4153 | 98.2266 |
| TMCAD | MLP | 81 | 1087 | 84.4526 | 84.0232 | 84.3864 | 73.1963 |
| TMCAD | DiffLoss | 50 | 1087 | 81.8767 | 81.2892 | 81.7391 | 69.5518 |
| FabWave | MLP | 15 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave | DiffLoss | 89 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |
|---|---:|---:|---:|---:|---|
| BlendIt | -0.0330 | -0.0194 | -0.0351 | -0.0349 | MLP |
| Fusion360Seg | +0.0337 | +0.1021 | +0.0339 | +0.0770 | DiffLoss |
| MFCAD++ | +0.0104 | +0.0314 | +0.0103 | +0.0616 | DiffLoss |
| TMCAD | -2.5759 | -2.7340 | -2.6472 | -3.6445 | MLP |
| FabWave | +0.0000 | +0.0000 | +0.0000 | +0.0000 | Tie |

## Run artifacts

- BlendIt / MLP: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-110819_edge_update_blendit_seg_mlp_new_occ_all_mlp_200_20260819-1100`; SHA-256 `91c016be72a1c17335f5d811eea6cff507ddf68666eb80cf961b102a832bb6b6`
- BlendIt / DiffLoss: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260818-211021_edge_update_blendit_seg_diffloss_xse_new_occ_diffloss_200_gpu0_20260818-1953`; SHA-256 `63698a2bfe78f8529cc5529d45f78fdad40e2d90a983dd92bff56e6ecea65dcd`
- Fusion360Seg / MLP: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-110419_edge_update_fusion360seg_mlp_new_occ_all_mlp_200_20260819-1100`; SHA-256 `622b026349a72a9aca909c97c30ce079cbcbd3708116900e4db3510be714ddc5`
- Fusion360Seg / DiffLoss: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260818-194812_edge_update_fusion360seg_diffloss_xse_new_occ_diffloss_200_20260818-195000`; SHA-256 `953e5e408b25f8968f15158c176889446ecb971115a44bc7d60f77268ba1e901`
- MFCAD++ / MLP: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-110419_edge_update_mfcadpp_seg_mlp_new_occ_all_mlp_200_20260819-1100`; SHA-256 `72d59ac5f837724d747b40223798df28f0c2a8c020de1b63102b695099e8ea2f`
- MFCAD++ / DiffLoss: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260818-194812_edge_update_mfcadpp_seg_diffloss_xse_new_occ_diffloss_200_20260818-195000`; SHA-256 `d498c9c356ab06410b76c4a9497c89f930d41fd55ad8709e85fa33c5b7f44273`
- TMCAD / MLP: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-111619_edge_update_tmcad_cls_mlp_new_occ_all_mlp_200_20260819-1100`; SHA-256 `f9f8326450cad13936d57aa8d94452bc9d34772cf5dbca99e3a4ec279ab9d076`
- TMCAD / DiffLoss: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260818-194812_edge_update_tmcad_cls_diffloss_xse_new_occ_diffloss_200_20260818-195000`; SHA-256 `aebb68913d7f180d9ebe021afdd3f676b03ae94e489a805dcde221121c494122`
- FabWave / MLP: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-105531_edge_update_fabwave_cls_mlp_new_occ_fabwave_mlp_200_20260819-1040`; SHA-256 `a776993142068e596669f139d8434aee31f3d7351be5a445d1284c72d249e1f1`
- FabWave / DiffLoss: `/home/hhfeng/Blendit/runs/new_occ_features_downstreams/finetune/20260819-104736_edge_update_fabwave_cls_diffloss_xse_new_occ_fabwave_diffloss_200_20260819-103615`; SHA-256 `fc4c8dd436b2ced5263ba4e82d954e73c76a15ac859d8b065330f7774fd1dbe6`
