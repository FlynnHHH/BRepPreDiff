# FabWave 新 OCC 特征：MLP 与 DiffLoss 对照

两组实验使用相同的新 OCC 特征缓存、Edge Update Attention encoder 预训练权重、
随机种子、Mean+Max pooling、200-epoch 预算、batch size 64、梯度累积 4，并按验证集
accuracy 选择 best checkpoint。差值定义为 `DiffLoss - MLP`。

| Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 15 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| DiffLoss | 89 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

| Comparison | Accuracy (pp) | Macro-F1 (pp) | Weighted-F1 (pp) | mIoU (pp) |
|---|---:|---:|---:|---:|
| DiffLoss - MLP | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

Accuracy winner: **Tie**.

- MLP run: `runs/new_occ_features_downstreams/finetune/20260819-105531_edge_update_fabwave_cls_mlp_new_occ_fabwave_mlp_200_20260819-1040`; best.pt SHA-256 `a776993142068e596669f139d8434aee31f3d7351be5a445d1284c72d249e1f1`
- DiffLoss run: `runs/new_occ_features_downstreams/finetune/20260819-104736_edge_update_fabwave_cls_diffloss_xse_new_occ_fabwave_diffloss_200_20260819-103615`; best.pt SHA-256 `fc4c8dd436b2ced5263ba4e82d954e73c76a15ac859d8b065330f7774fd1dbe6`
