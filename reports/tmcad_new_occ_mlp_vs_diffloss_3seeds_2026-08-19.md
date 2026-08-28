# TMCAD 新 OCC 特征：MLP 与 DiffLoss 三随机种子对照

Seed 42/43/44 使用同一预训练 encoder、数据划分、Mean+Max pooling、200 epochs、
batch size 64、梯度累积 4，并按 validation accuracy 选择 best checkpoint。
DiffLoss 使用 `L_x_start + 0.5 * L_epsilon` 和单步 DDIM。

| Seed | Head | Best epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---:|---|---:|---:|---:|---:|---:|
| 42 | MLP | 81 | 84.4526 | 84.0232 | 84.3864 | 73.1963 |
| 42 | DiffLoss | 50 | 81.8767 | 81.2892 | 81.7391 | 69.5518 |
| 43 | MLP | 119 | 83.3487 | 82.9020 | 83.2734 | 71.7704 |
| 43 | DiffLoss | 138 | 82.6127 | 82.0742 | 82.5065 | 70.6108 |
| 44 | MLP | 172 | 83.6247 | 83.2186 | 83.5845 | 72.2475 |
| 44 | DiffLoss | 102 | 84.2686 | 83.8688 | 84.1680 | 73.0721 |

差值定义为 `DiffLoss - MLP`，单位为百分点。

| Seed | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---:|---:|---:|---:|---:|
| 42 | -2.5759 | -2.7340 | -2.6472 | -3.6445 |
| 43 | -0.7360 | -0.8277 | -0.7669 | -1.1596 |
| 44 | +0.6440 | +0.6502 | +0.5836 | +0.8246 |

下表为三个 seed 的均值 ± 样本标准差。

| Head | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---:|---:|---:|---:|
| MLP | 83.8086 ± 0.5745 | 83.3813 ± 0.5781 | 83.7481 ± 0.5742 | 72.4047 ± 0.7258 |
| DiffLoss | 82.9193 ± 1.2251 | 82.4107 ± 1.3223 | 82.8046 ± 1.2416 | 71.0782 ± 1.8061 |

三 seed 平均 Accuracy winner：**MLP**；平均 `DiffLoss - MLP` 为 -0.8893 pp。

## Run artifacts

- Seed 42 / MLP: `runs/new_occ_features_downstreams/finetune/20260819-111619_edge_update_tmcad_cls_mlp_new_occ_all_mlp_200_20260819-1100`
- Seed 42 / DiffLoss: `runs/new_occ_features_downstreams/finetune/20260818-194812_edge_update_tmcad_cls_diffloss_xse_new_occ_diffloss_200_20260818-195000`
- Seed 43 / MLP: `runs/new_occ_features_downstreams/finetune/20260819-153351_edge_update_tmcad_mlp_seed43_new_occ_tmcad_seeds43_44_20260819-153340`
- Seed 43 / DiffLoss: `runs/new_occ_features_downstreams/finetune/20260819-153352_edge_update_tmcad_diffloss_seed43_new_occ_tmcad_seeds43_44_20260819-153340`
- Seed 44 / MLP: `runs/new_occ_features_downstreams/finetune/20260819-153352_edge_update_tmcad_mlp_seed44_new_occ_tmcad_seeds43_44_20260819-153340`
- Seed 44 / DiffLoss: `runs/new_occ_features_downstreams/finetune/20260819-153352_edge_update_tmcad_diffloss_seed44_new_occ_tmcad_seeds43_44_20260819-153340`
