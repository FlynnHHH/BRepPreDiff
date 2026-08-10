# MLP 与 DiffLoss：Max-accuracy `best.pt` 八组实验报告

日期：2026-08-07

## 1. 实验目的

在统一采用 validation accuracy 最大值选择 `best.pt` 的条件下，对比 MLP 与 DiffLoss
微调头在以下四项任务上的表现：

1. MFCAD++：25 类面分割；
2. TMCAD：10 类模型分类；
3. Fusion360Seg s2.0.0：8 类面分割；
4. FabWave min10：删除 `Rotary_Shaft`、Washers/O-Rings 重叠样本及有效样本数小于 10
   的类别后得到的 40 类模型分类任务。

共包含 4 组 MLP 和 4 组 DiffLoss 实验。所有实验均完成 200 epochs 训练，并使用各自
validation accuracy 最高的 `best.pt` 在完整 test split 上评估。

## 2. 统一实验设置

| 项目 | 设置 |
|---|---|
| 预训练权重 | `runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt` |
| 预训练权重 SHA-256 | `f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9` |
| Encoder | hidden dim 128，4 层，dropout 0.1，全量微调 |
| Epochs | 200 |
| Batch size | 256 |
| Optimizer | AdamW，learning rate `3e-4`，weight decay `1e-4` |
| Seed | 42 |
| 验证频率 | 每个 epoch |
| `best.pt` 选择 | validation `acc` 最大；严格增大时更新 |
| 测试 batch size | 512 |

四组 MLP 于 2026-08-06 20:18 开始训练，21:41 完成，完整测试于 21:43 完成。
四项任务分别使用 GPU 0、1、2、3。DiffLoss 采用相同预训练权重、训练轮数、随机种子
和模型选择规则。

## 3. 数据与测试规模

| 数据集 | 任务 | 类别数 | Test 模型数 | Test 面数 |
|---|---|---:|---:|---:|
| MFCAD++ | 面分割 | 25 | 8,949 | 268,982 |
| TMCAD | 模型分类 | 10 | 1,087 | — |
| Fusion360Seg s2.0.0 | 面分割 | 8 | 5,366 | 77,070 |
| FabWave min10 | 模型分类 | 40 | 391 | — |

## 4. 八组实验结果

所有指标均以百分数表示。Val 指标来自 `best.pt` 保存的验证结果，Test 指标来自对应
`test_metrics.json`。

| 数据集 | Head | Best epoch | Val Acc | Val Macro-F1 | Test Acc | Test Macro-P | Test Macro-R | Test Macro-F1 | Test weighted-F1 | Test Macro-IoU | Test weighted-IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | MLP | 200 | **99.1897** | 98.7452 | **99.1557** | **98.7791** | **98.5685** | **98.6718** | **99.1546** | **97.4071** | **98.3441** |
| MFCAD++ | DiffLoss | 176 | 99.1893 | **98.7737** | 99.1248 | 98.7341 | 98.5332 | 98.6323 | 99.1239 | 97.3280 | 98.2831 |
| TMCAD | MLP | 37 | 83.4862 | 83.1006 | 81.8767 | 81.4208 | 81.5580 | 81.3106 | 81.7353 | 69.4729 | 70.1217 |
| TMCAD | DiffLoss | 52 | **83.6697** | **83.2858** | **84.2686** | **84.0337** | **83.9724** | **83.9380** | **84.3438** | **73.2085** | **73.8397** |
| Fusion360Seg s2.0.0 | MLP | 158 | **93.2095** | **86.8147** | **92.9752** | **90.6975** | 82.4682 | **85.5269** | **92.8929** | **76.4586** | **86.9858** |
| Fusion360Seg s2.0.0 | DiffLoss | 146 | 93.0318 | 85.2574 | 92.8078 | 87.5821 | **82.7223** | 84.7906 | 92.7308 | 75.5451 | 86.7223 |
| FabWave min10 | MLP | 18 | 96.8059 | 98.1491 | **97.9540** | **99.5556** | **99.5238** | **99.4929** | **97.9461** | **99.0794** | **96.2717** |
| FabWave min10 | DiffLoss | 63 | **97.0516** | **99.0090** | **97.9540** | **99.5556** | **99.5238** | **99.4929** | **97.9461** | **99.0794** | **96.2717** |

## 5. MLP 相对 DiffLoss 的变化

下表为 `MLP - DiffLoss`，单位为百分点。正数表示 MLP 更高，负数表示 DiffLoss 更高。

| 数据集 | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test weighted-F1 | Test Macro-IoU | Test weighted-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | +0.0004 | -0.0285 | **+0.0309** | +0.0395 | +0.0307 | +0.0790 | +0.0609 |
| TMCAD | -0.1835 | -0.1852 | **-2.3919** | -2.6274 | -2.6084 | -3.7356 | -3.7180 |
| Fusion360Seg s2.0.0 | +0.1777 | +1.5573 | **+0.1674** | +0.7363 | +0.1622 | +0.9136 | +0.2635 |
| FabWave min10 | -0.2457 | -0.8599 | **+0.0000** | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## 6. 结论

- **MFCAD++：MLP 略优。** MLP 的测试 accuracy 为 99.1557%，比 DiffLoss 高 0.0309
  个百分点；Macro-F1、weighted-F1 和 IoU 也均小幅提高。两种 head 的差距很小。
- **TMCAD：DiffLoss 明显更优。** DiffLoss 的测试 accuracy 为 84.2686%，比 MLP 高
  2.3919 个百分点；Macro-F1 高 2.6274 个百分点，Macro-IoU 高 3.7356 个百分点。
- **Fusion360Seg：MLP 更优。** MLP 的测试 accuracy 为 92.9752%，高 0.1674 个百分点；
  Macro-F1 高 0.7363 个百分点，Macro-IoU 高 0.9136 个百分点。DiffLoss 仅在
  Macro-Recall 上高 0.2541 个百分点。
- **FabWave min10：测试结果完全相同。** 两个 head 均正确分类 383/391 个测试模型，
  8 个错误均为 Washers 被预测成 O_Rings。虽然 DiffLoss 的 validation accuracy 更高，
  但没有转化为测试集提升；MLP 在 epoch 18 就达到相同测试结果，DiffLoss 的最佳 epoch
  为 63。

若以本次单次实验的 test accuracy 作为选择依据，推荐 MFCAD++ 和 Fusion360Seg 使用
MLP，TMCAD 使用 DiffLoss；FabWave min10 两者测试性能相同，MLP 的最佳 checkpoint
出现更早且 head 更简单。

## 7. Run、checkpoint 与校验值

| 数据集 | Head | Run 目录 | `best.pt` SHA-256 |
|---|---|---|---|
| MFCAD++ | MLP | `runs/finetune/20260806-201810_mfcadpp_mlp_200_20260806-max-acc` | `5f013d6ce1d4e129bfc56e044b55b03935351f62a52e7ab9c127b41bf4684787` |
| MFCAD++ | DiffLoss | `runs/finetune/20260806-170032_mfcadpp_diffloss_20260806-max-acc` | `40dbcf237cc5359d9381aed58213f5f4b145f05411b7646f673ee44ea40108d3` |
| TMCAD | MLP | `runs/finetune/20260806-201810_tmcad_mlp_200_20260806-max-acc` | `58e413a035aedd4a570388ccd0b3463dac7b97db57a4484c7c7270ae70310d62` |
| TMCAD | DiffLoss | `runs/finetune/20260806-170032_tmcad_diffloss_20260806-max-acc` | `d6364ab0bed06e792a32bc3329f60ae5fe781e9e976e09a893bfa8a907fbc9d7` |
| Fusion360Seg s2.0.0 | MLP | `runs/finetune/20260806-201810_fusion360seg_s2_0_0_mlp_200_20260806-max-acc` | `ed7a2814cc52cbed7c2cde243abbfd67ba16e65ef035f5ed4cbceaa97559b38c` |
| Fusion360Seg s2.0.0 | DiffLoss | `runs/finetune/20260806-170032_fusion360seg_s2_0_0_diffloss_20260806-max-acc` | `031e7c721ec6d4f4a80cc600dffaa9079333ac85c61ba4f58e07f3c07fa4b21a` |
| FabWave min10 | MLP | `runs/finetune/20260806-201810_fabwave_min10_mlp_200_20260806-max-acc` | `ae7723399b9a205bed75031c959986957dee98114c2f398000cf66e898c059aa` |
| FabWave min10 | DiffLoss | `runs/finetune/20260806-172206_fabwave_min10_diffloss_acc_200` | `6c8d218f0c12c379b96911483335b0eafdc011c32c5080aef1be95d3acf4291d` |

每个 run 目录均保存 `checkpoints/best.pt`、`checkpoints/last.pt`、训练日志和
`test_metrics.json`。四组 MLP 的复现入口为 `scripts/run_four_acc_selected_mlp.sh`；
三项主数据集 DiffLoss 的复现入口为 `scripts/run_three_acc_selected_diffloss.sh`。
