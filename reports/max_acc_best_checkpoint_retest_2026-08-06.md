# Max-accuracy `best.pt` 重训与测试报告（2026-08-06）

## 1. 目的

在 `best.pt` 的选择策略由 validation Macro-F1 最大改为 validation accuracy 最大后，
重新训练并测试 MFCAD++、TMCAD 和 Fusion360Seg，检查模型选择指标变化对最终测试结果的影响。

三项任务均完整训练 200 epochs，而不是从旧 run 的稀疏周期 checkpoint 中近似选择。
这是必要的，因为旧实验每 10 epochs 才保存一次常规 checkpoint，无法恢复任意验证 epoch
对应的 max-accuracy 权重。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| MFCAD++ | GPU 1，DiffLoss，25 类面分割 |
| TMCAD | GPU 2，DiffLoss，10 类模型分类 |
| Fusion360Seg | GPU 3，DiffLoss，s2.0.0，8 类面分割 |
| 训练轮数 | 200 epochs |
| 随机种子 | 42 |
| 模型选择 | validation `acc` 最大；仅在严格增大时更新 `best.pt` |
| 预训练权重 | `runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt` |
| 预训练权重 SHA-256 | `f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9` |
| 训练时间 | 2026-08-06 17:00–18:25（Asia/Shanghai） |
| 测试完成时间 | 2026-08-06 18:26（Asia/Shanghai） |

测试规模：MFCAD++ 为 8,949 个模型、268,982 个面；TMCAD 为 1,087 个模型；
Fusion360Seg 为 5,366 个模型、77,070 个面。

## 3. Max-accuracy 新结果

下表中的指标均为百分数。Validation 指标来自新 `best.pt` 内保存的验证结果，
Test 指标来自相应 `test_metrics.json` 的完整测试集评估。

| 数据集 | Best epoch | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test weighted-F1 | Test Macro-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | 176 | 99.1893 | 98.7737 | **99.1248** | 98.6323 | 99.1239 | 97.3280 |
| TMCAD | 52 | 83.6697 | 83.2858 | **84.2686** | 83.9380 | 84.3438 | 73.2085 |
| Fusion360Seg s2.0.0 | 146 | 93.0318 | 85.2574 | **92.8078** | 84.7906 | 92.7308 | 75.5451 |

## 4. 与旧 Macro-F1 选择策略对比

旧结果来自 2026-08-05 的同系列 DiffLoss run。变化量单位为百分点，正数表示
max-accuracy 新结果更高。

| 数据集 | 旧 / 新 epoch | 旧 Test acc | 新 Test acc | Acc 变化 | Macro-F1 变化 | weighted-F1 变化 | Macro-IoU 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | 193 / 176 | 99.0880 | 99.1248 | **+0.0368** | +0.0711 | +0.0365 | +0.1342 |
| TMCAD | 126 / 52 | 84.2686 | 84.2686 | **+0.0000** | -0.0296 | -0.0000 | -0.0733 |
| Fusion360Seg s2.0.0 | 175 / 146 | 92.7352 | 92.8078 | **+0.0727** | -0.2613 | +0.0413 | -0.2819 |

结果表明：按 validation accuracy 选择后，MFCAD++ 与 Fusion360Seg 的测试 accuracy
均有提升，TMCAD 持平。Fusion360Seg 的 Macro-F1 和 Macro-IoU 小幅下降，说明以 accuracy
替代 Macro-F1 进行模型选择会更偏向整体正确率，而不保证类别均衡指标同步提高。

TMCAD 的 validation accuracy 在 epoch 52 已达到本次最大值；后续存在相同最大值时，
由于保存条件是严格的 `>`，因此保留首次达到最大 accuracy 的 epoch 52。

## 5. 产物与校验

| 数据集 | Run 目录 | `best.pt` SHA-256 |
|---|---|---|
| MFCAD++ | `runs/finetune/20260806-170032_mfcadpp_diffloss_20260806-max-acc` | `40dbcf237cc5359d9381aed58213f5f4b145f05411b7646f673ee44ea40108d3` |
| TMCAD | `runs/finetune/20260806-170032_tmcad_diffloss_20260806-max-acc` | `d6364ab0bed06e792a32bc3329f60ae5fe781e9e976e09a893bfa8a907fbc9d7` |
| Fusion360Seg s2.0.0 | `runs/finetune/20260806-170032_fusion360seg_s2_0_0_diffloss_20260806-max-acc` | `031e7c721ec6d4f4a80cc600dffaa9079333ac85c61ba4f58e07f3c07fa4b21a` |

每个 run 目录均包含：

- `checkpoints/best.pt`：validation accuracy 最优权重；
- `checkpoints/last.pt`：epoch 200 权重；
- `test_metrics.json`：完整测试指标、混淆矩阵和逐类指标；
- `test_evaluate.log`：测试日志；
- `logs/finetune.log`：完整训练与验证日志。

可复现实验的启动脚本为：
`scripts/run_three_acc_selected_diffloss.sh`。
