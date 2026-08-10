# 全量无标签联合预训练：MLP 微调头 Benchmark 报告

生成日期：2026-08-05（Asia/Shanghai）

## 1. 实验结论

四项 MLP 微调及正式 test split 评估均已完成。本实验与同日 DiffLoss 实验使用完全相同
的全量无标签联合预训练 checkpoint、数据 split、200 epochs、batch size 256、学习率
`3e-4`、随机种子 42 和完整 encoder 更新策略；唯一的核心变化是微调头由 DiffLoss
替换为 MLP。

测试结果显示：

- MFCAD++ 和 Fusion360Seg s2.0.0 上，MLP 的 Accuracy、Macro-F1 和 mIoU 均略高。
- TMCAD 上 DiffLoss 明显更好，Accuracy 高 2.208 pp，Macro-F1 高 2.496 pp。
- FabWave 上 DiffLoss 的 Accuracy 高 0.895 pp；Macro-F1 差距较小，为 0.134 pp。

## 2. 共享预训练基础

预训练使用 334,036 个可解析 STEP，包含 Blendit、TMCAD、Fusion360Seg s2.0.1、
MFCAD++、Fusion360Rec r1.0.1、Fusion360Ass j1.0.0 和 FabWave 的配置内全部
train/val/test geometry。

共享 checkpoint：

`runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt`

SHA-256：`f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9`

预训练 encoder 为初始 Blendit 4-layer MLP/FFN GraphMessageLayer。这里的“MLP 微调”
表示 downstream head 也使用 MLP，而不是 DiffLoss label-diffusion head。

## 3. 微调协议

- Fine-tune head：MLP。
- Encoder：不冻结，全部参数联合更新。
- Epochs：200；batch size：256；学习率：`3e-4`。
- Validation：每个 epoch；按 validation macro-F1 保存 `best.pt`。
- 设备：四项任务各使用一张 NVIDIA Titan RTX，并行训练。
- 训练时间：2026-08-05 15:24–16:47；测试评估于 16:48 完成。
- Fusion360Seg 微调数据保持为 s2.0.0。

## 4. 最佳验证结果

| Benchmark | 任务 | Best epoch | Val Accuracy | Val Macro-F1 |
|---|---|---:|---:|---:|
| MFCAD++ | 25 类面分割 | 191 | 99.197% | 98.775% |
| TMCAD | 10 类模型分类 | 176 | 84.312% | 83.850% |
| Fusion360Seg s2.0.0 | 8 类面分割 | 144 | 92.964% | 87.920% |
| FabWave | 44 类模型分类 | 18 | 85.745% | 90.173% |

## 5. MLP 正式测试结果

所有结果均来自 validation macro-F1 最优的 `best.pt`。

| Benchmark | Test 规模 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | 8,949 models / 268,982 faces | **99.169%** | 98.793% | 98.641% | **98.716%** | **97.490%** | 99.168% |
| TMCAD | 1,087 models | **82.061%** | 81.617% | 81.816% | **81.472%** | **69.362%** | 81.926% |
| Fusion360Seg s2.0.0 | 5,366 models / 77,070 faces | **92.862%** | 88.220% | 82.986% | **85.296%** | **76.123%** | 92.781% |
| FabWave | 447 models | **85.011%** | 91.974% | 91.856% | **91.258%** | **89.796%** | 84.372% |

完整 confusion matrix 与逐类 precision、recall、F1、IoU 位于各 run directory 的
`test_metrics.json`。

## 6. MLP 与 DiffLoss 直接对比

差值定义为 `MLP - DiffLoss`，单位为百分点（pp）。正值表示 MLP 更高。

| Benchmark | MLP Acc | DiffLoss Acc | ΔAcc | MLP Macro-F1 | DiffLoss Macro-F1 | ΔF1 | MLP mIoU | DiffLoss mIoU | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | 99.169% | 99.088% | **+0.081** | 98.716% | 98.561% | **+0.155** | 97.490% | 97.194% | **+0.296** |
| TMCAD | 82.061% | 84.269% | **-2.208** | 81.472% | 83.968% | **-2.496** | 69.362% | 73.282% | **-3.920** |
| Fusion360Seg s2.0.0 | 92.862% | 92.735% | **+0.127** | 85.296% | 85.052% | **+0.244** | 76.123% | 75.827% | **+0.296** |
| FabWave | 85.011% | 85.906% | **-0.895** | 91.258% | 91.392% | **-0.134** | 89.796% | 89.938% | **-0.142** |

在当前单 seed 结果中，面分割任务更偏向 MLP，两个模型分类任务更偏向 DiffLoss，
尤其 TMCAD 差异明显。低于约 0.3 pp 的变化仍建议通过多 seed 重复实验确认。

## 7. 产物与校验值

| Benchmark | Run directory | Best epoch | Best checkpoint SHA-256 |
|---|---|---:|---|
| MFCAD++ | `runs/finetune/20260805-152430_mfcadpp_mlp_200_20260805-unlabeled-v2` | 191 | `c0dc4e648ffdf05361e79046dfe75ab918f9015e2a228bdfeb74f50fbf5f5528` |
| TMCAD | `runs/finetune/20260805-152430_tmcad_mlp_200_20260805-unlabeled-v2` | 176 | `0f721c2a99d5f0e19be077b85acc12ed69d11cf5eeba09f2527325557517636b` |
| Fusion360Seg s2.0.0 | `runs/finetune/20260805-152430_fusion360seg_s2_0_0_mlp_200_20260805-unlabeled-v2` | 144 | `ac7ce593e81dcb67ba4bda37094ef6a346a5067e1ff6511b32cd653730a07be9` |
| FabWave | `runs/finetune/20260805-152430_fabwave_mlp_200_20260805-unlabeled-v2` | 18 | `862926c1ee4dc1f3086cabd50f0542519fd62c2c36a3a6eca28e45341742438f` |

后台调度日志：
`runs/launch_logs/joint_unlabeled_mlp_finetune_v2_tmux.log`。

