# FabWave 新 Split DiffLoss 测试报告

日期：2026-08-06

## 实验设置

- 数据处理：删除 `Rotary_Shaft`，并从 Washers 中删除 302 个与 O-Rings 重叠的模型及对应缓存。
- 分层划分：train 3,197，validation 408，test 392；随机种子为 42。
- 训练：DiffLoss 分类头，200 epochs，从联合预训练 checkpoint 初始化。
- 最优模型选择：validation accuracy 最高（`selection_metric: acc`）。
- `best.pt`：epoch 71，validation accuracy 96.814%，validation macro F1 93.000%。
- 测试：仅使用隔离的新 test split，并加载 `best.pt`。

## 最终测试指标

| 指标 | 结果 |
|---|---:|
| Test samples | 392 |
| Accuracy | **97.704%** |
| Macro precision | 91.447% |
| Macro recall | 92.580% |
| Macro F1 | **91.776%** |
| Macro IoU | 91.004% |
| Weighted F1 | **97.611%** |
| Weighted IoU | 95.899% |
| 正确 / 错误 | 383 / 9 |

## 错误构成

- 8 个真实 Washers 被预测为 O_Rings。
- 1 个真实 Sleeve Washers 被预测为 Miter Gear Set Screw。
- O_Rings：precision 82.222%，recall 100%，F1 90.244%。
- Washers：precision 100%，recall 80.952%，F1 89.474%。

配置中保留 43 个输出类别，但新 test split 中 `Miter Gears` 和 `Webbing Guide` 的 support 为 0。当前实现将这些零 support 类别也计入 macro 指标，因此标准输出的 macro F1 为 91.776%；若仅在 41 个实际有 support 的类别上取平均，macro F1 为约 96.253%。报告主表保留程序原始标准输出。

## 结果文件

- 运行目录：`runs/finetune/20260806-165418_fabwave_no_rotary_shaft_no_washer_overlap_diffloss_acc_200`
- 最优模型：`checkpoints/best.pt`
- 完整指标与混淆矩阵：`test_metrics.json`
- 训练日志：`logs/finetune.log`
