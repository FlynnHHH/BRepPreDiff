# FabWave 删除低样本类别后的 DiffLoss 结果

日期：2026-08-06

## 数据处理

在已经删除 `Rotary_Shaft`、302 个 Washers/O-Rings 重叠模型及其缓存的数据基础上，按照 train、validation、test 三个有效 split 的总样本数统计类别。

删除的类别：

| 类别 | 有效样本数 |
|---|---:|
| Webbing Guide | 0 |
| Miter Gears | 1 |
| Sleeve Washers | 7 |

删除后保留 40 类、3,989 条有效数据，重新映射为连续的 0–39 class ID：

| Split | 样本数 |
|---|---:|
| Train | 3,191 |
| Validation | 407 |
| Test | 391 |

原始 STEP 文件未物理删除；上述类别仅从派生微调数据、标签、缓存和 split 中排除。

## 训练设置

- DiffLoss 分类头，200 epochs。
- 从联合预训练 checkpoint 初始化编码器。
- `best.pt` 选择依据：最高 validation accuracy。
- 最优 checkpoint：epoch 63。
- 最优 validation accuracy：97.052%。
- 对应 validation macro F1：99.009%。

## Test 结果

| 指标 | 结果 |
|---|---:|
| Test samples | 391 |
| Accuracy | **97.954%** |
| Macro precision | 99.556% |
| Macro recall | 99.524% |
| Macro F1 | **99.493%** |
| Macro IoU | 99.079% |
| Weighted F1 | **97.946%** |
| Weighted IoU | 96.272% |
| 正确 / 错误 | 383 / 8 |

40 个保留类别在 test split 中都有至少一个样本，因此本次 macro 指标不受零 support 类别影响。

## 错误构成

- 8 个真实 `Washers` 被预测为 `O_Rings`。
- `O_Rings`：precision 82.222%，recall 100%，F1 90.244%。
- `Washers`：precision 100%，recall 80.952%，F1 89.474%。
- 其余 38 类在本次 test split 上均全部预测正确。

## 结果文件

- 运行目录：`runs/finetune/20260806-172206_fabwave_min10_diffloss_acc_200`
- 最优模型：`checkpoints/best.pt`
- 完整指标和混淆矩阵：`test_metrics.json`
- 训练日志：`logs/finetune.log`
- 类别映射与删除审计：`data/splits/fabwave_no_rotary_shaft_no_washer_overlap_min10_class_map.json`
