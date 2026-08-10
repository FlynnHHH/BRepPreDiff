# FabWave 删除 Rotary_Shaft 后的 DiffLoss 微调结果

生成日期：2026-08-06

## 结论

从 FabWave 微调数据中完整删除 `Rotary_Shaft`、将类别从 44 类重映射为 43 类并重新训练
DiffLoss 后，过滤后 test split 的 Accuracy 为 **91.943%**，配置口径 Macro-F1 为
**93.803%**，Weighted-F1 为 **92.099%**。

`Keyway_Shaft` 的 25 个测试样本全部预测正确；原来的
`Keyway_Shaft ↔ Rotary_Shaft` 28 个双向错误完全消失。新模型的 34 个测试错误全部集中在
`O_Rings ↔ Washers`。

## 数据处理

原始数据和原 44 类实验均未覆盖或删除。本实验使用独立派生数据：

| Split | 原样本数 | 删除 Rotary_Shaft | 新样本数 |
|---|---:|---:|---:|
| Train | 3,639 | 200 | 3,439 |
| Validation | 463 | 25 | 438 |
| Test | 447 | 25 | 422 |
| 合计 | 4,549 | 250 | 4,299 |

类别处理规则：

- 删除原 class 19 `Rotary_Shaft`。
- 原 class 0–18 保持不变。
- 原 class 20–43 前移为 class 19–42。
- 派生 `.cls` 和特征 cache 均已重写标签，而不是只过滤 split，避免旧 44 类 ID 泄漏进
  43 类模型。
- 验证结果：train/val/test 的全部缓存标签均位于 0–42；相关分类测试 6 项通过。

## 训练协议

- 模型：43 类 DiffLoss classification head。
- Encoder 初始化：
  `runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt`
- Epochs：200。
- Batch size：256。
- Optimizer：AdamW，learning rate `3e-4`，weight decay `1e-4`。
- DiffLoss：bipolar one-hot、`x_start` prediction、4 个 noise samples、1-step DDIM。
- Seed：42。
- Checkpoint 选择：最高 validation Macro-F1。
- Best epoch：158。
- Best validation Accuracy / Macro-F1：93.151% / 94.037%。
- 训练日志未出现 NaN、CUDA OOM 或其他错误。

## 正式测试结果

使用 epoch 158 的 `best.pt`，对过滤后的完整 422 个 test 样本单进程评估：

| 指标 | 结果 |
|---|---:|
| Accuracy | **91.943%** |
| Macro-Precision | 93.791% |
| Macro-Recall | 93.853% |
| Macro-F1 | **93.803%** |
| Macro-IoU | 93.055% |
| Weighted-F1 | **92.099%** |
| Weighted-IoU | 88.036% |
| 正确 / 错误 | 388 / 34 |

配置 Macro 指标仍包含两个零 test support 类别 `Miter Gears` 和 `Webbing Guide`，两者按 0
计入。只在 41 个有 test support 的类别上计算，Macro-F1 为 **98.379%**。

## 剩余错误

| 真实类别 | 预测类别 | 错误数 |
|---|---|---:|
| Washers | O_Rings | 21 |
| O_Rings | Washers | 13 |
| **合计** | — | **34** |

相关逐类结果：

| 类别 | Support | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| Keyway_Shaft | 25 | 100.000% | 100.000% | 100.000% | 100.000% |
| O_Rings | 37 | 53.333% | 64.865% | 58.537% | 41.379% |
| Washers | 72 | 79.688% | 70.833% | 75.000% | 60.000% |

除 `O_Rings` 和 `Washers` 外，其余 39 个有 test support 的类别全部预测正确。

## 与原 44 类模型对比

原模型：
`runs/finetune/20260805-132209_fabwave_diffloss_20260805-unlabeled-v2`。

### 直接报告口径

| 指标 | 原 44 类 / 447 test | 新 43 类 / 422 test | 变化 |
|---|---:|---:|---:|
| Accuracy | 85.906% | **91.943%** | **+6.037 pp** |
| Macro-Precision | 91.436% | **93.791%** | +2.355 pp |
| Macro-Recall | 91.536% | **93.853%** | +2.318 pp |
| Macro-F1 | 91.392% | **93.803%** | **+2.411 pp** |
| Macro-IoU | 89.938% | **93.055%** | +3.117 pp |
| Weighted-F1 | 86.073% | **92.099%** | **+6.026 pp** |
| Weighted-IoU | 80.390% | **88.036%** | +7.646 pp |
| 有 support 类别 Macro-F1 | 95.744% | **98.379%** | +2.635 pp |

这组直接差值同时包含“删除 25 个 Rotary_Shaft 测试样本”和“重新训练 43 类模型”两个
影响，不能全部解释为模型能力提升。

### 相同 422 个测试样本上的 Accuracy

从原模型 confusion matrix 中只移除真实类别为 `Rotary_Shaft` 的 25 个样本后，原模型在
剩余同一批 422 个样本上正确 371 个，Accuracy 为 87.915%。新模型正确 388 个，Accuracy
为 91.943%，提升 **4.028 pp**，错误数从 51 降到 34。

| 错误类型 | 原模型（同一 422 test） | 新模型 |
|---|---:|---:|
| Keyway_Shaft → Rotary_Shaft | 16 | 0 |
| O_Rings ↔ Washers | 35 | 34 |
| 总错误 | 51 | 34 |

因此，同样本上的 17 个净减少错误中，16 个来自消除 `Keyway_Shaft → Rotary_Shaft`，另有
1 个来自 `O_Rings/Washers` 混淆的净减少。主要收益符合“两个 shaft 类定义完全相同、应删去
重复类别”的判断；该修改基本没有解决另一组独立的 `O_Rings/Washers` 边界问题。

## 产物

- 派生数据配置：`data/fabwave_no_rotary_shaft.yaml`
- 派生类映射：`data/splits/fabwave_no_rotary_shaft_class_map.json`
- 派生数据脚本：`scripts/prepare_fabwave_without_class.py`
- 微调配置：`configs/finetune_joint_fabwave_no_rotary_shaft_diffloss_200.yaml`
- Run directory：
  `runs/finetune/20260806-154943_fabwave_no_rotary_shaft_diffloss_20260806`
- Test metrics：上述 run directory 下的 `test_metrics.json`
- Best checkpoint SHA-256：
  `7b11f2c239ee6106e19edd00c184ecf355ba6fba3599424136b44668f1fae623`
