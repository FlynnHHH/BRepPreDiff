# Fusion360Seg 连续特征预处理消融报告

实验日期：2026-08-04

## 结论

在本次单随机种子、完整 100 epochs 微调实验中，**完全不标准化（`none`）是 Fusion360Seg 上表现最好的方案**：test Accuracy `93.11%`、Macro-F1 `86.34%`、mIoU `77.43%`。它相对当前逐图标准化基线分别提高 `0.69`、`1.80` 和 `2.43` 个百分点。

逐图标准化（`per_graph`）排名第二；按特征类型处理并使用联合训练集全局统计量（`typewise_global`）明显退化，Macro-F1 比逐图标准化低 `11.56` 个百分点。

因此，对于当前 Fusion360Seg 数据、模型和微调协议，建议采用 `none`。不过本实验只有 seed 42，尚未量化随机波动；正式替换默认方案前建议至少补跑 3 个随机种子。

## 实验设置

三组实验均使用与预训练方式匹配的输入预处理和 checkpoint：

| 方法 | 预训练 checkpoint | 微调时预处理 |
|---|---|---|
| `per_graph` | `20260803-155054_joint_ablation_per_graph/last.pt` | 单图内逐通道 z-score |
| `none` | `20260803-173114_joint_ablation_none/last.pt` | 不做额外标准化 |
| `typewise_global` | `20260803-190220_joint_ablation_typewise_global/last.pt` | 非方向通道全局 z-score；法向量 L2；角度/点积限幅 |

共同微调协议：

| 项目 | 设置 |
|---|---|
| 数据集 | Fusion360Seg |
| 任务 | 8 类面分割 |
| 划分 | 24,964 train / 5,350 val / 5,366 test |
| Test 面数 | 77,070 |
| 下游头 | MLP |
| Encoder | 全量解冻 |
| Epochs | 100 |
| Batch size | 256 |
| 优化器 | AdamW，学习率 `3e-4`，weight decay `1e-4` |
| 随机种子 | 42 |
| 模型选择 | 验证集 Macro-F1 最高的 `best.pt` |
| Test 评估 | 相同缓存 test split，batch size 512 |

三组分别在 TITAN RTX GPU 0/1/2 上并行运行。训练及测试过程未发现 Traceback、CUDA OOM、NaN 或 Inf。

## 总体测试结果

| 排名 | 方法 | Best epoch | Val Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | mIoU | Weighted-F1 | Weighted-IoU |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `none` | 84 | **87.34%** | **93.11%** | **88.50%** | **84.59%** | **86.34%** | **77.43%** | **93.04%** | **87.22%** |
| 2 | `per_graph` | 55 | 86.29% | 92.42% | 86.53% | 82.88% | 84.54% | 75.00% | 92.35% | 86.10% |
| 3 | `typewise_global` | 55 | 75.46% | 81.21% | 74.68% | 71.70% | 72.98% | 59.72% | 81.04% | 69.06% |

其中 Val Macro-F1 是选中 checkpoint 时的验证指标，其余均为该 checkpoint 在 test split 上的结果。

### 相对逐图标准化的变化

| 方法 | Accuracy | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|
| `none` | **+0.69 pp** | **+1.80 pp** | **+2.43 pp** | **+0.69 pp** |
| `typewise_global` | -11.21 pp | -11.56 pp | -15.28 pp | -11.31 pp |

`none` 不只是通过多数类提高 Accuracy；Macro-F1 和 mIoU 的增幅更大，说明少数类别的整体表现也有所改善。

## 逐类别测试结果

### F1

| 类别 | Support | `per_graph` | `none` | `typewise_global` | 最优 |
|---|---:|---:|---:|---:|---|
| ExtrudeSide | 39,495 | 95.19% | **95.52%** | 85.94% | `none` |
| ExtrudeEnd | 12,223 | 94.01% | **94.50%** | 80.92% | `none` |
| CutSide | 9,799 | 84.53% | **86.86%** | 60.87% | `none` |
| CutEnd | 1,904 | 76.37% | **79.53%** | 57.94% | `none` |
| Fillet | 7,094 | **91.55%** | 90.82% | 89.00% | `per_graph` |
| Chamfer | 2,181 | 90.24% | **93.17%** | 88.77% | `none` |
| RevolveSide | 4,301 | 89.47% | **90.35%** | 76.23% | `none` |
| RevolveEnd | 73 | 54.96% | **60.00%** | 44.16% | `none` |

`none` 在 8 个类别中的 7 个取得最高 F1，主要收益出现在 CutSide、CutEnd、Chamfer 和 RevolveEnd。`per_graph` 仅在 Fillet 上领先 `0.74` 个百分点。

RevolveEnd 只有 73 个测试面，其指标对少数预测变化非常敏感，不应单独用于总体判断。

### IoU

| 类别 | `per_graph` | `none` | `typewise_global` |
|---|---:|---:|---:|
| ExtrudeSide | 90.83% | **91.43%** | 75.34% |
| ExtrudeEnd | 88.70% | **89.58%** | 67.96% |
| CutSide | 73.21% | **76.78%** | 43.75% |
| CutEnd | 61.77% | **66.02%** | 40.79% |
| Fillet | **84.42%** | 83.18% | 80.18% |
| Chamfer | 82.22% | **87.22%** | 79.81% |
| RevolveSide | 80.94% | **82.39%** | 61.59% |
| RevolveEnd | 37.89% | **42.86%** | 28.33% |

## 分析

### 1. 预训练 total loss 不能预测下游优劣

联合预训练阶段，`typewise_global` 的 total/noise losses 最低，而 `none` 的 total loss 最高；Fusion360Seg 微调结果却完全相反。这验证了前一份报告中的限制：预处理改变了重建目标的尺度与表示空间，不能用预训练 loss 直接选择下游 checkpoint。

### 2. Fusion360Seg 可能从绝对几何尺度中获益

`per_graph` 会分别消除每个零件各通道的均值和标准差，可能同时丢失与建模操作类别有关的绝对尺寸、曲率或位置分布。`none` 保留这些信号，并在 Cut、Chamfer、Revolve 等类别上取得一致收益。

这是一项由结果支持的推断，而不是已隔离验证的因果结论。要确认原因，需要继续做按通道遮蔽或只保留尺度通道的消融。

### 3. 联合全局统计量与 Fusion360Seg 域可能不匹配

`typewise_global` 使用 BRepPreDiff、TMCAD、Fusion360Seg 和 MFCAD++ 联合几何拟合 mean/std。不同数据源的 CAD 单位、坐标原点和曲率分布可能不同，使 Fusion360Seg 特征在统一变换后产生域偏移。此外，法向量保持单位化而其他面通道全局 z-score，改变了各特征组进入投影层时的相对权重。

这是对明显退化的合理解释，但尚未通过 Fusion360Seg-only 统计量实验验证。

### 4. 当前结果是单 seed 结论

`none` 相对 `per_graph` 的 Macro-F1 优势为 1.80 pp，幅度具有实际意义，但本实验只有 seed 42，无法提供均值、标准差或置信区间。若用于论文或默认配置决策，应补充多 seed 实验。

## 建议

1. **当前 Fusion360Seg 模型选择 `none` 的 best checkpoint。**
2. 使用 seeds 43、44 重复 `none` 与 `per_graph`，报告 3-seed 均值和标准差。
3. 暂停使用当前联合统计量版本的 `typewise_global`。
4. 若继续研究全局标准化，优先尝试只用 Fusion360Seg train split 拟合统计量，并分别处理坐标、曲率、尺度与法向量。
5. 对外报告时明确联合预训练使用 transductive 几何协议：下游 val/test 几何参与了无标签预训练。

## 产物

| 方法 | Run directory | Best checkpoint SHA-256 |
|---|---|---|
| `per_graph` | `runs/finetune/20260804-111247_fusion360seg_ablation_per_graph` | `2ffaf7332baa7bbc6e3ef535f88576cddfa00d46e113426a28d4fc0cc486545e` |
| `none` | `runs/finetune/20260804-111248_fusion360seg_ablation_none` | `1d4aaddf63a8c7f6a696eea6149537992976f3d3139ba5afab1b56b7f19a7646` |
| `typewise_global` | `runs/finetune/20260804-111247_fusion360seg_ablation_typewise_global` | `c91d4bfd1b39cd3f7fd23573980ad5566de56441e8c3e404a0c78554388bed64` |

每个 run directory 中均包含：

- `checkpoints/best.pt`：验证 Macro-F1 最优模型。
- `checkpoints/last.pt`：epoch 100 模型。
- `test_metrics.json`：完整测试指标、逐类指标和 confusion matrix。
- `test_evaluate.log`：测试命令输出。

