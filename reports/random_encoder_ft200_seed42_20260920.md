# Random encoder 初始化后全量微调

GPU 4 串行运行 Fusion360Seg、Blend、TMCAD；随机初始化完整编码器及 MLP head，不加载预训练权重，encoder_freeze_mode=none。
复用先前迁移消融的源码快照与下游协议：seed 42、200 epochs、完整训练划分、50% SO(3)、相同 batch/累积、优化器、损失和 validation accuracy 选优。TMCAD 使用 mean_max。初次训练 resume=null，重启仅恢复本实验自己的 checkpoint。
与先前 LODO 微调的实质区别只有预训练初始化；保存的 LODO 配置中 checkpoint 为 null，实际运行时由 override 注入。测试使用 validation 最佳 checkpoint。

| 数据集 | 状态 | 随机初始化 Acc % | Macro-F1 % | mIoU % | 完整预训练 Acc % | LODO Acc % | 完整预训练−随机 pp | LODO−随机 pp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fusion360seg | completed | 96.8937 | 91.8052 | 85.9316 | 96.9275 | 97.0170 | +0.0337 | +0.1233 |
| blend | completed | 98.8697 | 97.1533 | 94.5203 | 98.9901 | 98.9969 | +0.1204 | +0.1272 |
| tmcad | completed | 85.2806 | 84.9558 | 74.5256 | 86.3845 | 85.9246 | +1.1040 | +0.6440 |

仅单 seed 对照，不将百分点差异视为统计显著性。各任务状态、日志、配置及源码哈希见 `runs/random_encoder_ft200_seed42_20260920/`。
