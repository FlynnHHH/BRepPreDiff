# 全量预训练 Linear probe / Leave-one-dataset-out

GPU 2：完整 Inductive9 checkpoint 冻结编码器的纯线性 probe。GPU 1：Fusion360Seg/Blend；GPU 0：TMCAD。各目标独立排除后预训练 100 epochs，再全量微调 200 epochs。下游完整训练划分，seed 42，validation-best 测试。

设计与证据边界见 [实验设计](lodo_transfer_design_20260918.md)。未完成指标留空，不把验证集结果当作测试结果。

| Dataset | Arm | Status | Test Accuracy % | Macro-F1 % | mIoU % | ΔAcc vs full FT (pp) |
|---|---|---|---:|---:|---:|---:|
| fusion360seg | Full pretrain + MLP FT | completed | 96.9275 | 92.2254 | 86.6362 | +0.0000 |
| fusion360seg | Full pretrain + Linear probe | completed | 71.9476 | 48.7656 | 36.2535 | -24.9799 |
| fusion360seg | LODO pretrain + MLP FT | completed | 97.0170 | 91.6537 | 85.9780 | +0.0895 |
| blend | Full pretrain + MLP FT | completed | 98.9901 | 97.3687 | 94.9185 | +0.0000 |
| blend | Full pretrain + Linear probe | completed | 94.5077 | 84.7958 | 74.7332 | -4.4824 |
| blend | LODO pretrain + MLP FT | completed | 98.9969 | 97.4312 | 95.0355 | +0.0068 |
| tmcad | Full pretrain + MLP FT | completed | 86.3845 | 85.8850 | 75.9397 | +0.0000 |
| tmcad | Full pretrain + Linear probe | completed | 69.6412 | 68.5854 | 53.9198 | -16.7433 |
| tmcad | LODO pretrain + MLP FT | completed | 85.9246 | 85.4620 | 75.2434 | -0.4600 |

预训练状态：
- fusion360seg: completed
- blend: completed
- tmcad: completed

probe 与 FT 的差值同时受冻结、head 和分类 pooling 实现差异影响；不能解释为冻结编码器的单独因果效应。单 seed 结果不代表统计显著性。
