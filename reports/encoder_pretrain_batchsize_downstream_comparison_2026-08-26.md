# Encoder 预训练 batch size：五个下游任务 MLP / DiffLoss 对照

## 对照对象

| Encoder | 预训练 run | 每卡 batch | 4-GPU 全局 batch | Epochs | LR | Encoder checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---|
| BS64 | `20260820-163406_new_occ_seven_source_edge_update_20260820-151913` | 64 | 256 | 150 | 1e-3 | `7e4cd232083f6d0bb867edd5389e90f931dc72056f4825b4dd0e5f923cbb5af9` |
| BS32 | `20260821-200250_new_occ_seven_source_edge_update_batch32_wandb_resume_e035_20260821-2003` | 32 | 128 | 150 | 1e-3 | `c9810f2903ff9389d9def334e3dccad30257193259f60368f5ee610c7bcf6ac3` |

两次预训练均为七源数据、711/63 OCC 特征、`edge_update_attention`、4 layers、
hidden dim 128、4 heads、seed 42、AdamW、相同损失权重和 150 epochs。BS32 训练从
epoch 35 checkpoint 续训至 epoch 150；续训不改变其有效训练配置。日志中的
`batch_size_per_rank` 分别为 64 和 32，`world_size=4`。

## 下游控制变量

两侧共 20 个结果使用完全一致的下游有效配置；逐项比较十对 `config.yaml` 后，除
`run.name`、`run.output_dir` 和 `train.pretrain_checkpoint` 外没有配置差异：

- seed 42，dataloader seed 42；
- 200 epochs，下游 batch size 64，梯度累积 4；
- AdamW，lr 3e-4，weight decay 1e-4；
- encoder 全量微调；
- 分类任务统一 `graph_pooling: mean_max`；
- DiffLoss 统一 `x_start_epsilon`，x-start/epsilon 权重 1.0/0.5；
- 按 validation accuracy 选择 best checkpoint，再在 test split 上评估。

## Test 结果

`Δ = BS32 - BS64`，单位均为百分点。Accuracy 是 checkpoint 选择和主比较指标；
mIoU 同时列出以观察分割质量。

| Task | Head | BS64 Acc | BS32 Acc | ΔAcc | BS64 mIoU | BS32 mIoU | ΔmIoU | Acc winner |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BRepPreDiff | MLP | 98.8571 | 98.7502 | -0.1069 | 94.5522 | 93.7644 | -0.7878 | BS64 |
| BRepPreDiff | DiffLoss | 98.8338 | 98.7590 | -0.0748 | 94.3293 | 94.1108 | -0.2185 | BS64 |
| Fusion360Seg | MLP | 95.8492 | 96.0322 | +0.1830 | 83.5425 | 82.7660 | -0.7765 | BS32 |
| Fusion360Seg | DiffLoss | 95.9971 | 95.8596 | -0.1375 | 82.8839 | 82.8296 | -0.0543 | BS64 |
| MFCAD++ | MLP | 99.4089 | 99.4345 | +0.0256 | 98.1886 | 98.2574 | +0.0688 | BS32 |
| MFCAD++ | DiffLoss | 99.4059 | 99.4494 | +0.0435 | 98.1384 | 98.2767 | +0.1383 | BS32 |
| TMCAD | MLP | 82.8887 | 82.7047 | -0.1840 | 71.0397 | 70.8423 | -0.1974 | BS64 |
| TMCAD | DiffLoss | 83.0727 | 82.6127 | -0.4600 | 70.8140 | 70.3167 | -0.4973 | BS64 |
| FabWave | MLP | 97.9540 | 97.9540 | +0.0000 | 99.0794 | 99.0794 | +0.0000 | Tie |
| FabWave | DiffLoss | 97.9540 | 97.9540 | +0.0000 | 99.0794 | 99.0794 | +0.0000 | Tie |

## 汇总

- 十个 task/head 组合按 accuracy 计：BS64 胜 5，BS32 胜 3，平 2。
- 五任务平均 accuracy：MLP 为 BS64 94.9916%、BS32 94.9751%（BS32 -0.0165 pp）；
  DiffLoss 为 BS64 95.0527%、BS32 94.9269%（BS32 -0.1258 pp）。
- 十个组合的宏平均 accuracy：BS64 95.0221%、BS32 94.9510%，BS64 高 0.0711 pp。
- BS32 的明确收益集中在 Fusion360Seg/MLP（+0.1830 pp）以及 MFCAD++ 两个 head
  （+0.0256 / +0.0435 pp）；BS64 在 TMCAD 上更稳，尤其 DiffLoss 高 0.4600 pp。
- FabWave 已饱和，两种 encoder、两种 head 的 accuracy 都是 97.9540%，无法区分。
- 每个任务中 accuracy 最优组合：BRepPreDiff=BS64/MLP 98.8571%，
  Fusion360Seg=BS32/MLP 96.0322%，MFCAD++=BS32/DiffLoss 99.4494%，
  TMCAD=BS64/DiffLoss 83.0727%，FabWave 四者并列 97.9540%。

结论：在本次单 seed、严格匹配的下游协议下，减小 encoder 预训练 batch size 没有带来
一致收益；BS64 的总体 accuracy 略高，但差距很小且具有明显的任务/head 依赖性。
如果要把 0.0711 pp 的宏平均差异解释为稳定优势，需要至少补 3 seeds，而不能仅凭本次
单 seed 结果下普遍性结论。

## 明细报告

- BS64：`reports/seven_source_711_all_downstreams_20260820-204011.md`
- BS32：`reports/batch32_encoder_all_downstreams_20260825-1127.md`

