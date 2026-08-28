# 三组 encoder：五个下游任务 MLP / DiffLoss 对比

## 实验组

| ID | 预训练日期 | 每卡 batch | 4-GPU 全局 batch | LR | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---|
| E1 | 2026-08-20 | 64 | 256 | 1e-3 | `7e4cd232083f6d0bb867edd5389e90f931dc72056f4825b4dd0e5f923cbb5af9` |
| E2 | 2026-08-21 | 32 | 128 | 1e-3 | `c9810f2903ff9389d9def334e3dccad30257193259f60368f5ee610c7bcf6ac3` |
| E3 | 2026-08-24 | 32 | 128 | 5e-4 | `17723791fd8c27055b5ad1f5eac373fb33f7472e9ba6ae3a55c52f3709b08a73` |

注意：8 月 24 日 run 的实际配置是 `train.lr: 0.0005`（5e-4），不是 5e-5。
E1 vs E2 隔离预训练 batch size；E2 vs E3 隔离预训练学习率；E1 vs E3 同时改变
batch size 和学习率，不能解释为单变量消融。

三组下游实验均为 seed 42、200 epochs、batch size 64、梯度累积 4、encoder 全量
微调；分类统一 Mean+Max pooling，DiffLoss 统一 x-start/epsilon 目标及 1.0/0.5
损失权重，按 validation accuracy 选择 best checkpoint 后在 test split 评估。

## Test accuracy

单位为百分比；`Δ31 = E3 - E1`，`Δ32 = E3 - E2`，单位为百分点。

| Task | Head | E1 | E2 | E3 | Δ31 | Δ32 | Best |
|---|---|---:|---:|---:|---:|---:|---|
| BRepPreDiff | MLP | 98.8571 | 98.7502 | **98.8745** | +0.0174 | +0.1243 | E3 |
| BRepPreDiff | DiffLoss | 98.8338 | 98.7590 | **98.8677** | +0.0339 | +0.1087 | E3 |
| Fusion360Seg | MLP | 95.8492 | 96.0322 | **96.0439** | +0.1947 | +0.0117 | E3 |
| Fusion360Seg | DiffLoss | 95.9971 | 95.8596 | **96.0179** | +0.0208 | +0.1583 | E3 |
| MFCAD++ | MLP | 99.4089 | 99.4345 | **99.4420** | +0.0331 | +0.0075 | E3 |
| MFCAD++ | DiffLoss | 99.4059 | **99.4494** | 99.4405 | +0.0346 | -0.0089 | E2 |
| TMCAD | MLP | 82.8887 | 82.7047 | **83.9006** | +1.0119 | +1.1959 | E3 |
| TMCAD | DiffLoss | 83.0727 | 82.6127 | **83.7167** | +0.6440 | +1.1040 | E3 |
| FabWave | MLP | **97.9540** | **97.9540** | **97.9540** | +0.0000 | +0.0000 | Tie |
| FabWave | DiffLoss | **97.9540** | **97.9540** | **97.9540** | +0.0000 | +0.0000 | Tie |

## Test mIoU

| Task | Head | E1 | E2 | E3 | E3 - E1 | E3 - E2 | Best |
|---|---|---:|---:|---:|---:|---:|---|
| BRepPreDiff | MLP | **94.5522** | 93.7644 | 94.4582 | -0.0940 | +0.6938 | E1 |
| BRepPreDiff | DiffLoss | 94.3293 | 94.1108 | **94.3549** | +0.0256 | +0.2441 | E3 |
| Fusion360Seg | MLP | **83.5425** | 82.7660 | 83.1263 | -0.4162 | +0.3603 | E1 |
| Fusion360Seg | DiffLoss | **82.8839** | 82.8296 | 82.6592 | -0.2247 | -0.1704 | E1 |
| MFCAD++ | MLP | 98.1886 | 98.2574 | **98.2892** | +0.1006 | +0.0318 | E3 |
| MFCAD++ | DiffLoss | 98.1384 | **98.2767** | 98.2467 | +0.1083 | -0.0300 | E2 |
| TMCAD | MLP | 71.0397 | 70.8423 | **72.0196** | +0.9799 | +1.1773 | E3 |
| TMCAD | DiffLoss | 70.8140 | 70.3167 | **72.3751** | +1.5611 | +2.0584 | E3 |
| FabWave | MLP | **99.0794** | **99.0794** | **99.0794** | +0.0000 | +0.0000 | Tie |
| FabWave | DiffLoss | **99.0794** | **99.0794** | **99.0794** | +0.0000 | +0.0000 | Tie |

## 汇总与结论

- 十个 task/head 组合按 accuracy：E3 胜 7，E2 胜 1，FabWave 两项三组并列；
  E1 没有单独胜出的组合。
- 五任务平均 accuracy：
  - MLP：E1 94.9916%，E2 94.9751%，E3 **95.2430%**；
  - DiffLoss：E1 95.0527%，E2 94.9269%，E3 **95.1994%**；
  - 两种 head 合并宏平均：E1 95.0221%，E2 94.9510%，E3 **95.2212%**。
- E3 的合并宏平均比 E1 高 0.1990 pp、比同 batch 的 E2 高 0.2702 pp。
- E3 的主要优势来自 TMCAD：相对 E2，MLP +1.1959 pp、DiffLoss +1.1040 pp；
  其余未饱和组合的 accuracy 变化在 -0.0089 至 +0.1583 pp。
- MFCAD++/DiffLoss 是唯一由 E2 获得最高 accuracy 的组合，领先 E3 0.0089 pp。
- FabWave 三组均已饱和，无法用于区分 encoder。

在本次单 seed 结果中，E3（batch 32、LR 5e-4）是三组里总体最好的 encoder，且明显
改善 TMCAD。但其总体优势高度受 TMCAD 驱动；若要判断 5e-4 学习率是否稳定优于
1e-3，应对 E2/E3 至少补 3 seeds，并报告均值与标准差。

## 明细报告

- E1：`reports/seven_source_711_all_downstreams_20260820-204011.md`
- E2：`reports/batch32_encoder_all_downstreams_20260825-1127.md`
- E3：`reports/lr5e4_encoder_all_downstreams_20260826-1150.md`

