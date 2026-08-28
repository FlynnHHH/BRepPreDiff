# 8.24 encoder：预训练 / 微调 epoch 二维消融设计

## 目的

使用 2026-08-24 的 seven-source encoder 训练轨迹，量化预训练时长与下游微调时长的交互：

- encoder run：`runs/pretrain/20260824-114039_seven_source_711_4gpu_wandb_20260824-113934`
- 预训练条件：4 GPU、batch size 32/GPU、global batch size 128、learning rate `5e-4`
- 预训练 epoch：20、50、100、150
- 微调 epoch：10、20、50、100、150、200
- 每个下游任务形成 `4 × 6 = 24` 个结果点

首轮固定 MLP head，覆盖 BRepPreDiff、Fusion360Seg、MFCAD++、TMCAD 和 FabWave。这样只研究
encoder 预训练时长和微调预算，不把 MLP / DiffLoss 的差异混入主效应。TMCAD 和 FabWave
统一使用 `model.graph_pooling: mean_max`。

## 控制变量

所有单元格固定同一数据划分、缓存、seed 42、dataloader seed 42、encoder 结构、优化器、
微调 learning rate、有效 batch size 256、权重衰减和 evaluation protocol。为支持单卡双任务，
TMCAD 使用 micro-batch 16、梯度累积 16；其余任务使用 micro-batch 64、梯度累积 4。
下游分割项目默认预算仍为 200 epoch；10/20/50/100/150 是本实验明确指定的预算消融。

微调训练代码没有随 `train.epochs` 改变的 scheduler，因此每个预训练 checkpoint 只需训练一条
200-epoch 轨迹，并在指定 epoch 读取快照。这和使用相同随机种子分别运行 10、20、50、100、
150、200 epoch 等价，将每个任务的训练次数从 24 次降到 4 次。测试集在全部 24 个固定网格点
上只用于事后报告，不据此选择超参数；最终配置应由 validation 指标决定后再报告一次 test 指标。

## 指标与分析

- 主指标：accuracy；分割同时报告 macro IoU，分类同时报告 macro F1。
- 每个数据集输出一个 4 行 × 6 列 heatmap/table。
- 分别计算固定微调预算下增加预训练 epoch 的边际收益，以及固定预训练 checkpoint 下增加微调
  epoch 的边际收益。
- 推荐同时报告最佳 validation 单元格、200-epoch 默认预算单元格，以及从 150 降到 50 微调
  epoch 的性能/计算折中。

## 执行

```bash
bash scripts/run_pretrain_finetune_epoch_grid.sh
```

可用 `TASKS=tmcad_cls` 只运行单个任务；`GPU_IDS`、`RUN_TAG`、`RUN_ROOT`、`PYTHON_BIN` 均可通过
环境变量覆盖。脚本默认使用 4 张 GPU、每卡两条任务队列，并自动生成逐点 JSON、汇总 CSV 和
Markdown 表格。
