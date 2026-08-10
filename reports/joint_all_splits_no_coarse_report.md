# 四数据源联合无 coarse label 自监督预训练与微调报告

生成日期：2026-07-24

## 结论

四数据源联合无 coarse-label 自监督预训练已完成 150 epochs，随后完成四项
100-epoch MLP full 微调和完整 test split 精确评估。

| 数据集 | Task | Accuracy | Macro-F1 | Macro-IoU | Weighted-IoU |
|---|---|---:|---:|---:|---:|
| Blendit | Seg | **0.981403** | **0.963957** | **0.931435** | **0.964574** |
| TMCAD | Cls | **0.811408** | **0.803404** | **0.681889** | **0.690816** |
| Fusion360Seg | Seg | **0.923628** | **0.850743** | **0.755406** | **0.860097** |
| MFCAD++ | Seg | **0.991040** | **0.985739** | **0.972147** | **0.982424** |

## 实验协议

本实验按指定要求采用 **transductive self-supervised pretraining**：

- Blendit pretrain 使用其 train split；
- TMCAD、Fusion360Seg、MFCAD++ 的 train、validation、test 三个 split 全部参与
  自监督预训练；
- 下游标签不会从 NPZ cache 中加载，联合 batch 的 `labels` 恒为 `None`；
- 预训练模型不实例化 coarse-label head，
  `coarse_label_loss_weight: 0.0`；
- 自监督目标仅包含连续几何噪声/重建、surface type、edge type 和 topology
  relation；
- 四个来源均使用 `uv_grid_size: 10`，encoder 输入维度一致。

由于下游 test 模型的无标签几何参与了预训练，最终指标不能与 test 几何完全不可见
的 inductive protocol 直接比较。微调和 checkpoint 选择本身仍只使用各下游任务的
train/validation split，test 标签只在最终评估时使用。

## 联合预训练数据

| 数据集 | 来源 split | 图数量 |
|---|---|---:|
| Blendit pretrain | train | 172,287 |
| TMCAD | train | 8,709 |
| TMCAD | validation | 1,090 |
| TMCAD | test | 1,087 |
| Fusion360Seg | train | 24,964 |
| Fusion360Seg | validation | 5,350 |
| Fusion360Seg | test | 5,366 |
| MFCAD++ | train | 41,766 |
| MFCAD++ | validation | 8,950 |
| MFCAD++ | test | 8,949 |
| **合计** | — | **278,518** |

TMCAD 的计数是 10,886 个 OCC 可解析模型；原始数据中 11 个无效 STEP 已由现有
clean split 排除。其余三个来源的 split 与现有完整 cache 一一对应。

## 配置

### 自监督预训练

| 配置项 | 值 |
|---|---|
| Hidden dimension / message-passing layers | 128 / 4 |
| Diffusion timesteps | 1,000 |
| Epochs | 150 |
| Batch size | 512 / GPU |
| Optimizer | AdamW |
| Learning rate / weight decay | `1e-3` / `1e-4` |
| Gradient clipping | 1.0 |
| Coarse-label head | 禁用 |
| Coarse-label loss | 0.0 |
| GPU / global batch | 4 × TITAN RTX 24 GB / 2,048 graphs |
| 运行时间 | 约 1 小时 39 分 |
| 最终 total loss | 1.33085 |
| 数据配置 | `data/pretrain_joint_all_splits.yaml` |
| 训练配置 | `configs/pretrain_joint_all_splits_no_coarse.yaml` |
| 运行目录 | `runs/pretrain/20260723-213828_joint_all_splits_no_coarse` |

### 下游微调

四项任务均使用 MLP head、完整 encoder 联合更新（`encoder_freeze_mode: none`）、
100 epochs、AdamW、学习率 `3e-4`、weight decay `1e-4`，并以 validation
Macro-F1 选择 `best.pt`。四项任务分别使用一张 TITAN RTX 并行训练，每卡 batch
size 为 256。

| 数据集 | 任务 | 类别数 | Train / Val / Test | 配置 |
|---|---|---:|---:|---|
| Blendit | face segmentation | 3 | 6,127 / 766 / 766 | `configs/finetune_joint_blendit_mlp.yaml` |
| TMCAD | model classification | 10 | 8,709 / 1,090 / 1,087 | `configs/finetune_joint_tmcad_mlp.yaml` |
| Fusion360Seg | face segmentation | 8 | 24,964 / 5,350 / 5,366 | `configs/finetune_joint_fusion360seg_mlp.yaml` |
| MFCAD++ | face segmentation | 25 | 41,766 / 8,950 / 8,949 | `configs/finetune_joint_mfcadpp_mlp.yaml` |

## 实现与验证

- 联合 Dataset 实际解析得到 278,518 个图，10 个来源分量计数与上表一致。
- 从 Blendit、TMCAD、Fusion360Seg、MFCAD++ test 各抽一个图组成真实混合 batch：
  156 faces、748 edges。
- 混合 batch 的 `labels is None`。
- `model.coarse_label_head is None`。
- loss 仅含 `face_noise`、`face_recon`、`surface`、`edge_noise`、`edge_recon`、
  `edge_type`、`relation` 和 `total`，无 coarse-label 项。
- CPU 前向、反向和梯度计算成功，检查 batch 的总 loss 为 `6.772156`。
- 完整测试结果：`89 passed`。

## 下游结果

| 数据集 | Best epoch | Test samples | Test faces | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Macro-IoU | Weighted-F1 | Weighted-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Blendit | 21 | 766 | 41,459 | 0.981403 | 0.957206 | 0.971104 | 0.963957 | 0.931435 | 0.981600 | 0.964574 |
| TMCAD | 94 | 1,087 | — | 0.811408 | 0.806434 | 0.807725 | 0.803404 | 0.681889 | 0.809230 | 0.690816 |
| Fusion360Seg | 60 | 5,366 | 77,070 | 0.923628 | 0.861352 | 0.841178 | 0.850743 | 0.755406 | 0.923119 | 0.860097 |
| MFCAD++ | 98 | 8,949 | 268,982 | 0.991040 | 0.986831 | 0.984696 | 0.985739 | 0.972147 | 0.991031 | 0.982424 |

### 主要困难类别

| 数据集 | 类别 | Support | F1 | IoU |
|---|---|---:|---:|---:|
| Blendit | EBF | 5,531 | 0.932558 | 0.873637 |
| TMCAD | coupling | 107 | 0.641304 | 0.472000 |
| TMCAD | pulley | 101 | 0.666667 | 0.500000 |
| Fusion360Seg | RevolveEnd | 73 | 0.583333 | 0.411765 |
| Fusion360Seg | CutEnd | 1,904 | 0.775453 | 0.633257 |
| MFCAD++ | Rectangular through slot | 4,097 | 0.957636 | 0.918716 |

Fusion360Seg 的 `RevolveEnd` 只有 73 个 test faces，仍是 Macro-F1 的主要限制项。
TMCAD 的 coupling、pulley 和 shaft 是最弱的三个模型类别。MFCAD++ 的最低类别
F1 仍达到 0.957636。

## 与仓库既有结果的参考比较

下表只用于提供量级参考，不是严格受控消融。既有 Blendit 行来自无 coarse-label
MLP 实验；其余既有结果使用原 Blendit-only、带 coarse-label 的预训练 encoder。
本次实验还让 test 几何参与了无标签预训练，因此不能把差异完全归因于联合数据。

| 数据集 | Δ Accuracy | Δ Macro-F1 | Δ Macro-IoU |
|---|---:|---:|---:|
| Blendit | +0.169 pp | +0.410 pp | +0.744 pp |
| TMCAD | -0.736 pp | -1.002 pp | -1.238 pp |
| Fusion360Seg | +0.231 pp | -0.578 pp | -0.643 pp |
| MFCAD++ | +0.138 pp | +0.198 pp | +0.371 pp |

联合无 coarse-label 预训练对 Blendit 和 MFCAD++ 的三项聚合指标均有提升；对
Fusion360Seg 的 Accuracy 有小幅提升，但 Macro-F1/IoU 略降；TMCAD 三项指标均
略降。由于协议同时改变了数据构成、coarse head 和 test 几何可见性，这里只报告
观察结果，不作单因素因果结论。

## 产物与校验

| 产物 | 路径 | SHA-256 |
|---|---|---|
| 联合预训练 `last.pt` | `runs/pretrain/20260723-213828_joint_all_splits_no_coarse/checkpoints/last.pt` | `1689ab00fe1d9e110b6a38f052a773a241b2765dd7c5addee1cef28fc8e3d464` |
| Blendit `best.pt` | `runs/finetune/20260723-232008_joint_blendit_mlp/checkpoints/best.pt` | `f1a234cbe402eb788c8eaafd84e1c641f6d755dabd325a6864c99a3b3f5c4bb3` |
| TMCAD `best.pt` | `runs/finetune/20260723-232049_joint_tmcad_mlp/checkpoints/best.pt` | `02adcfb2db6451bb0f30af65ad674a3a4f65248309dc1b2ceca4f42d524a4c01` |
| Fusion360Seg `best.pt` | `runs/finetune/20260723-232129_joint_fusion360seg_mlp/checkpoints/best.pt` | `7a748af4f499a5d2a92082c820f83887c80af29eb9dcbb92bee952b6492c35b6` |
| MFCAD++ `best.pt` | `runs/finetune/20260723-232210_joint_mfcadpp_mlp/checkpoints/best.pt` | `d3f9fa1b512cc9cba78e070176993cfbe077175f0383b9389e1d64da2132bb21` |

完整 confusion matrix 和逐类 precision/recall/F1/IoU 位于：

- `runs/finetune/20260723-232008_joint_blendit_mlp/test_metrics.json`
- `runs/finetune/20260723-232049_joint_tmcad_mlp/test_metrics.json`
- `runs/finetune/20260723-232129_joint_fusion360seg_mlp/test_metrics.json`
- `runs/finetune/20260723-232210_joint_mfcadpp_mlp/test_metrics.json`

## 复现命令

四卡联合预训练：

```bash
/home/hhfeng/miniconda3/envs/blendit/bin/torchrun \
  --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config configs/pretrain_joint_all_splits_no_coarse.yaml
```

```bash
PRETRAIN=runs/pretrain/20260723-213828_joint_all_splits_no_coarse/checkpoints/last.pt
```

四项微调可分别在四张卡上并行执行；下面是一项命令示例，替换 GPU ID 和 config
即可复现其余任务：

```bash
CUDA_VISIBLE_DEVICES=0 /home/hhfeng/miniconda3/envs/blendit/bin/python \
  -m blendit.training.finetune \
  --config configs/finetune_joint_blendit_mlp.yaml \
  --override train.pretrain_checkpoint="$PRETRAIN"
```

每项训练结束后，对相应 `best.pt` 做单进程精确 test 评估：

```bash
/home/hhfeng/miniconda3/envs/blendit/bin/python \
  -m blendit.training.evaluate \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --split test \
  --output runs/finetune/<run>/test_metrics.json \
  --batch-size 512 \
  --num-workers 16 \
  --device cuda
```
