# Fusion 360 Segmentation：Blendit MLP full 微调结果

生成日期：2026-07-23

## 结论

Blendit 的 MLP full 基模已在 Fusion 360 Segmentation s2.0.0 的 8 类 face
segmentation 任务上完成 100 epochs 微调。最佳 checkpoint 由 validation Macro-F1
选择，对完整且未参与训练/选模的官方 test split 进行单卡精确评估后，结果如下。

| Split | Models | Faces | Accuracy | Macro Precision | Macro Recall | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 5,350 | 79,920 | 0.924299 | 0.882783 | 0.836808 | 0.856431 | 0.762431 | 0.923753 |
| Test | 5,366 | 77,070 | **0.921318** | **0.884647** | **0.834971** | **0.856523** | **0.761837** | **0.920709** |

最佳 checkpoint 位于 epoch 84；训练期间记录的 DDP validation Macro-F1 为
0.856397，单卡去重后的精确 validation Macro-F1 为 0.856431。

## 数据集与 70/15/15 划分

- 数据集根目录：`/data/hhfeng/fusion360seg/s2.0.0`
- 官方 split 文件：`train_test.json`
- 官方划分：30,314 train / 5,366 test，共 35,680 个模型
- STEP 与 SEG 配对检查：35,680 / 35,680，缺失数为 0
- SEG 标签值域：`0..7`
- 全数据集 face 数：526,069
- 随机种子：42

划分时完整保留官方 test split，只对官方 train 列表使用
`random.Random(42).shuffle`。然后按 `70 / 85` 的比例取 24,964 个模型作为新
train，其余 5,350 个模型作为 validation。最终比例因官方 split 的整数取整而略有
偏差：

| Split | Models | 实际占比 | Faces | Split 文件 |
|---|---:|---:|---:|---|
| Train | 24,964 | 69.966% | 369,079 | `data/splits/fusion360seg_train.txt` |
| Validation | 5,350 | 14.994% | 79,920 | `data/splits/fusion360seg_val.txt` |
| Test | 5,366 | 15.039% | 77,070 | `data/splits/fusion360seg_test.txt` |
| 合计 | 35,680 | 100.000% | 526,069 | — |

三个 split 的样本 ID 两两交集均为 0，合集与官方数据全集一致。文件 SHA-256：

| Split | SHA-256 |
|---|---|
| Train | `62a8f6620bb77676b3951c94dd5925b16cba61de4215534fe762ce6b249247f4` |
| Validation | `3788a59cac5a39804df76dc49b7c31021ab3f5a5b87846ea42b9412ef9fdb81c` |
| Test | `6213e66cdbc32f668bd276764b39b0f4589324732e6ec3ca5524f3a65f117711` |

### Face 标签分布

| ID | 类别 | Train | Validation | Test |
|---:|---|---:|---:|---:|
| 0 | ExtrudeSide | 186,250 (50.464%) | 41,202 (51.554%) | 39,495 (51.246%) |
| 1 | ExtrudeEnd | 58,052 (15.729%) | 12,662 (15.843%) | 12,223 (15.860%) |
| 2 | CutSide | 47,464 (12.860%) | 10,291 (12.877%) | 9,799 (12.714%) |
| 3 | CutEnd | 8,535 (2.313%) | 1,876 (2.347%) | 1,904 (2.471%) |
| 4 | Fillet | 37,878 (10.263%) | 7,777 (9.731%) | 7,094 (9.205%) |
| 5 | Chamfer | 11,148 (3.021%) | 2,149 (2.689%) | 2,181 (2.830%) |
| 6 | RevolveSide | 19,456 (5.272%) | 3,897 (4.876%) | 4,301 (5.581%) |
| 7 | RevolveEnd | 296 (0.080%) | 66 (0.083%) | 73 (0.095%) |

## 训练协议

“MLP full” 在本次实验中表示 `finetune_head: mlp` 且
`encoder_freeze_mode: none`，即 MLP segmentation head 和完整 encoder 联合更新。

| 配置项 | 值 |
|---|---|
| 初始化权重 | Blendit 150-epoch pretrain `last.pt` |
| 载入情况 | 61 个预训练张量载入；8 类 MLP head 的 2 个不匹配张量跳过并重新初始化 |
| 可训练参数 | 832,136（全部可训练） |
| Hidden dimension / layers | 128 / 4 |
| Dropout | 0.1 |
| 输出类别数 | 8 |
| Loss | Cross-entropy + 0.3 × Dice |
| Optimizer | AdamW |
| Learning rate / weight decay | `3e-4` / `1e-4` |
| Gradient clipping | 1.0 |
| Epochs | 100 |
| GPU | 4 × NVIDIA TITAN RTX 24 GB |
| Batch size | 256/GPU，global batch 1,024 |
| Validation | 每个 epoch |
| 最佳权重选择 | Validation Macro-F1 最大 |
| 训练耗时 | 约 8 分 56 秒（16:49:00–16:57:56） |

B-Rep 提取使用 `uv_grid_size: 10`。完整缓存包含 35,680 个有效 NPZ，
约 355 MB；全量有限值和 split 完整性检查未发现无效缓存。

## Test 逐类结果

| ID | 类别 | Support | Precision | Recall | F1 | IoU |
|---:|---|---:|---:|---:|---:|---:|
| 0 | ExtrudeSide | 39,495 | 0.940192 | 0.956855 | 0.948450 | 0.901955 |
| 1 | ExtrudeEnd | 12,223 | 0.933183 | 0.947231 | 0.940154 | 0.887067 |
| 2 | CutSide | 9,799 | 0.843123 | 0.823247 | 0.833067 | 0.713894 |
| 3 | CutEnd | 1,904 | 0.839088 | 0.714811 | 0.771980 | 0.628637 |
| 4 | Fillet | 7,094 | 0.934805 | 0.889343 | 0.911508 | 0.837404 |
| 5 | Chamfer | 2,181 | 0.929702 | 0.915635 | 0.922615 | 0.856346 |
| 6 | RevolveSide | 4,301 | 0.892379 | 0.898396 | 0.895377 | 0.810573 |
| 7 | RevolveEnd | 73 | 0.764706 | 0.534247 | 0.629032 | 0.458824 |

`RevolveEnd` 仅占 test faces 的 0.095%，其 73 个 face 中正确预测 39 个，是
Macro-F1 的主要限制项。数量较多的错误还包括 CutSide → ExtrudeSide（1,349）、
ExtrudeSide → CutSide（1,051）、Fillet → ExtrudeSide（536）和
CutEnd → ExtrudeEnd（389）。

## Test 混淆矩阵

行是真值，列是预测。

| GT \ Pred | ExtrudeSide | ExtrudeEnd | CutSide | CutEnd | Fillet | Chamfer | RevolveSide | RevolveEnd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExtrudeSide | 37,791 | 201 | 1,051 | 28 | 202 | 89 | 130 | 3 |
| ExtrudeEnd | 282 | 11,578 | 114 | 152 | 3 | 5 | 87 | 2 |
| CutSide | 1,349 | 130 | 8,067 | 37 | 116 | 26 | 68 | 6 |
| CutEnd | 39 | 389 | 83 | 1,361 | 0 | 0 | 32 | 0 |
| Fillet | 536 | 3 | 141 | 0 | 6,309 | 3 | 102 | 0 |
| Chamfer | 94 | 17 | 22 | 5 | 11 | 1,997 | 34 | 1 |
| RevolveSide | 102 | 77 | 83 | 39 | 108 | 28 | 3,864 | 0 |
| RevolveEnd | 2 | 12 | 7 | 0 | 0 | 0 | 13 | 39 |

## 产物

- 数据配置：`data/fusion360seg.yaml`
- 训练配置：`configs/finetune_fusion360seg_mlp_full.yaml`
- 运行目录：`runs/finetune/20260723-164805_fusion360seg_mlp_full`
- 最佳权重：`runs/finetune/20260723-164805_fusion360seg_mlp_full/checkpoints/best.pt`
- 最后权重：`runs/finetune/20260723-164805_fusion360seg_mlp_full/checkpoints/last.pt`
- 精确 validation 指标：`runs/finetune/20260723-164805_fusion360seg_mlp_full/val_metrics.json`
- 精确 test 指标：`runs/finetune/20260723-164805_fusion360seg_mlp_full/test_metrics.json`
- 训练日志：`runs/finetune/20260723-164805_fusion360seg_mlp_full/logs/finetune.log`

最佳权重大小为 10,063,753 bytes，SHA-256 为
`1b583bc22832206c72d11841351c8a9fdbef60ad1702fd0eddfc778d533ed9bb`。

## 复现命令

```bash
# 完整缓存构建与检查
/home/hhfeng/miniconda3/envs/blendit/bin/blendit-prepare-data \
  --config data/fusion360seg.yaml \
  --workers 16

# 4-GPU MLP full finetune
/home/hhfeng/miniconda3/envs/blendit/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m blendit.training.finetune \
  --config configs/finetune_fusion360seg_mlp_full.yaml

# 用最佳 checkpoint 精确评估官方 test split
/home/hhfeng/miniconda3/envs/blendit/bin/python \
  -m blendit.training.evaluate \
  --checkpoint runs/finetune/20260723-164805_fusion360seg_mlp_full/checkpoints/best.pt \
  --split test \
  --output runs/finetune/20260723-164805_fusion360seg_mlp_full/test_metrics.json \
  --batch-size 512 \
  --num-workers 16 \
  --device cuda
```
