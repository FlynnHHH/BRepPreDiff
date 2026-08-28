# 全量 Fusion Gallery 无标签联合预训练与 DiffLoss Benchmark 报告

生成日期：2026-08-05（Asia/Shanghai）

## 1. 结论

修正后的后台流水线已全部完成：全量 STEP 数据准备、4 × Titan RTX 联合预训练、
四项 DiffLoss 微调以及四个 test split 的正式评估均成功结束。

本次对 334,080 个候选 STEP 全部进行了特征提取，最终纳入 334,036 个；仅过滤 44 个
无法生成有限 B-Rep 特征的文件。Fusion360Rec 实际纳入 27,941 / 27,958，
Fusion360Ass 实际纳入 23,028 / 23,029，不再发生任务 JSON 被误当成分割标签的问题。

## 2. 实际预训练语料

采用 transductive 无标签协议：配置中的 train/val/test geometry 全部并入预训练 train
split，但训练时不加载任何下游标签。

| 来源 | 候选 STEP | 实际纳入 | 过滤 | 实际 split 组成 |
|---|---:|---:|---:|---|
| Blendit | 172,287 | 172,287 | 0 | 172,287 train |
| TMCAD | 10,886 | 10,886 | 0 | 8,709 / 1,090 / 1,087 |
| Fusion360Seg s2.0.1 | 35,680 | 35,680 | 0 | 24,964 / 5,350 / 5,366 |
| MFCAD++ | 59,665 | 59,665 | 0 | 41,766 / 8,950 / 8,949 |
| Fusion360Rec r1.0.1 | 27,958 | 27,941 | 17 | 22,354 / 2,793 / 2,794 |
| Fusion360Ass j1.0.0 | 23,029 | 23,028 | 1 | 18,422 / 2,303 / 2,303 |
| FabWave | 4,575 | 4,549 | 26 | 3,639 / 463 / 447 |
| **总计** | **334,080** | **334,036** | **44** | — |

过滤原因：

- Fusion360Rec：17 个 STEP 触发 OCC `TopExp::MapShapes` 类型错误。
- Fusion360Ass：1 个 STEP 的 `face_cont` 含 1,812 个 NaN/Inf 值。
- FabWave：沿用已验证 clean split，26 个 Webbing Guide STEP 无法由 OCC 安全解析。

## 3. 联合预训练

- Encoder：初始 Blendit 4-layer MLP/FFN GraphMessageLayer。
- Hidden dimension：128；dropout：0.1；`uv_grid_size: 10`。
- Geometry diffusion timesteps：1,000。
- Optimizer：AdamW；学习率 `1e-3`；weight decay `1e-4`。
- Epochs：150；每卡 batch size 512；全局 batch size 2,048。
- DDP：4 × NVIDIA Titan RTX；每卡每 epoch 164 batches。
- 训练时间：2026-08-05 11:32–13:22，约 1 小时 50 分钟。

| Epoch | Total loss | Face noise | Edge noise | Face recon | Edge recon |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.06506 | 0.97704 | 0.31111 | 0.59069 | 0.57772 |
| 150 | 1.33056 | 0.85518 | 0.10054 | 0.37914 | 0.37018 |

预训练 checkpoint：

- `runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt`
- SHA-256：`f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9`

## 4. 微调协议与最佳验证结果

四个 benchmark 均从上述 `last.pt` 初始化，完整更新 encoder，使用 DiffLoss head，训练
200 epochs，并按 validation macro-F1 保存 `best.pt`。四项任务各占一张 Titan RTX
并行运行。微调数据及 split 保持不变；其中 Fusion360Seg 微调仍使用 s2.0.0。

| Benchmark | 任务 | 类别数 | Train / Val / Test | Best epoch | Val Accuracy | Val Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| MFCAD++ | 面分割 | 25 | 41,766 / 8,950 / 8,949 | 193 | 99.175% | 98.736% |
| TMCAD | 模型分类 | 10 | 8,709 / 1,090 / 1,087 | 126 | 83.670% | 83.329% |
| Fusion360Seg s2.0.0 | 面分割 | 8 | 24,964 / 5,350 / 5,366 | 175 | 92.965% | 85.769% |
| FabWave | 模型分类 | 44 | 3,639 / 463 / 447 | 159 | 86.609% | 90.637% |

## 5. 四个 Benchmark 测试结果

所有测试结果均使用 validation macro-F1 最优的 `best.pt`，不是 epoch 200 的
`last.pt`。

| Benchmark | Test 规模 | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCAD++ | 8,949 models / 268,982 faces | **99.088%** | 98.605% | 98.521% | **98.561%** | **97.194%** | 99.087% |
| TMCAD | 1,087 models | **84.269%** | 84.146% | 83.857% | **83.968%** | **73.282%** | 84.344% |
| Fusion360Seg s2.0.0 | 5,366 models / 77,070 faces | **92.735%** | 89.127% | 82.608% | **85.052%** | **75.827%** | 92.689% |
| FabWave | 447 models | **85.906%** | 91.436% | 91.536% | **91.392%** | **89.938%** | 86.073% |

完整 confusion matrix 与逐类 precision、recall、F1、IoU 位于各 run directory 的
`test_metrics.json`。

## 6. 与错误语料运行的审计对比

下表只用于说明修正数据纳入逻辑后的变化。两个实验均只有单一 seed，微小差异不能视为
稳定提升或退化。

| Benchmark | ΔAccuracy | ΔMacro-F1 | ΔmIoU |
|---|---:|---:|---:|
| MFCAD++ | -0.052 pp | -0.091 pp | -0.172 pp |
| TMCAD | +1.012 pp | +1.021 pp | +1.345 pp |
| Fusion360Seg s2.0.0 | -0.112 pp | -0.020 pp | -0.217 pp |
| FabWave | -0.447 pp | -0.062 pp | -0.073 pp |

此前错误运行实际缺少全部 Fusion360Ass 和 8,642 个 Fusion360Rec STEP；本次运行才是
符合“将所有可解析 STEP 加入无标签联合数据集”要求的正式结果。

## 7. 产物与校验值

| Benchmark | Run directory | Best checkpoint SHA-256 |
|---|---|---|
| MFCAD++ | `runs/finetune/20260805-132209_mfcadpp_diffloss_20260805-unlabeled-v2` | `74bb7fda4c3d8da524a33c7e0614b59c3d3e33624880e2b1c53a792f520d9b08` |
| TMCAD | `runs/finetune/20260805-132209_tmcad_diffloss_20260805-unlabeled-v2` | `5da690755461b6065913d08b54008f82983cb2e235bf3e9e7bbc711ab0cd7985` |
| Fusion360Seg s2.0.0 | `runs/finetune/20260805-132209_fusion360seg_s2_0_0_diffloss_20260805-unlabeled-v2` | `6bb4371b6efd69290697770839f9ca7a0c3ea500ee446c9898e9d6f8368e91c9` |
| FabWave | `runs/finetune/20260805-132209_fabwave_diffloss_20260805-unlabeled-v2` | `19e24d87ef97c5407dc00078e229a0428318929594c0009492a055f49d64d80a` |

完整后台流水线日志：
`runs/launch_logs/joint_fusion_gallery_unlabeled_v2_tmux.log`。

数据准备、预训练、微调和测试评估总耗时约 3 小时 45 分钟（11:03–14:48）。
