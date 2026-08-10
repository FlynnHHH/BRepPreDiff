# Fusion Gallery 联合预训练与 DiffLoss 微调报告

生成日期：2026-08-05（Asia/Shanghai）

## 1. 运行结论

后台流水线已正常结束：数据准备、150 epoch 四卡联合预训练及四个 200 epoch
DiffLoss 微调进程均以成功状态退出。正式 test split 已在训练结束后使用各任务的
`best.pt` 补充评估。

但本次运行存在一项重要的数据纳入偏差：Fusion360Ass j1.0.0 的 23,029 个 STEP
实际纳入数为 0；Fusion360Rec r1.0.1 仅纳入 19,316 / 27,958 个。原因不是这些
JSON 本身应作为分割标签，而是无标签预训练配置把 `segs_dir` 指向了 STEP 同目录，
缓存构建器因而匹配到同名任务 JSON，并尝试按“标签列表”解析。Fusion360Ass 的
23,029 个样本全部因此被过滤；Fusion360Rec 的 8,625 个最终模型也因此被过滤，
另有 17 个 STEP 在 OCC `TopExp::MapShapes` 调用中失败。

因此，本报告记录的是一次**训练流程完整结束、但没有完整覆盖目标联合语料**的实验，
不应视为“Fusion360Ass 已参与联合预训练”的最终结论。

## 2. 实际联合预训练数据

联合预训练采用 transductive 协议：下游数据集的 train/val/test geometry 全部并入
无标签预训练 train split。编码器为初始 Blendit 4-layer MLP/FFN GraphMessageLayer。

| 来源 | 请求/发现 STEP | 实际纳入 | 过滤 | 实际组成 |
|---|---:|---:|---:|---|
| Blendit | 172,287 | 172,287 | 0 | train |
| TMCAD | 10,886 | 10,886 | 0 | 8,709 / 1,090 / 1,087 |
| Fusion360Seg s2.0.1 | 35,680 | 35,680 | 0 | 24,964 / 5,350 / 5,366 |
| MFCAD++ | 59,665 | 59,665 | 0 | 41,766 / 8,950 / 8,949 |
| Fusion360Rec r1.0.1 | 27,958 | 19,316 | 8,642 | 15,445 / 1,921 / 1,950 |
| Fusion360Ass j1.0.0 | 23,029 | 0 | 23,029 | 0 / 0 / 0 |
| FabWave | 4,575 | 4,549 | 26 | 3,639 / 463 / 447 |
| **总计** | **334,080** | **302,383** | **31,697** | — |

预训练 DataLoader 日志确认实际样本数为 302,383；四卡 DDP 下每卡 batch size 为
512，全局 batch size 为 2,048，每卡每 epoch 148 batches。

## 3. 联合预训练设置与结果

- 模型：hidden dimension 128，4 个原始 GraphMessageLayer，dropout 0.1。
- 预训练：geometry diffusion，1,000 diffusion timesteps，不启用 coarse-label head。
- 优化：AdamW，学习率 `1e-3`，weight decay `1e-4`，150 epochs。
- 特征处理：per-graph normalization，`uv_grid_size: 10`。
- 设备：4 × NVIDIA Titan RTX。
- 开始训练：2026-08-04 21:23；完成：2026-08-04 23:05。

| Epoch | Total loss | Face noise | Edge noise | Face recon | Edge recon |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.11906 | 0.98242 | 0.32560 | 0.60621 | 0.58643 |
| 150 | 1.32790 | 0.85543 | 0.10032 | 0.37491 | 0.36900 |

预训练 checkpoint：

- `runs/pretrain/20260804-212208_joint_fusion_gallery_mlp_all_splits_20260804-205328/checkpoints/last.pt`
- SHA-256：`8004cad598f052e9e57e78d8b99306409d7548b8c3c7e01aa517619bcf155039`

## 4. DiffLoss 微调协议

四项任务均从上述预训练 `last.pt` 初始化，完整更新 encoder，使用 DiffLoss head，
训练 200 epochs，并按 validation macro-F1 保存 `best.pt`。每项任务在单张 Titan RTX
上训练，四项并行运行。

Fusion360Seg 微调明确使用 `/data/hhfeng/fusion360seg/s2.0.0/`；s2.0.1 只用于本次
联合预训练语料。

| 任务 | 类型 | 类别数 | Train / Val / Test | Best epoch | Best val Acc | Best val Macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| MFCAD++ | 面分割 | 25 | 41,766 / 8,950 / 8,949 | 186 | 99.185% | 98.742% |
| TMCAD | 模型分类 | 10 | 8,709 / 1,090 / 1,087 | 118 | 83.670% | 83.448% |
| Fusion360Seg s2.0.0 | 面分割 | 8 | 24,964 / 5,350 / 5,366 | 191 | 93.073% | 85.989% |
| FabWave | 模型分类 | 44 | 3,639 / 463 / 447 | 80 | 86.825% | 90.463% |

## 5. 正式测试集结果

以下结果均来自 validation macro-F1 最优的 `best.pt`，不是第 200 epoch 的 `last.pt`。

| 任务 | Test samples/faces | Accuracy | Macro-F1 | mIoU | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| MFCAD++ | 8,949 models / 268,982 faces | 99.140% | 98.652% | 97.366% | 99.139% |
| TMCAD | 1,087 models | 83.257% | 82.947% | 71.937% | 83.341% |
| Fusion360Seg s2.0.0 | 5,366 models / 77,070 faces | 92.847% | 85.072% | 76.044% | 92.807% |
| FabWave | 447 models | 86.353% | 91.454% | 90.011% | 86.402% |

完整 confusion matrix 和逐类 precision/recall/F1/IoU 位于各 run 的
`test_metrics.json`。

## 6. 微调产物

| 任务 | Run directory | Best checkpoint SHA-256 |
|---|---|---|
| MFCAD++ | `runs/finetune/20260804-230559_mfcadpp_diffloss_20260804-205328` | `a82c08f189211ae44b66a718c6ab521e7714874fc12247d006017fbedf73d4af` |
| TMCAD | `runs/finetune/20260804-230559_tmcad_diffloss_20260804-205328` | `a10e05357cf2ec8d2e24164b8f291c0b5efd48f953f92071262989528e481713` |
| Fusion360Seg s2.0.0 | `runs/finetune/20260804-230559_fusion360seg_s2_0_0_diffloss_20260804-205328` | `c5947e890b7c21614cd063b8e133f0a8ee5ca173aa16684425a782889b63c63d` |
| FabWave | `runs/finetune/20260804-230559_fabwave_diffloss_20260804-205328` | `8c58eebc334b1cd0f4dca6013a3a72a54c849ea1e325d32de8c892ec1270e73b` |

## 7. 后续建议

若目标是严格完成原始实验要求，应修正无标签缓存构建逻辑：在
`labels_required: false` 时不要匹配或解析同目录 JSON，或将预训练 `segs_dir` 指向
不会匹配任务 JSON 的目录。之后至少重新构建 Fusion360Rec/Fusion360Ass 缓存并重新
执行联合预训练及四项微调。本次结果可作为流程验证和不完整语料基线，但不宜与完整
七来源联合预训练结果直接等同。

