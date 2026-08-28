# BRepPreDiff 四层 Encoder 改进实验最终报告

> 生成日期：2026-07-30  
> 运行环境：`brepprediff` Conda、4 × NVIDIA TITAN RTX（24 GB）
> 结果范围：7 种 Encoder × 4 个数据集，共 28 个独立测试结果

## 1. 结论摘要

- **Edge Attention 是综合首选**：在 TMCAD 分类上达到最高 Accuracy 84.45%，较 Baseline 提升 1.38 pp；同时取得 BRepPreDiff 最高 Macro-F1 和 mIoU。
- **Edge Update Attention 最适合 BRepPreDiff Accuracy**：达到 98.21%，并在 MFCAD++ 上取得 99.33% Accuracy / 97.94% mIoU。
- **Hybrid Transformer 最适合 MFCAD++**：达到全实验最高的 99.36% Accuracy、98.99% Macro-F1 和 98.01% mIoU。
- **Fusion360Seg 没有从复杂注意力中稳定获益**：SwiGLU 的 Accuracy 最高（92.80%），但 Baseline 的 Macro-F1（86.44%）和 mIoU（77.22%）最高。
- **纯 Global Transformer 明显退化**：四个数据集均弱于 Baseline；当前对超大图采用的稀疏全局近似不足以替代局部 B-Rep 拓扑归纳偏置。
- **Attention Pooling 不推荐**：TMCAD Accuracy 比 Baseline 下降 1.56 pp，说明仅替换图级池化没有带来收益。

## 2. 实验协议

所有内部消融使用相同数据划分、隐藏维度 128、4 个 Encoder block、dropout 0.1；适用时使用 4 个 attention head。每种 Encoder 进行 100 epoch 自监督预训练，随后在四个下游任务上 full fine-tuning 100 epoch，并按验证集 Macro-F1 选择最佳 checkpoint，最终仅在 test split 上评测一次。

| Variant | Git 分支 | 预训练 micro-batch × 累积 | 微调 micro-batch × 累积 |
|---|---|---:|---:|
| Baseline 4-layer FFN | `experiment/encoder-benchmark` | 512 × 1 = 512 | 256 × 1 = 256 |
| SwiGLU FFN | `experiment/encoder-swiglu` | 512 × 1 = 512 | 256 × 1 = 256 |
| Edge Attention | `experiment/encoder-edge-attention` | 256 × 2 = 512 | 128 × 2 = 256 |
| Edge Update Attention | `experiment/encoder-edge-update` | 128 × 4 = 512 | 128 × 2 = 256 |
| Hybrid Transformer | `experiment/encoder-hybrid-transformer` | 128 × 4 = 512 | 64 × 4 = 256 |
| Global Transformer | `experiment/encoder-global-transformer` | 64 × 8 = 512 | 32 × 8 = 256 |
| Attention Pooling | `experiment/encoder-attention-pooling` | 512 × 1 = 512（复用 Baseline 预训练） | 256 × 1 = 256 |

Edge Update 最初以 256、128 的 micro-batch 运行时均在包含超大图的 batch 上 OOM；最终使用 `128 × 4` 从检查点恢复并完成 100 epoch。一次独立的 128 单 epoch 重试成功，随后同配置完整续训也成功。预训练缓存最大图包含 96,794 个面，因此峰值显存受 batch 内图规模长尾显著影响。

Global Transformer 对不超过 128 个面的图使用精确全局注意力；更大的图使用确定性稀疏全局近似（每个目标包含 self 和图级采样源），以避免二次复杂度导致显存溢出。

## 3. 最佳结果总览

| 数据集 | 最佳 Accuracy | 最佳 Macro-F1 | 最佳 mIoU |
|---|---|---|---|
| BRepPreDiff（3 类面分割） | Edge Update Attention **98.21%** | Edge Attention **96.42%** | Edge Attention **93.20%** |
| Fusion360Seg（8 类面分割） | SwiGLU FFN **92.80%** | Baseline 4-layer FFN **86.44%** | Baseline 4-layer FFN **77.22%** |
| MFCAD++（25 类面分割） | Hybrid Transformer **99.36%** | Hybrid Transformer **98.99%** | Hybrid Transformer **98.01%** |
| TMCAD（10 类图分类） | Edge Attention **84.45%** | Edge Attention **84.19%** | Edge Attention **73.47%** |

## 4. 各数据集完整结果

### 4.1 BRepPreDiff（3 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Baseline (pp) | Macro-F1 (%) | ΔF1 (pp) | mIoU (%) | ΔmIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 98.04 | +0.00 | 95.99 | +0.00 | 92.40 | +0.00 |
| SwiGLU FFN | 98.08 | +0.04 | 96.34 | +0.35 | 93.05 | +0.65 |
| Edge Attention | 98.15 | +0.11 | 96.42 | +0.43 | 93.20 | +0.79 |
| Edge Update Attention | 98.21 | +0.17 | 96.41 | +0.42 | 93.16 | +0.76 |
| Hybrid Transformer | 98.11 | +0.07 | 95.99 | -0.01 | 92.39 | -0.02 |
| Global Transformer | 96.50 | -1.54 | 92.26 | -3.73 | 85.96 | -6.45 |
| Attention Pooling | 98.04 | +0.00 | 95.99 | +0.00 | 92.40 | +0.00 |

### 4.2 Fusion360Seg（8 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Baseline (pp) | Macro-F1 (%) | ΔF1 (pp) | mIoU (%) | ΔmIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 92.56 | +0.00 | 86.44 | +0.00 | 77.22 | +0.00 |
| SwiGLU FFN | 92.80 | +0.24 | 86.12 | -0.32 | 76.97 | -0.25 |
| Edge Attention | 92.32 | -0.23 | 84.68 | -1.76 | 75.20 | -2.02 |
| Edge Update Attention | 92.54 | -0.02 | 85.90 | -0.54 | 76.72 | -0.50 |
| Hybrid Transformer | 92.42 | -0.13 | 85.46 | -0.97 | 76.22 | -1.00 |
| Global Transformer | 90.30 | -2.25 | 80.42 | -6.01 | 69.84 | -7.38 |
| Attention Pooling | 92.43 | -0.13 | 85.32 | -1.12 | 75.92 | -1.30 |

### 4.3 MFCAD++（25 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Baseline (pp) | Macro-F1 (%) | ΔF1 (pp) | mIoU (%) | ΔmIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 99.11 | +0.00 | 98.60 | +0.00 | 97.27 | +0.00 |
| SwiGLU FFN | 99.10 | -0.01 | 98.58 | -0.02 | 97.23 | -0.04 |
| Edge Attention | 99.19 | +0.08 | 98.72 | +0.12 | 97.50 | +0.23 |
| Edge Update Attention | 99.33 | +0.22 | 98.95 | +0.35 | 97.94 | +0.68 |
| Hybrid Transformer | 99.36 | +0.25 | 98.99 | +0.39 | 98.01 | +0.75 |
| Global Transformer | 98.24 | -0.87 | 97.21 | -1.39 | 94.67 | -2.60 |
| Attention Pooling | 99.11 | +0.00 | 98.60 | -0.00 | 97.26 | -0.01 |

### 4.4 TMCAD（10 类图分类）

| Variant | Accuracy (%) | ΔAcc vs Baseline (pp) | Macro-F1 (%) | ΔF1 (pp) | mIoU (%) | ΔmIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 83.07 | +0.00 | 82.69 | +0.00 | 71.34 | +0.00 |
| SwiGLU FFN | 83.26 | +0.18 | 82.81 | +0.12 | 71.39 | +0.05 |
| Edge Attention | 84.45 | +1.38 | 84.19 | +1.50 | 73.47 | +2.13 |
| Edge Update Attention | 83.99 | +0.92 | 83.61 | +0.93 | 72.66 | +1.32 |
| Hybrid Transformer | 83.35 | +0.28 | 82.99 | +0.30 | 71.91 | +0.57 |
| Global Transformer | 79.94 | -3.13 | 79.61 | -3.07 | 67.30 | -4.04 |
| Attention Pooling | 81.51 | -1.56 | 81.12 | -1.56 | 68.99 | -2.35 |

## 5. 与 BRep2Shape 的数值比较

BRep2Shape 数值取自论文主表。该比较用于定位量级，不是严格同协议复现。

| 数据集 | 本实验最佳方案 | 本实验 Acc (%) | BRep2Shape Acc (%) | ΔAcc (pp) | 本实验 mIoU (%) | BRep2Shape IoU (%) | ΔIoU (pp) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff（3 类面分割） | Edge Update Attention（Acc）/ Edge Attention（mIoU） | 98.21 | — | — | 93.20 | — | — |
| Fusion360Seg（8 类面分割） | SwiGLU FFN（Acc）/ Baseline 4-layer FFN（mIoU） | 92.80 | 96.88 | -4.08 | 77.22 | 83.77 | -6.55 |
| MFCAD++（25 类面分割） | Hybrid Transformer | 99.36 | 99.35 | +0.01 | 98.01 | 98.02 | -0.01 |
| TMCAD（10 类图分类） | Edge Attention | 84.45 | 84.72 | -0.27 | 73.47 | — | — |

比较限制：

- BRep2Shape 没有报告 BRepPreDiff 三分类 transition segmentation，因此 BRepPreDiff 无法比较。
- BRep2Shape 使用约 250k 预训练样本和不同的几何解析/tokenizer；本实验预训练缓存为 172,287 个图。
- 本地 TMCAD 有效样本数与 BRep2Shape 论文不同，Accuracy 差值不能解释为纯架构收益。
- 论文报告的 IoU 与本实验 Macro-IoU 在实现细节上可能不完全相同；表中仅作数值参考。
- BRep2Shape 主表未报告 Macro-F1，因此不进行 F1 对比。
- 参考：[BRep2Shape arXiv](https://arxiv.org/abs/2602.07429)、[OpenReview](https://openreview.net/forum?id=i5knGuNoRX)。

## 6. 方案选择建议

1. 默认 Encoder 建议采用 **Edge Attention**。它在图分类任务上收益最大，同时保持 BRepPreDiff 分割的领先表现，结构复杂度也低于全局 Transformer。
2. 如果核心目标是 MFCAD++ 特征识别，采用 **Hybrid Transformer**；其局部注意力保留拓扑归纳偏置，最后一层全局交互补充跨区域信息。
3. 如果核心目标是 BRepPreDiff transition face Accuracy，可采用 **Edge Update Attention**；但其显存峰值更敏感，建议保留 `128 × 4` 或更小 micro-batch。
4. Fusion360Seg 优先保留 Baseline，或在只关注 Accuracy 时采用轻量 **SwiGLU**。
5. 不建议继续投入当前 **Global Transformer** 和 **Attention Pooling** 实现，除非先修改稀疏全局 token 设计或引入层次化/区域级 token。

## 7. 统计与复现实验注意事项

本报告每个配置只有一次完整训练，没有多随机种子均值和标准差。小于约 1 pp 的差异可能包含随机初始化和数据顺序波动；在合并默认架构前，建议对 Edge Attention、Edge Update、Hybrid 和 Baseline 至少运行 3 个随机种子。

原始指标位于 `runs/encoder_benchmark/<variant>/*_test_metrics.json`，训练状态位于相同目录的 `state.json`，模型检查点位于 `runs/encoder_benchmark/{pretrain,finetune}/`。
