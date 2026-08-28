# BRepPreDiff Encoder 联合数据预训练最终报告

> 生成日期：2026-08-07  
> 实验完成：2026-07-31 03:06（Asia/Shanghai）  
> 环境：`brepprediff` Conda，4 × NVIDIA TITAN RTX（24 GB）
> 规模：7 种 Encoder × 4 个下游任务，共 28 个测试结果

## 1. 核心结论

- **Edge Update Attention 是联合预训练下的综合最佳方案**：取得 BRepPreDiff 全部三个最佳指标、Fusion360Seg 最佳 Macro-F1/mIoU，并在 TMCAD 上取得最高 Accuracy。
- **Hybrid Transformer 仍最适合 MFCAD++**：Accuracy 99.43%、Macro-F1 99.09%、mIoU 98.21%，三项均为该任务最高。
- 联合预训练对 **TMCAD Baseline** 收益明显：Accuracy 相比单语料预训练提高 1.29 pp；但收益并不在所有架构上稳定出现，SwiGLU、Edge Attention 和 Hybrid 的 TMCAD Accuracy 反而下降。
- Fusion360Seg 的最佳 Accuracy 为 Attention Pooling 的 92.88%，最佳 Macro-F1/mIoU 则来自 Edge Update（86.87%/77.93%）。
- Global Transformer 和 Attention Pooling 的 TMCAD 表现仍明显落后，不建议作为默认方案。

## 2. 联合预训练协议

联合预训练数据共 **278,518 个 B-Rep 图**。所有下游标签在数据加载时剥离。预训练任务只使用几何扩散、几何重建、曲面/边类型和拓扑关系目标。

| 数据组成 | 图数量 | 占比 |
|---|---:|---:|
| BRepPreDiff train | 172,287 | 61.86% |
| TMCAD train/val/test | 10,886 | 3.91% |
| Fusion360Seg train/val/test | 35,680 | 12.81% |
| MFCAD++ train/val/test | 59,665 | 21.42% |
| **合计** | **278,518** | **100.00%** |

这是 **transductive self-supervised** 协议：TMCAD、Fusion360Seg 和 MFCAD++ 的 validation/test 几何参与了无标签预训练。报告中的结果不能视为严格 inductive benchmark，也不能与未见 test geometry 的方法作完全公平比较。

所有架构保持 100 epoch 预训练、100 epoch full fine-tuning，并按验证集 Macro-F1 选择 checkpoint。Attention Pooling 复用联合 Baseline 的预训练权重。

### 2.1 七种 Encoder/Pooling 方法说明

| 方法 | 核心改动 | 单层信息范围 | 边特征是否逐层更新 | 主要计算量 |
|---|---|---|---|---|
| Baseline 4-layer FFN | 均值消息传递 + 标准 4× FFN | B-Rep 一跳邻接 | 否 | 约 `O(E)` |
| SwiGLU FFN | 仅将标准 FFN 换成门控 SwiGLU | B-Rep 一跳邻接 | 否 | 约 `O(E)`，FFN 参数更多 |
| Edge Attention | 4 头、边条件稀疏注意力 | B-Rep 一跳邻接 | 否 | 约 `O(E)` |
| Edge Update Attention | Edge Attention + 每层边状态更新 | B-Rep 一跳邻接 | 是 | 约 `O(E)`，显存敏感 |
| Hybrid Transformer | 3 层局部 Edge Attention + 1 层全局注意力 | 局部后全图 | 否 | 小图 `O(N²)`；大图稀疏 |
| Global Transformer | 4 层均为全局注意力 | 全图 | 否 | 小图 `O(N²)`；大图稀疏 |
| Attention Pooling | Baseline Encoder + 学习式图池化 | Encoder 不变；图级汇聚全体面 | 否 | 额外约 `O(N)` |

#### Baseline 4-layer FFN（对照组）

Baseline 首先将面连续特征与曲面类型嵌入相加，将边连续特征、边类型和拓扑关系嵌入相加。每层对每条有向 B-Rep 边拼接“源面表示 + 边表示”，经 MLP 生成消息，再在目标面上做 degree-normalized mean aggregation。聚合结果通过残差、LayerNorm 后进入扩张 4 倍的 `Linear → SiLU → Dropout → Linear` FFN。四层堆叠后，一个面最多融合四跳局部拓扑信息。它提供参数量和归纳偏置都较稳定的参照系。

#### SwiGLU FFN

SwiGLU 保留 Baseline 的消息构造、邻居均值和四层拓扑感受野，只替换每层 FFN。输入同时投影为 value 与 gate，两支按 `value × SiLU(gate)` 相乘后再投影回隐藏维度。门控机制可按通道选择信息，通常比普通 SiLU FFN 有更强的非线性表达；代价是 FFN 投影参数和计算增加。因为图传播方式不变，它检验的是“更强通道混合”而不是“更强拓扑建模”。

#### Edge Attention

该方案把固定权重的邻居均值替换为 4 头稀疏图注意力。目标面生成 query；源面与边表示共同生成 key/value，边表示还直接生成每个 head 的 attention bias。softmax 只在具有同一目标面的入边之间计算，因此模型能根据面几何、边类型和拓扑关系自适应选择邻居，同时保持与边数 `E` 近似线性的复杂度。各层使用 SwiGLU FFN，但边 embedding 在四层中保持不变。

#### Edge Update Attention

在 Edge Attention 之后，额外用 `[更新后的源面, 更新后的目标面, 当前边]` 经过 MLP 计算 edge delta，并以残差 + LayerNorm 更新边表示。下一层注意力因而使用已经融合两端面上下文的动态边，而非固定输入边。这形成“面影响边、边再调制面”的交替细化，适合加工特征边界和 transition face；代价是额外激活、参数和较高的显存峰值。本实验中它是联合预训练下的综合最佳方案。

#### Hybrid Transformer

前 3 层使用 Edge Attention 提取局部、拓扑约束明确的几何模式，第 4 层再引入图内全局注意力，使远距离面能够直接交互。对不超过 128 个面的图使用精确全连接注意力；更大的图为了适配 24 GB 显存，使用确定性稀疏近似，每个目标面关注自身和一个图级采样源。该结构在局部归纳偏置与长程关系之间折中，并在 MFCAD++ 上取得最佳结果。

#### Global Transformer

四层全部采用图内全局注意力，不再沿显式 B-Rep 邻接边传播。小图使用精确全连接 attention，大图使用与 Hybrid 相同的稀疏近似。它理论上具有最大的长程建模能力，但会削弱“共享边即强关系”的拓扑先验；而大图每个目标仅有极少全局采样连接，信息覆盖也偏稀。本实验四个任务均明显退化，说明当前全局近似无法替代局部 B-Rep 拓扑。

#### Attention Pooling

该方案不修改四层 Baseline Encoder，并复用同一个联合预训练 checkpoint；只在图分类下游把所有面表示的简单均值替换为学习式加权和。一个两层 `Linear → Tanh → Linear` 网络为每个面产生标量分数，再在每个图内 softmax，得到图级 embedding。它希望突出少数判别性面，因此只实质影响 TMCAD 分类；面分割模型不使用图池化。实验中 TMCAD 指标下降，表明学习到的权重可能过度集中或丢失整体形状统计。

## 3. 联合预训练最佳结果

| 数据集 | 最佳 Accuracy | 最佳 Macro-F1 | 最佳 mIoU |
|---|---|---|---|
| BRepPreDiff（3 类面分割） | Edge Update Attention **98.22%** | Edge Update Attention **96.49%** | Edge Update Attention **93.31%** |
| Fusion360Seg（8 类面分割） | Attention Pooling **92.88%** | Edge Update Attention **86.87%** | Edge Update Attention **77.93%** |
| MFCAD++（25 类面分割） | Hybrid Transformer **99.43%** | Hybrid Transformer **99.09%** | Hybrid Transformer **98.21%** |
| TMCAD（10 类图分类） | Edge Update Attention **84.45%** | Baseline 4-layer FFN **84.04%** | Baseline 4-layer FFN **73.22%** |

## 4. 联合预训练完整结果

### 4.1 BRepPreDiff（3 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Joint Baseline | Macro-F1 (%) | ΔF1 | mIoU (%) | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 98.11 | +0.00 | 96.13 | +0.00 | 92.66 | +0.00 |
| SwiGLU FFN | 98.10 | -0.01 | 96.30 | +0.16 | 92.96 | +0.30 |
| Edge Attention | 98.00 | -0.11 | 95.91 | -0.22 | 92.26 | -0.40 |
| Edge Update Attention | 98.22 | +0.11 | 96.49 | +0.36 | 93.31 | +0.65 |
| Hybrid Transformer | 98.06 | -0.05 | 96.01 | -0.12 | 92.44 | -0.22 |
| Global Transformer | 96.65 | -1.46 | 92.30 | -3.83 | 86.00 | -6.66 |
| Attention Pooling | 98.11 | +0.00 | 96.13 | +0.00 | 92.66 | +0.00 |

### 4.2 Fusion360Seg（8 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Joint Baseline | Macro-F1 (%) | ΔF1 | mIoU (%) | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 92.80 | +0.00 | 85.90 | +0.00 | 76.71 | +0.00 |
| SwiGLU FFN | 92.84 | +0.04 | 85.59 | -0.32 | 76.47 | -0.24 |
| Edge Attention | 92.16 | -0.64 | 85.59 | -0.31 | 76.08 | -0.63 |
| Edge Update Attention | 92.83 | +0.03 | 86.87 | +0.97 | 77.93 | +1.22 |
| Hybrid Transformer | 92.40 | -0.40 | 85.79 | -0.11 | 76.58 | -0.13 |
| Global Transformer | 90.43 | -2.37 | 81.41 | -4.49 | 70.74 | -5.97 |
| Attention Pooling | 92.88 | +0.08 | 86.52 | +0.61 | 77.37 | +0.65 |

### 4.3 MFCAD++（25 类面分割）

| Variant | Accuracy (%) | ΔAcc vs Joint Baseline | Macro-F1 (%) | ΔF1 | mIoU (%) | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 99.09 | +0.00 | 98.59 | +0.00 | 97.25 | +0.00 |
| SwiGLU FFN | 99.11 | +0.02 | 98.63 | +0.03 | 97.32 | +0.07 |
| Edge Attention | 99.21 | +0.12 | 98.76 | +0.17 | 97.57 | +0.32 |
| Edge Update Attention | 99.36 | +0.27 | 99.00 | +0.41 | 98.04 | +0.79 |
| Hybrid Transformer | 99.43 | +0.34 | 99.09 | +0.50 | 98.21 | +0.96 |
| Global Transformer | 98.19 | -0.90 | 97.19 | -1.40 | 94.62 | -2.63 |
| Attention Pooling | 99.10 | +0.01 | 98.60 | +0.01 | 97.27 | +0.02 |

### 4.4 TMCAD（10 类图分类）

| Variant | Accuracy (%) | ΔAcc vs Joint Baseline | Macro-F1 (%) | ΔF1 | mIoU (%) | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4-layer FFN | 84.36 | +0.00 | 84.04 | +0.00 | 73.22 | +0.00 |
| SwiGLU FFN | 82.43 | -1.93 | 82.06 | -1.97 | 70.56 | -2.66 |
| Edge Attention | 83.44 | -0.92 | 83.23 | -0.81 | 72.09 | -1.13 |
| Edge Update Attention | 84.45 | +0.09 | 83.95 | -0.09 | 72.81 | -0.42 |
| Hybrid Transformer | 82.34 | -2.02 | 81.79 | -2.24 | 70.37 | -2.85 |
| Global Transformer | 81.14 | -3.22 | 80.82 | -3.22 | 68.60 | -4.62 |
| Attention Pooling | 81.23 | -3.13 | 80.83 | -3.20 | 68.55 | -4.68 |

## 5. 联合预训练相对单语料预训练的变化

下表比较同一架构在联合预训练与原 172,287 图单语料预训练之间的差值；正数表示联合预训练更高。

| Variant | 数据集 | ΔAccuracy (pp) | ΔMacro-F1 (pp) | ΔmIoU (pp) |
|---|---|---:|---:|---:|
| Baseline 4-layer FFN | BRepPreDiff（3 类面分割） | +0.07 | +0.14 | +0.26 |
| Baseline 4-layer FFN | Fusion360Seg（8 类面分割） | +0.24 | -0.53 | -0.50 |
| Baseline 4-layer FFN | MFCAD++（25 类面分割） | -0.02 | -0.01 | -0.02 |
| Baseline 4-layer FFN | TMCAD（10 类图分类） | +1.29 | +1.35 | +1.88 |
| SwiGLU FFN | BRepPreDiff（3 类面分割） | +0.02 | -0.05 | -0.09 |
| SwiGLU FFN | Fusion360Seg（8 类面分割） | +0.04 | -0.53 | -0.50 |
| SwiGLU FFN | MFCAD++（25 类面分割） | +0.01 | +0.04 | +0.09 |
| SwiGLU FFN | TMCAD（10 类图分类） | -0.83 | -0.75 | -0.83 |
| Edge Attention | BRepPreDiff（3 类面分割） | -0.15 | -0.51 | -0.94 |
| Edge Attention | Fusion360Seg（8 类面分割） | -0.16 | +0.91 | +0.88 |
| Edge Attention | MFCAD++（25 类面分割） | +0.02 | +0.04 | +0.07 |
| Edge Attention | TMCAD（10 类图分类） | -1.01 | -0.96 | -1.38 |
| Edge Update Attention | BRepPreDiff（3 类面分割） | +0.01 | +0.08 | +0.15 |
| Edge Update Attention | Fusion360Seg（8 类面分割） | +0.30 | +0.97 | +1.21 |
| Edge Update Attention | MFCAD++（25 类面分割） | +0.02 | +0.05 | +0.09 |
| Edge Update Attention | TMCAD（10 类图分类） | +0.46 | +0.34 | +0.14 |
| Hybrid Transformer | BRepPreDiff（3 类面分割） | -0.06 | +0.03 | +0.05 |
| Hybrid Transformer | Fusion360Seg（8 类面分割） | -0.02 | +0.33 | +0.37 |
| Hybrid Transformer | MFCAD++（25 类面分割） | +0.07 | +0.10 | +0.19 |
| Hybrid Transformer | TMCAD（10 类图分类） | -1.01 | -1.19 | -1.53 |
| Global Transformer | BRepPreDiff（3 类面分割） | +0.15 | +0.04 | +0.05 |
| Global Transformer | Fusion360Seg（8 类面分割） | +0.13 | +0.99 | +0.91 |
| Global Transformer | MFCAD++（25 类面分割） | -0.05 | -0.02 | -0.05 |
| Global Transformer | TMCAD（10 类图分类） | +1.20 | +1.20 | +1.30 |
| Attention Pooling | BRepPreDiff（3 类面分割） | +0.07 | +0.14 | +0.26 |
| Attention Pooling | Fusion360Seg（8 类面分割） | +0.45 | +1.20 | +1.45 |
| Attention Pooling | MFCAD++（25 类面分割） | -0.02 | +0.00 | +0.01 |
| Attention Pooling | TMCAD（10 类图分类） | -0.28 | -0.29 | -0.45 |

### 各数据集最优值的协议变化

| 数据集 | 单语料最佳 Acc | 联合最佳 Acc | ΔAcc | 单语料最佳 mIoU | 联合最佳 mIoU | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff（3 类面分割） | 98.21 | 98.22 | +0.01 | 93.20 | 93.31 | +0.12 |
| Fusion360Seg（8 类面分割） | 92.80 | 92.88 | +0.08 | 77.22 | 77.93 | +0.71 |
| MFCAD++（25 类面分割） | 99.36 | 99.43 | +0.07 | 98.01 | 98.21 | +0.19 |
| TMCAD（10 类图分类） | 84.45 | 84.45 | +0.00 | 73.47 | 73.22 | -0.25 |

## 6. 与 BRep2Shape 的数值参考

| 数据集 | 联合实验最佳方案 | Acc (%) | BRep2Shape Acc (%) | ΔAcc | mIoU (%) | BRep2Shape IoU (%) | ΔIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff（3 类面分割） | Edge Update Attention | 98.22 | — | — | 93.31 | — | — |
| Fusion360Seg（8 类面分割） | Attention Pooling（Acc）/ Edge Update Attention（mIoU） | 92.88 | 96.88 | -4.00 | 77.93 | 83.77 | -5.84 |
| MFCAD++（25 类面分割） | Hybrid Transformer | 99.43 | 99.35 | +0.08 | 98.21 | 98.02 | +0.19 |
| TMCAD（10 类图分类） | Edge Update Attention（Acc）/ Baseline 4-layer FFN（mIoU） | 84.45 | 84.72 | -0.27 | 73.22 | — | — |

BRep2Shape 使用不同的预训练语料、tokenizer、模型和数据处理协议；尤其本联合实验在无标签预训练阶段见过下游 test geometry，因此这里只能比较数值量级，不能据此提出严格优于 BRep2Shape 的结论。论文主表也没有报告 Macro-F1。

参考：[BRep2Shape arXiv](https://arxiv.org/abs/2602.07429)、[OpenReview](https://openreview.net/forum?id=i5knGuNoRX)。

## 7. 建议

1. 若接受 transductive 协议，默认选择 **Edge Update Attention**。
2. MFCAD++ 专用模型选择 **Hybrid Transformer**。
3. 若要求严格 inductive 泛化，应重新构造仅含各数据集 train geometry 的联合预训练集，排除 validation/test，并重新运行重点方案。
4. 当前每个配置只有一个随机种子；低于 1 pp 的差异需要至少 3 个种子确认。
5. 原始指标位于 `runs/encoder_benchmark_joint/<variant>/`；原单语料结果位于 `runs/encoder_benchmark/<variant>/`。
