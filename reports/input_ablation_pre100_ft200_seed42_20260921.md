# 输入消融：全语料预训练 → Fusion360Seg / Blend / TMCAD

GPU 3 串行；Inductive9 全部九源 train 划分，去标签；预训练 100 epochs，MLP 全量微调 200 epochs，seed 42，50% SO(3)，TMCAD mean_max。其余协议复制完整输入基线。
三组独立消融，预训练、微调、测试均保持相同屏蔽：
- no_face_geometry：屏蔽 face_cont 前 11 维和 surface type，保留面 UV-grid。
- no_edge_geometry：屏蔽全部 edge_cont（3 维基础量及 60 维边采样）、edge type、edge relation；保留图连接及边更新。
- no_uv_grid：屏蔽 face_cont 第 11 维之后的全部面 UV 采样（位置、法向、trim mask）；保留边采样。
屏蔽发生在编码器入口（归一化、旋转、扩散加噪之后），保留原参数形状和初始化；被屏蔽通道的预训练重建/噪声/类别监督仍保留。这是输入信息消融。
使用各组预训练 last.pt；下游按 validation accuracy 选 best.pt 后测试。完整输入对照复用现有基线，单 seed。

| 变体 | 任务 | 状态 | Acc % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |
|---|---|---|---:|---:|---:|---:|---:|
| full_input | fusion360seg | 复用基线 | 96.9275 | 0 | 92.2254 | 86.6362 | 0 |
| full_input | blend | 复用基线 | 98.9901 | 0 | 97.3687 | 94.9185 | 0 |
| full_input | tmcad | 复用基线 | 86.3845 | 0 | 85.8850 | 75.9397 | 0 |
| face_geometry | pretrain | completed | — | — | — | — | — |
| face_geometry | fusion360seg | completed | 96.7575 | -0.1700 | 91.2064 | 85.2659 | -1.3703 |
| face_geometry | blend | completed | 98.8910 | -0.0990 | 97.1223 | 94.4608 | -0.4577 |
| face_geometry | tmcad | completed | 85.2806 | -1.1040 | 84.7303 | 74.0010 | -1.9386 |
| edge_geometry | pretrain | completed | — | — | — | — | — |
| edge_geometry | fusion360seg | completed | 94.1183 | -2.8091 | 87.4521 | 79.3459 | -7.2903 |
| edge_geometry | blend | completed | 98.9988 | +0.0087 | 97.4979 | 95.1613 | +0.2429 |
| edge_geometry | tmcad | completed | 85.7406 | -0.6440 | 85.4007 | 75.2563 | -0.6834 |
| uv_grid | pretrain | completed | — | — | — | — | — |
| uv_grid | fusion360seg | completed | 94.0976 | -2.8299 | 88.1427 | 80.2787 | -6.3575 |
| uv_grid | blend | completed | 98.8707 | -0.1194 | 96.9303 | 94.1056 | -0.8128 |
| uv_grid | tmcad | completed | 86.5685 | +0.1840 | 86.2698 | 76.4293 | +0.4896 |

配置、源码快照、校验哈希、日志和状态：`/home/nvme03/hhfeng/BRepPreDiff/runs/input_ablation_pre100_ft200_seed42_20260921`。
