# 输入消融：全语料预训练 → Fusion360Seg / Blend / TMCAD

GPU 0 串行；Inductive9 全部九源 train 划分，去标签；预训练 100 epochs，MLP 全量微调 200 epochs，seed 42，50% SO(3)，TMCAD mean_max。其余协议复制完整输入基线。
单组消融：预训练、微调、测试同时屏蔽面 UV-grid 和边 U-grid 输入。
no_face_edge_grid：face_cont[:, 11:] 的 700 维面采样与 edge_cont[:, 3:] 的 60 维边采样均置零。
保留 11 维面基础几何、3 维边基础几何、面类型、边类型/关系与拓扑。旧 uv_grid 配置保持仅屏蔽面采样的原含义，以便复现旧结果。
屏蔽发生在编码器入口（归一化、旋转、扩散加噪之后），保留原参数形状和初始化；被屏蔽通道的预训练重建/噪声/类别监督仍保留。这是输入信息消融。
使用各组预训练 last.pt；下游按 validation accuracy 选 best.pt 后测试。完整输入对照复用现有基线，单 seed。

| 变体 | 任务 | 状态 | Acc % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |
|---|---|---|---:|---:|---:|---:|---:|
| full_input | fusion360seg | 复用基线 | 96.9275 | 0 | 92.2254 | 86.6362 | 0 |
| full_input | blend | 复用基线 | 98.9901 | 0 | 97.3687 | 94.9185 | 0 |
| full_input | tmcad | 复用基线 | 86.3845 | 0 | 85.8850 | 75.9397 | 0 |
| face_edge_grid | pretrain | completed | — | — | — | — | — |
| face_edge_grid | fusion360seg | completed | 84.0054 | -12.9220 | 76.9711 | 64.0615 | -22.5747 |
| face_edge_grid | blend | completed | 98.8648 | -0.1253 | 97.0033 | 94.2405 | -0.6780 |
| face_edge_grid | tmcad | completed | 84.5446 | -1.8399 | 84.2569 | 73.4062 | -2.5334 |

配置、源码快照、校验哈希、日志和状态：`/home/nvme03/hhfeng/BRepPreDiff/runs/face_edge_grid_ablation_pre100_ft200_seed42_20260924`。
