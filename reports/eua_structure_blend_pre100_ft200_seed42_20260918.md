# EUA 结构消融：Blend（BRepPreDiff 三分类过渡面分割）/ MLP

E0：复用完整 EUA；E1：attention + 固定初始边；E2：attention + MLP([0,0,edge])；A1：均匀权重 + 完整边更新；A2：均匀权重 + 固定初始边。
固定边不冻结初始 embedding 参数。保留模块形状和初始化，关闭路径不参与计算；注册参数量一致不等于有效容量一致。预训练 edge_context 的端点交互仍保留。
复制基线实际配置：Inductive9 预训练 100 epochs，batch 128，LR 1e-4；Blend MLP 微调 200 epochs；seed 42，50% SO(3)，其余 batch/累积、损失、验证选优及数据划分不变。
仅使用 GPU 4，每次一个变体；复用 Fusion360Seg 消融中各变体的 100-epoch 预训练权重，仅重新微调 Blend，并用 validation accuracy 最优 checkpoint 测试。单 seed 结果不代表统计显著性。
基线结果：`/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json`。实验目录：`/home/nvme03/hhfeng/BRepPreDiff/runs/eua_structure_blend_pre100_ft200_seed42_20260918`。

| 变体 | 预训练 | 微调/测试 | Accuracy % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |
|---|---|---|---:|---:|---:|---:|---:|
| E0 | 已完成 | 复用基线 | 98.9901 | 0 | 97.3687 | 94.9185 | 0 |
| E1 | 复用100 epochs | completed | 98.9872 | -0.0029 | 97.3976 | 94.9732 | +0.0547 |
| E2 | 复用100 epochs | completed | 98.9386 | -0.0515 | 97.3643 | 94.9128 | -0.0057 |
| A1 | 复用100 epochs | completed | 98.8726 | -0.1175 | 97.1233 | 94.4635 | -0.4550 |
| A2 | 复用100 epochs | completed | 98.8998 | -0.0903 | 97.1768 | 94.5616 | -0.3569 |

交互量 I = (E0−E1)−(A1−A2)，单位百分点：
- accuracy: +0.0301
- macro_iou: +0.0434

新增测试结果：4/4；完整运行状态见 status.json，配置差异见 manifest.json。
