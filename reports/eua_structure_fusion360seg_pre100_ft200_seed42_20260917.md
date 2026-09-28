# EUA 结构消融：Fusion360Seg / MLP

E0：复用完整 EUA；E1：attention + 固定初始边；E2：attention + MLP([0,0,edge])；A1：均匀权重 + 完整边更新；A2：均匀权重 + 固定初始边。
固定边不冻结初始 embedding 参数。保留模块形状和初始化，关闭路径不参与计算；注册参数量一致不等于有效容量一致。预训练 edge_context 的端点交互仍保留。
复制基线实际配置：Inductive9 预训练 100 epochs，batch 128，LR 1e-4；Fusion360Seg MLP 微调 200 epochs；seed 42，50% SO(3)，其余 batch/累积、损失、验证选优及数据划分不变。
仅使用 GPU 4，每次一个变体；每个变体先预训练，再微调并用 validation accuracy 最优 checkpoint 测试。单 seed 结果不代表统计显著性。
基线结果：`/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/test_metrics.json`。实验目录：`/home/nvme03/hhfeng/BRepPreDiff/runs/eua_structure_fusion360seg_pre100_ft200_seed42_20260917`。

| 变体 | 预训练 | 微调/测试 | Accuracy % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |
|---|---|---|---:|---:|---:|---:|---:|
| E0 | 已完成 | 复用基线 | 96.9275 | 0 | 92.2254 | 86.6362 | 0 |
| E1 | completed | completed | 96.7419 | -0.1855 | 90.9147 | 84.8964 | -1.7398 |
| E2 | completed | completed | 96.6641 | -0.2634 | 91.8239 | 85.9653 | -0.6709 |
| A1 | completed | completed | 97.0339 | +0.1064 | 92.5410 | 87.0829 | +0.4466 |
| A2 | completed | completed | 95.8440 | -1.0834 | 90.0018 | 83.1705 | -3.4657 |

交互量 I = (E0−E1)−(A1−A2)，单位百分点：
- accuracy: -1.0043
- macro_iou: -2.1725

新增测试结果：4/4；完整运行状态见 status.json，配置差异见 manifest.json。
