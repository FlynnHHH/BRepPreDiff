# Rotate-mix：Edge Update Attention → 现有 message_passing（ffn）消融

基线：`reports/inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final.md`。

仅将 encoder_type 改为现有 message_passing（ffn 别名）：使用原 GraphMessageLayer 的 MLP([src, edge]) 消息和邻居均值聚合、原 post-norm 和 SiLU FFN，edge embedding 在各层不更新。这是整个 encoder 层类型替换，不是只移除 attention。保留离散输入和离散监督；num_heads 保留原配置但本 encoder 不使用。
从基线实际 config.yaml 复制配置：预训练 100 epochs、微调 200 epochs、seed 42、50% SO(3)、原 batch/梯度累积、数据划分、Mean+Max 分类池化和验证选优规则。
预训练使用 GPU 0 单卡保持 batch 128；随后 GPU 0、1 每卡一路微调，只测试 MLP head，共七个下游任务。复用完整离散属性基线，差值为 message_passing encoder 减去 attention encoder（百分点）。这是算子替换消融，不保证参数量相等；单种子不代表统计显著性。

实验目录：`/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_message_passing_pre100_ft200_seed42_20260916`；配置差异见 `manifest.json`，状态见 `status.json`。

| Task / Head | 状态 | Accuracy % | ΔAcc | Macro-F1 % | ΔF1 | Weighted-F1 % | ΔWF1 | mIoU % | ΔmIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain | completed | — | — | — | — | — | — | — | — |
| brepprediff_seg_mlp | completed | 98.9600 | -0.0301 | 97.3299 | -0.0388 | 98.9588 | -0.0300 | 94.8475 | -0.0710 |
| fusion360seg_mlp | completed | 96.4785 | -0.4489 | 90.3610 | -1.8643 | 96.4600 | -0.4524 | 84.0110 | -2.6252 |
| mfcadpp_seg_mlp | completed | 99.4639 | -0.1521 | 99.1884 | -0.2387 | 99.4635 | -0.1523 | 98.4002 | -0.4665 |
| tmcad_cls_mlp | completed | 86.8445 | +0.4600 | 86.5552 | +0.6702 | 86.8657 | +0.5919 | 76.8737 | +0.9341 |
| solidletters_cls_mlp | completed | 97.0555 | -0.4074 | 97.1175 | -0.3973 | 97.0481 | -0.3985 | 94.6232 | -0.7633 |
| cadsynth_seg_mlp | completed | 99.6378 | -0.0111 | 99.4717 | -0.0102 | 99.6374 | -0.0111 | 98.9508 | -0.0202 |
| mfinstseg_seg_mlp | completed | 99.3336 | -0.1132 | 98.9165 | -0.2160 | 99.3328 | -0.1136 | 97.8777 | -0.4138 |

下游测试完成：7/7。原始指标与基线见实验目录 results.csv（指标为 0–1，差值为百分点）。
