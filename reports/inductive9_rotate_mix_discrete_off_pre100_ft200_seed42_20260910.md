# Rotate-mix：关闭离散预训练损失对照

基线：`reports/inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final.md`。

仅将 categorical_loss_weight 从 0.5 改为 0、relation_loss_weight 从 0.3 改为 0；保留离散输入特征和网络结构。
从基线实际 config.yaml 复制配置：预训练 100 epochs、微调 200 epochs、seed 42、50% SO(3)、原 batch/梯度累积、数据划分、Mean+Max 分类池化和验证选优规则。
预训练使用 GPU 0 单卡保持 batch 128；随后 GPU 0–3 每卡一路微调。基线结果复用，差值为关闭损失后减去基线（百分点）。

实验目录：`/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910`；配置差异见 `manifest.json`，状态见 `status.json`。

| Task / Head | 状态 | Accuracy % | ΔAcc | Macro-F1 % | ΔF1 | Weighted-F1 % | ΔWF1 | mIoU % | ΔmIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain | completed | — | — | — | — | — | — | — | — |
| brepprediff_seg_mlp | completed | 98.9144 | -0.0757 | 97.1606 | -0.2081 | 98.9116 | -0.0772 | 94.5318 | -0.3867 |
| brepprediff_seg_diffloss | completed | 99.0095 | +0.0233 | 97.4261 | +0.0318 | 99.0069 | +0.0228 | 95.0259 | +0.0584 |
| fusion360seg_mlp | completed | 97.0845 | +0.1570 | 92.0885 | -0.1369 | 97.0746 | +0.1621 | 86.4786 | -0.1576 |
| fusion360seg_diffloss | completed | 96.9378 | -0.0804 | 91.1295 | +0.1772 | 96.9164 | -0.0938 | 85.3073 | +0.0600 |
| mfcadpp_seg_mlp | completed | 99.6119 | -0.0041 | 99.4320 | +0.0050 | 99.6117 | -0.0041 | 98.8768 | +0.0101 |
| mfcadpp_seg_diffloss | completed | 99.5963 | +0.0000 | 99.3937 | +0.0094 | 99.5961 | -0.0002 | 98.8013 | +0.0183 |
| tmcad_cls_mlp | completed | 84.9126 | -1.4719 | 84.3389 | -1.5460 | 84.7178 | -1.5560 | 73.6299 | -2.3098 |
| tmcad_cls_diffloss | completed | 86.7525 | +0.1840 | 86.4239 | +0.1974 | 86.6753 | +0.0941 | 76.6551 | +0.2368 |
| solidletters_cls_mlp | completed | 97.2463 | -0.2166 | 97.3194 | -0.1954 | 97.2471 | -0.1995 | 95.0167 | -0.3698 |
| solidletters_cls_diffloss | completed | 97.3546 | +0.1650 | 97.4163 | +0.1509 | 97.3441 | +0.1576 | 95.2065 | +0.2597 |
| cadsynth_seg_mlp | completed | 99.6589 | +0.0100 | 99.5072 | +0.0253 | 99.6586 | +0.0100 | 99.0206 | +0.0496 |
| cadsynth_seg_diffloss | completed | 99.6589 | -0.0029 | 99.5018 | -0.0011 | 99.6585 | -0.0029 | 99.0098 | -0.0022 |
| mfinstseg_seg_mlp | completed | 99.4721 | +0.0252 | 99.1773 | +0.0448 | 99.4717 | +0.0253 | 98.3793 | +0.0878 |
| mfinstseg_seg_diffloss | completed | 99.4545 | -0.0088 | 99.1272 | -0.0190 | 99.4546 | -0.0084 | 98.2823 | -0.0368 |
