# Edge Update Attention：三种分割 Head 延续至 200 Epochs

> 生成日期：2026-08-10  
> 任务：Blendit、Fusion360Seg、MFCAD++  
> Head：MLP、DiffLoss x_start、DiffLoss x_start+0.5epsilon  
> 方法：从各自 epoch-100 `last.pt` 继承模型和 AdamW optimizer，继续训练 epochs 101-200；按 validation accuracy 在完整 epochs 1-200 中重新选择全局 best。

## 全局 best 是否发生变化

差值是延长到 200 epochs 后的全局 best test 指标减去原 100-epoch best；若后半程 validation accuracy 未超过原 best，则保留原 checkpoint，差值为零。

| Benchmark | Head | 1-100 best | 101-200 best | ΔVal Acc (late-old, pp) | Global best period | Global epoch | ΔTest Acc | ΔMacro-F1 | ΔmIoU |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Blendit | MLP | 18 | 117 | -0.0197 | 1-100 | 18 | +0.0000 | +0.0000 | +0.0000 |
| Blendit | DiffLoss x_start | 36 | 172 | +0.0372 | 101-200 | 172 | +0.0217 | +0.0002 | -0.0035 |
| Blendit | DiffLoss x_start+0.5eps | 37 | 181 | -0.0393 | 1-100 | 37 | +0.0000 | +0.0000 | +0.0000 |
| Fusion360Seg s2.0.0 | MLP | 97 | 165 | +0.0951 | 101-200 | 165 | +0.0337 | -0.1589 | -0.1991 |
| Fusion360Seg s2.0.0 | DiffLoss x_start | 58 | 172 | +0.1251 | 101-200 | 172 | +0.2764 | +1.0422 | +1.3370 |
| Fusion360Seg s2.0.0 | DiffLoss x_start+0.5eps | 64 | 168 | +0.0613 | 101-200 | 168 | +0.1103 | +0.8498 | +1.0767 |
| MFCAD++ | MLP | 88 | 172 | +0.0416 | 101-200 | 172 | +0.0141 | +0.0520 | +0.1029 |
| MFCAD++ | DiffLoss x_start | 94 | 174 | +0.0442 | 101-200 | 174 | +0.0461 | +0.0680 | +0.1314 |
| MFCAD++ | DiffLoss x_start+0.5eps | 96 | 190 | +0.0471 | 101-200 | 190 | +0.0580 | +0.0708 | +0.1346 |

## Epochs 101-200 候选本身的测试结果

| Benchmark | Head | Late epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|
| Blendit | MLP | 117 | 98.1042 | 96.0522 | 98.1169 | 92.5123 |
| Blendit | DiffLoss x_start | 172 | 98.1452 | 96.2543 | 98.1575 | 92.8807 |
| Blendit | DiffLoss x_start+0.5eps | 181 | 97.9836 | 95.6238 | 97.9973 | 91.7374 |
| Fusion360Seg s2.0.0 | MLP | 165 | 93.0933 | 87.2338 | 93.0572 | 78.5003 |
| Fusion360Seg s2.0.0 | DiffLoss x_start | 172 | 93.0946 | 86.9494 | 93.0608 | 78.1552 |
| Fusion360Seg s2.0.0 | DiffLoss x_start+0.5eps | 168 | 93.2672 | 87.9302 | 93.2362 | 79.3756 |
| MFCAD++ | MLP | 172 | 99.3773 | 99.0433 | 99.3774 | 98.1201 |
| MFCAD++ | DiffLoss x_start | 174 | 99.4286 | 99.1056 | 99.4287 | 98.2397 |
| MFCAD++ | DiffLoss x_start+0.5eps | 190 | 99.3910 | 99.0336 | 99.3915 | 98.1001 |

## 续训运行与最终 checkpoint

- Blendit / MLP：续训 `runs/edge_update_new_joint/finetune/20260810-144236_edge_update_extend200_blendit_mlp_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260807-200941_edge_update_blendit_seg_20260807-123259/checkpoints/best.pt`；SHA-256 `10f8bb47a845206701a5c44272c5bb0c29e90ed473cb051baff5d8c7dc474c9d`
- Blendit / DiffLoss x_start：续训 `runs/edge_update_new_joint/finetune/20260810-144740_edge_update_extend200_blendit_xstart_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-144740_edge_update_extend200_blendit_xstart_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `cfbf7e992ff4d9c6d11ab7909e33542822a54527726da369f663a19a479262b2`
- Blendit / DiffLoss x_start+0.5eps：续训 `runs/edge_update_new_joint/finetune/20260810-145200_edge_update_extend200_blendit_xse_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260808-183835_edge_update_blendit_seg_diffloss_xse_20260808-xse-titan/checkpoints/best.pt`；SHA-256 `1e1ffff9eb6b7a86b23609d91cc780ded2c5efe9aa06837ce378b4629d6c37b7`
- Fusion360Seg s2.0.0 / MLP：续训 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_fusion360seg_mlp_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_fusion360seg_mlp_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `3fa9827d42e815a3cf532140a9149da5ada2c828cc39e61be865a2078a2b23c9`
- Fusion360Seg s2.0.0 / DiffLoss x_start：续训 `runs/edge_update_new_joint/finetune/20260810-141509_edge_update_extend200_fusion360seg_xstart_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-141509_edge_update_extend200_fusion360seg_xstart_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `2f5a3b20de6c37e451b4e1c96aed1bf2c7cb6d7c9bc6f4c16fed165fbd6920e0`
- Fusion360Seg s2.0.0 / DiffLoss x_start+0.5eps：续训 `runs/edge_update_new_joint/finetune/20260810-150034_edge_update_extend200_fusion360seg_xse_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-150034_edge_update_extend200_fusion360seg_xse_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `fc58cd5657790ea07b06839d6c19fd063b5bbac1ca224f16dfa2f365d75cbabe`
- MFCAD++ / MLP：续训 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_mlp_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_mlp_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `881d6ab776c31df3778903b97681c8b2c0c68def13458a9c6e3ae056d1bc0f6f`
- MFCAD++ / DiffLoss x_start：续训 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xstart_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xstart_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `9534e82dee8b1c9a08728490b168738e17b835447d662881152a8a1167786bc7`
- MFCAD++ / DiffLoss x_start+0.5eps：续训 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xse_20260810-seg200-titan`；全局选择 `runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xse_20260810-seg200-titan/checkpoints/best.pt`；SHA-256 `5c67ed0802778d56c4c1cd597e4be3621eaeec7dd355b9328e3698d568c1a38f`

说明：旧 checkpoint 未保存 RNG 状态，因此续训会恢复模型和 optimizer，但不会恢复 epoch 100 末尾的随机数流；这不是一次从 epoch 1 不间断运行到 200 的完全等价复现。
