# Edge Update Attention：联合 x_start/epsilon DiffLoss 五项复验

> 生成日期：2026-08-10  
> 目标函数：`MSE(x_start) + 0.5 * MSE(epsilon)`（未除以权重和）  
> 标签：bipolar one-hot；cosine 1000-step；每 token 4 个训练噪声样本；1-step DDIM 测试  
> Encoder：Edge Update Attention；相同联合预训练 checkpoint；seed 42；按 validation accuracy 选择 `best.pt`
> 训练预算：全部任务 200 epochs；分割任务由 epochs 1-100 与续训 epochs 101-200 的候选共同选择全局 best

## 完整测试结果

| Benchmark | Variant | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | MLP | 18 | 766 | 98.1210 | 96.2606 | 98.1351 | 92.8949 |
| BRepPreDiff | DiffLoss x_start | 172 | 766 | 98.1452 | 96.2543 | 98.1575 | 92.8807 |
| BRepPreDiff | DiffLoss x_start+0.5eps | 37 | 766 | 98.2127 | 96.5933 | 98.2250 | 93.5098 |
| Fusion360Seg s2.0.0 | MLP | 165 | 5,366 | 93.0933 | 87.2338 | 93.0572 | 78.5003 |
| Fusion360Seg s2.0.0 | DiffLoss x_start | 172 | 5,366 | 93.0946 | 86.9494 | 93.0608 | 78.1552 |
| Fusion360Seg s2.0.0 | DiffLoss x_start+0.5eps | 168 | 5,366 | 93.2672 | 87.9302 | 93.2362 | 79.3756 |
| MFCAD++ | MLP | 172 | 8,949 | 99.3773 | 99.0433 | 99.3774 | 98.1201 |
| MFCAD++ | DiffLoss x_start | 174 | 8,949 | 99.4286 | 99.1056 | 99.4287 | 98.2397 |
| MFCAD++ | DiffLoss x_start+0.5eps | 190 | 8,949 | 99.3910 | 99.0336 | 99.3915 | 98.1001 |
| TMCAD | MLP | 192 | 1,087 | 84.9126 | 84.6192 | 84.9482 | 73.9883 |
| TMCAD | DiffLoss x_start | 173 | 1,087 | 83.9926 | 83.4995 | 83.9063 | 72.3311 |
| TMCAD | DiffLoss x_start+0.5eps | 177 | 1,087 | 83.9926 | 83.4477 | 83.8715 | 72.3906 |
| FabWave min10 | MLP | 6 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10 | DiffLoss x_start | 87 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10 | DiffLoss x_start+0.5eps | 112 | 391 | 97.9540 | 99.4929 | 97.9461 | 99.0794 |

## 新 DiffLoss 的差异

单位为百分点；正值表示联合 x_start/epsilon DiffLoss 更高。

| Benchmark | ΔAcc vs x_start | ΔMacro-F1 vs x_start | ΔmIoU vs x_start | ΔAcc vs MLP |
|---|---:|---:|---:|---:|
| BRepPreDiff | +0.0675 | +0.3390 | +0.6291 | +0.0917 |
| Fusion360Seg s2.0.0 | +0.1726 | +0.9809 | +1.2204 | +0.1739 |
| MFCAD++ | -0.0375 | -0.0720 | -0.1396 | +0.0138 |
| TMCAD | +0.0000 | -0.0518 | +0.0594 | -0.9200 |
| FabWave min10 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |

## 运行产物

- BRepPreDiff / MLP：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_brepprediff_seg_20260807-123259`；`best.pt` SHA-256 `10f8bb47a845206701a5c44272c5bb0c29e90ed473cb051baff5d8c7dc474c9d`
- BRepPreDiff / DiffLoss x_start：`runs/edge_update_new_joint/finetune/20260810-144740_edge_update_extend200_brepprediff_xstart_20260810-seg200-titan`；`best.pt` SHA-256 `cfbf7e992ff4d9c6d11ab7909e33542822a54527726da369f663a19a479262b2`
- BRepPreDiff / DiffLoss x_start+0.5eps：`runs/edge_update_new_joint/finetune/20260808-183835_edge_update_brepprediff_seg_diffloss_xse_20260808-xse-titan`；`best.pt` SHA-256 `1e1ffff9eb6b7a86b23609d91cc780ded2c5efe9aa06837ce378b4629d6c37b7`
- Fusion360Seg s2.0.0 / MLP：`runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_fusion360seg_mlp_20260810-seg200-titan`；`best.pt` SHA-256 `3fa9827d42e815a3cf532140a9149da5ada2c828cc39e61be865a2078a2b23c9`
- Fusion360Seg s2.0.0 / DiffLoss x_start：`runs/edge_update_new_joint/finetune/20260810-141509_edge_update_extend200_fusion360seg_xstart_20260810-seg200-titan`；`best.pt` SHA-256 `2f5a3b20de6c37e451b4e1c96aed1bf2c7cb6d7c9bc6f4c16fed165fbd6920e0`
- Fusion360Seg s2.0.0 / DiffLoss x_start+0.5eps：`runs/edge_update_new_joint/finetune/20260810-150034_edge_update_extend200_fusion360seg_xse_20260810-seg200-titan`；`best.pt` SHA-256 `fc58cd5657790ea07b06839d6c19fd063b5bbac1ca224f16dfa2f365d75cbabe`
- MFCAD++ / MLP：`runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_mlp_20260810-seg200-titan`；`best.pt` SHA-256 `881d6ab776c31df3778903b97681c8b2c0c68def13458a9c6e3ae056d1bc0f6f`
- MFCAD++ / DiffLoss x_start：`runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xstart_20260810-seg200-titan`；`best.pt` SHA-256 `9534e82dee8b1c9a08728490b168738e17b835447d662881152a8a1167786bc7`
- MFCAD++ / DiffLoss x_start+0.5eps：`runs/edge_update_new_joint/finetune/20260810-133433_edge_update_extend200_mfcadpp_xse_20260810-seg200-titan`；`best.pt` SHA-256 `5c67ed0802778d56c4c1cd597e4be3621eaeec7dd355b9328e3698d568c1a38f`
- TMCAD / MLP：`runs/edge_update_new_joint/finetune/20260808-163205_edge_update_tmcad_cls_mlp_20260808-head-complements-titan`；`best.pt` SHA-256 `e11e379ecd925bc1881c652667dd52eccfa67eca47841ce9b850f36e07cc2871`
- TMCAD / DiffLoss x_start：`runs/edge_update_new_joint/finetune/20260807-200941_edge_update_tmcad_cls_20260807-123259`；`best.pt` SHA-256 `0abb8b936a4dd76f79296980db1d8ed510b6fe94924d481b22ff20941bf7ccfa`
- TMCAD / DiffLoss x_start+0.5eps：`runs/edge_update_new_joint/finetune/20260808-183835_edge_update_tmcad_cls_diffloss_xse_20260808-xse-titan`；`best.pt` SHA-256 `f5c39f7f3a49b144d121aa8378694875717ad512cbf9383af371a5c2cd7a407b`
- FabWave min10 / MLP：`runs/edge_update_new_joint/finetune/20260808-174658_edge_update_fabwave_cls_mlp_20260808-head-complements-titan`；`best.pt` SHA-256 `d2776a349a44a78531211ebd9d75333d69b6dd9135e90c6f49db13c4760c08e8`
- FabWave min10 / DiffLoss x_start：`runs/edge_update_new_joint/finetune/20260807-211612_edge_update_fabwave_cls_20260807-123259`；`best.pt` SHA-256 `95c439f3ca66a7e490d7cb6838ca4127e6c8626573c5407bda1612eca15fdff8`
- FabWave min10 / DiffLoss x_start+0.5eps：`runs/edge_update_new_joint/finetune/20260808-195219_edge_update_fabwave_cls_diffloss_xse_20260808-xse-titan`；`best.pt` SHA-256 `feaf822d47eb2342145c42102600b8aa32238c34dfd5a7a14211c23af443c5a4`
