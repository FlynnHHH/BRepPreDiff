# Rotate-mix：仅几何输入预训练与微调

基线：`reports/inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final.md`。

预训练、微调和评估均设置 model.use_discrete_attributes=false；绕过面类型、边类型和关系类型 embedding，且不读取离散监督目标。两项离散损失权重为 0。缓存中的离散字段保留但不参与模型计算。
从基线实际 config.yaml 复制配置：预训练 100 epochs、微调 200 epochs、seed 42、50% SO(3)、原 batch/梯度累积、数据划分、Mean+Max 分类池化和验证选优规则。
GPU 1 单卡预训练，随后 GPU 0–3 共享每卡互斥锁；启动前至少 50000 MiB 空闲。保持原 batch，不采用多卡 DDP。表中差值相对 full-loss 基线。

实验目录：`/home/nvme03/hhfeng/BRepPreDiff/runs/inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910`；配置差异见 `manifest.json`，状态见 `status.json`。

| Task / Head | 状态 | Accuracy % | ΔAcc | Macro-F1 % | ΔF1 | Weighted-F1 % | ΔWF1 | mIoU % | ΔmIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain | completed | — | — | — | — | — | — | — | — |
| brepprediff_seg_mlp | completed | 98.9532 | -0.0369 | 97.3914 | +0.0227 | 98.9510 | -0.0377 | 94.9631 | +0.0446 |
| brepprediff_seg_diffloss | completed | 98.9551 | -0.0311 | 97.4254 | +0.0311 | 98.9527 | -0.0314 | 95.0267 | +0.0592 |
| fusion360seg_mlp | completed | 96.7367 | -0.1907 | 91.8004 | -0.4250 | 96.7207 | -0.1917 | 85.9125 | -0.7237 |
| fusion360seg_diffloss | completed | 96.6109 | -0.4074 | 90.5005 | -0.4518 | 96.5892 | -0.4210 | 84.3126 | -0.9347 |
| mfcadpp_seg_mlp | completed | 99.5944 | -0.0216 | 99.4065 | -0.0205 | 99.5941 | -0.0217 | 98.8271 | -0.0396 |
| mfcadpp_seg_diffloss | completed | 99.6160 | +0.0197 | 99.4206 | +0.0364 | 99.6160 | +0.0196 | 98.8540 | +0.0709 |
| tmcad_cls_mlp | completed | 85.2806 | -1.1040 | 84.9153 | -0.9697 | 85.2027 | -1.0712 | 74.4692 | -1.4705 |
| tmcad_cls_diffloss | completed | 86.6605 | +0.0920 | 86.2914 | +0.0649 | 86.6076 | +0.0264 | 76.5335 | +0.1152 |
| solidletters_cls_mlp | completed | 97.4216 | -0.0413 | 97.4737 | -0.0411 | 97.3998 | -0.0468 | 95.3147 | -0.0718 |
| solidletters_cls_diffloss | completed | 97.2102 | +0.0206 | 97.3006 | +0.0353 | 97.2113 | +0.0249 | 95.0598 | +0.1130 |
| cadsynth_seg_mlp | completed | 99.6531 | +0.0043 | 99.4942 | +0.0122 | 99.6528 | +0.0043 | 98.9951 | +0.0241 |
| cadsynth_seg_diffloss | completed | 99.6403 | -0.0215 | 99.4572 | -0.0456 | 99.6399 | -0.0215 | 98.9221 | -0.0899 |
| mfinstseg_seg_mlp | completed | 99.4961 | +0.0493 | 99.2595 | +0.1269 | 99.4961 | +0.0497 | 98.5400 | +0.2485 |
| mfinstseg_seg_diffloss | completed | 99.4574 | -0.0059 | 99.1830 | +0.0367 | 99.4572 | -0.0058 | 98.3907 | +0.0717 |

## 相对正在运行的“仅关闭离散损失”实验

| Task / Head | ΔAcc | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |
|---|---:|---:|---:|---:|
| brepprediff_seg_mlp | +0.0388 | +0.2308 | +0.0394 | +0.4313 |
| brepprediff_seg_diffloss | -0.0544 | -0.0007 | -0.0542 | +0.0008 |
| fusion360seg_mlp | -0.3477 | -0.2882 | -0.3538 | -0.5661 |
| fusion360seg_diffloss | -0.3270 | -0.6290 | -0.3273 | -0.9947 |
| mfcadpp_seg_mlp | -0.0175 | -0.0255 | -0.0176 | -0.0497 |
| mfcadpp_seg_diffloss | +0.0197 | +0.0270 | +0.0198 | +0.0526 |
| tmcad_cls_mlp | +0.3680 | +0.5764 | +0.4848 | +0.8393 |
| tmcad_cls_diffloss | -0.0920 | -0.1325 | -0.0677 | -0.1216 |
| solidletters_cls_mlp | +0.1753 | +0.1544 | +0.1527 | +0.2980 |
| solidletters_cls_diffloss | -0.1444 | -0.1156 | -0.1328 | -0.1468 |
| cadsynth_seg_mlp | -0.0057 | -0.0131 | -0.0058 | -0.0255 |
| cadsynth_seg_diffloss | -0.0186 | -0.0445 | -0.0186 | -0.0877 |
| mfinstseg_seg_mlp | +0.0241 | +0.0822 | +0.0243 | +0.1607 |
| mfinstseg_seg_diffloss | +0.0029 | +0.0558 | +0.0026 | +0.1084 |

保留连续几何中的夹角、法向等信息和图连接关系；它们可能蕴含类型线索，本实验消融的是显式离散属性输入。
保留未使用模块的初始化顺序和 checkpoint 结构，以保持其他参数的随机初始化一致；离散 embedding 不参与前向传播、无梯度。
