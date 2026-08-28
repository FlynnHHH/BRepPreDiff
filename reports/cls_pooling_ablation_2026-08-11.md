# CLS 面特征聚合改进：TMCAD 与 FabWave 实验报告

> 生成日期：2026-08-11  
> 硬件：3 × NVIDIA TITAN RTX（物理 GPU 1、2、3）  
> 随机种子：42；每组 200 epochs；按 validation accuracy 选择 `best.pt`  
> 下表均为训练期间未参与优化与选模的正式 test split

## 1. 改进动机与实现

当前 CLS 路径把一个 CAD 图内的所有 face embedding 做简单均值，能够稳定表示整体，但会丢失极值、离散程度和少量关键面的信息。本次只改图级 pooling，保持 Encoder、预训练权重、MLP 分类头、数据划分和训练超参数一致。

- **Mean + Max**：拼接均值与逐通道最大值，用小型残差 MLP 融合；面向少量判别性面被均值稀释的问题。
- **Mean + Std**：拼接均值与逐通道标准差，用残差 MLP 融合；显式保留一个模型内部面表示的异质性。
- **Residual Attention**：学习 face 权重，但不直接替换 mean；输出为 mean 与 attention readout 的门控残差组合，score 零初始化、初始输出严格等价于 mean，以缓解旧版纯 attention pooling 的权重塌缩。

Mean + Max 和 Mean + Std 的末层也使用零初始化，因此三种改进都从相同的 mean 表示出发。所有方法输出仍为 128 维，可同时用于 MLP 和 DiffLoss 分类头；本次用 MLP 头隔离 pooling 变量。

## 2. 实验协议

- Encoder：4-layer B-Rep FFN baseline，hidden dim 128。
- 初始化：同一联合无标签预训练 checkpoint；SHA-256 `f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9`。
- 优化：AdamW，lr `3e-4`，weight decay `1e-4`，batch size 256，full fine-tuning。
- 数据：TMCAD train/val/test = 8,709/1,090/1,087；FabWave min10 = 3,191/407/391。
- 归一化：per-graph feature normalization；模型 seed 42；独立 DataLoader seed 42，保证各方法每个 epoch 的样本顺序一致。

联合无标签预训练包含下游 split 的几何，因此属于 transductive self-supervised 协议；结果适合本仓库内公平消融，不应直接当作严格 inductive benchmark。

## 3. 完整测试结果

| Dataset | Pooling | GPU | Params | Best epoch | Best val Acc (%) | Samples | Test Acc (%) | ΔTest Acc vs Mean (pp) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TMCAD（10 类） | Mean（基线） | 1 | 832,394 | 64 | 84.4950 | 1,087 | 82.6127 | +0.0000 | 82.2055 | 82.3539 | 82.1251 | 82.5222 | 70.4983 |
| TMCAD（10 类） | Mean + Max | 2 | 882,314 | 105 | 85.1380 | 1,087 | 85.5566 | +2.9439 | 85.5094 | 85.2658 | 85.2855 | 85.5820 | 75.0315 |
| TMCAD（10 类） | Mean + Std | 3 | 882,314 | 126 | 84.7710 | 1,087 | 83.7167 | +1.1040 | 83.3673 | 83.4308 | 83.2798 | 83.6900 | 72.2585 |
| TMCAD（10 类） | Residual Attention | 2 | 849,292 | 105 | 85.2290 | 1,087 | 83.3487 | +0.7360 | 83.1156 | 83.0980 | 83.0489 | 83.3776 | 71.8193 |
| FabWave min10（40 类） | Mean（基线） | 1 | 836,264 | 16 | 96.8060 | 391 | 97.9540 | +0.0000 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10（40 类） | Mean + Max | 1 | 886,184 | 10 | 96.8060 | 391 | 97.9540 | +0.0000 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10（40 类） | Mean + Std | 3 | 886,184 | 10 | 96.8060 | 391 | 97.9540 | +0.0000 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |
| FabWave min10（40 类） | Residual Attention | 3 | 853,162 | 16 | 96.8060 | 391 | 97.9540 | +0.0000 | 99.5556 | 99.5238 | 99.4929 | 97.9461 | 99.0794 |

## 4. 结论

- **TMCAD（10 类）**：Mean + Max 最优，Test Accuracy 85.5566%，相对 Mean +2.9439 pp；Macro-F1 +3.1604 pp，mIoU +4.5331 pp。参数量增加 6.00%。
- **FabWave min10（40 类）**：Mean（基线）、Mean + Max、Mean + Std、Residual Attention 的全部 test 指标完全并列，Accuracy 均为 97.9540%。新增 pooling 没有产生可见收益，因此应保留参数更少的 Mean。

这是单随机种子消融。小于 1 pp 的差异应视为候选信号；若要升级默认配置，建议对领先方法补做至少 3 个 seed，并报告均值和标准差。

## 5. 运行产物

- TMCAD（10 类） / Mean（基线）：`runs/cls_pooling_ablation/finetune/20260810-215528_pooling_tmcad_mean_20260810-pooling-v3`；`best.pt` SHA-256 `83b7dc34d86999f90ab1e679475ce7996b096d8c34890cf19ecc4c62c1c24c68`
- TMCAD（10 类） / Mean + Max：`runs/cls_pooling_ablation/finetune/20260810-215528_pooling_tmcad_mean_max_20260810-pooling-v3`；`best.pt` SHA-256 `f578f3bd01ead49491cd8ed7af011ad5882cdfb125b3198e63d54a7b28040256`
- TMCAD（10 类） / Mean + Std：`runs/cls_pooling_ablation/finetune/20260810-215528_pooling_tmcad_mean_std_20260810-pooling-v3`；`best.pt` SHA-256 `bfe15c28316563e284c7284744c27f227e821f0ca78fb2fca183de5a2617371f`
- TMCAD（10 类） / Residual Attention：`runs/cls_pooling_ablation/finetune/20260810-231059_pooling_tmcad_residual_attention_20260810-pooling-v3`；`best.pt` SHA-256 `36af5cccb5dcbcb1f5c743e1786f73966bd871fee37f0df26638cafd78fabbc5`
- FabWave min10（40 类） / Mean（基线）：`runs/cls_pooling_ablation/finetune/20260810-231035_pooling_fabwave_mean_20260810-pooling-v3`；`best.pt` SHA-256 `33bfbb3b473ffbb2703d21913e9b11302f25e908ffcad759c0a3b237ad205476`
- FabWave min10（40 类） / Mean + Max：`runs/cls_pooling_ablation/finetune/20260810-233459_pooling_fabwave_mean_max_20260810-pooling-v3`；`best.pt` SHA-256 `7a7133892f898a0be52867d033a824ab810dac01fd69251fb20499b33be66a03`
- FabWave min10（40 类） / Mean + Std：`runs/cls_pooling_ablation/finetune/20260810-231136_pooling_fabwave_mean_std_20260810-pooling-v3`；`best.pt` SHA-256 `5ab1ffe3b5575342d259b8d33ed21e1d5048aadf5ca84a0b6c6f1c9a688c6efc`
- FabWave min10（40 类） / Residual Attention：`runs/cls_pooling_ablation/finetune/20260810-233500_pooling_fabwave_residual_attention_20260810-pooling-v3`；`best.pt` SHA-256 `4fc0199b0c563248202f8b3043e6b6720561a7992858c3736dee3b905c59d73d`

复现入口：`RUN_TAG=20260810-pooling-v3 scripts/run_cls_pooling_ablation_titan.sh`。
