# 离散预训练损失 Encoder 下游消融（2026-08-20）

## 结论

在本次单 seed、严格配对的下游 MLP 微调中，将预训练离散损失权重置零没有造成一致的性能下降：

- TMCAD 分类明显提升：test accuracy `+1.3799 pp`，Macro-F1 `+1.6151 pp`。
- FabWave min10 分类完全持平。
- 三个分割数据集变化很小且方向不一致；按 Macro-F1，BRepPreDiff `+0.0135 pp`、Fusion360Seg `-0.2844 pp`、MFCAD++ `+0.0015 pp`。

因此，本次结果不支持“离散预训练损失对所有下游任务稳定有益”。较可信的信号是 TMCAD 上关闭离散损失更好；其余数据集基本持平或存在轻微权衡。由于只运行了 seed 42，不能把小于约 `0.3 pp` 的差异解释为稳定收益。

## 严格控制变量

两个预训练 run 的配置逐行比较后，只有以下两项不同：

| 配置 | 离散损失开启 | 离散损失关闭 |
|---|---:|---:|
| `diffusion.categorical_loss_weight` | 0.5 | 0.0 |
| `diffusion.relation_loss_weight` | 0.3 | 0.0 |

其余预训练数据、split、seed 42、Edge Update Attention encoder、4 layers、hidden dim 128、4 heads、batch size 128、学习率、归一化及 150 epochs 均相同。

- 开启 checkpoint：`runs/pretrain/20260813-183150_joint_fusion_gallery_mlp_all_splits/checkpoints/last.pt`
  - SHA-256：`2d2bd167d8076582d6623485809414aa2f1850f41c3aff9cab237e042bf7bbc8`
- 关闭 checkpoint：`runs/pretrain/20260814-160525_joint_fusion_gallery_mlp_all_splits/checkpoints/last.pt`
  - SHA-256：`e02c70190badeee10333dc4b7c48cc5ecc7a548229436e93d7a7ce4a7adf51ce`

下游每一对实验均使用同一配置、数据 split 和缓存，只替换 encoder checkpoint。统一设置如下：

- MLP head，全 encoder 微调；
- seed 42、dataloader seed 42；
- batch size 64、gradient accumulation 4；
- 200 epochs，按 validation accuracy 选择 best checkpoint；
- 分类任务使用 Mean+Max pooling；
- 使用 best checkpoint 在官方 test split 上独立评估。

## Test 结果

差值均为“离散损失关闭 − 开启”，单位为百分点（pp）。

| 数据集 | 任务 | 指标 | 开启 | 关闭 | 差值 |
|---|---|---|---:|---:|---:|
| BRepPreDiff | 分割 | Accuracy | 98.2609% | 98.2199% | -0.0410 |
|  |  | Macro-F1 | 96.6414% | 96.6550% | +0.0135 |
|  |  | Macro-IoU | 93.5942% | 93.6236% | +0.0294 |
| Fusion360Seg | 分割 | Accuracy | 92.9363% | 93.2620% | +0.3257 |
|  |  | Macro-F1 | 87.8317% | 87.5473% | -0.2844 |
|  |  | Macro-IoU | 79.0917% | 78.9521% | -0.1396 |
| MFCAD++ | 分割 | Accuracy | 99.3691% | 99.3751% | +0.0059 |
|  |  | Macro-F1 | 99.0234% | 99.0248% | +0.0015 |
|  |  | Macro-IoU | 98.0804% | 98.0837% | +0.0033 |
| TMCAD | 分类 | Accuracy | 82.7967% | 84.1766% | **+1.3799** |
|  |  | Macro-F1 | 82.2865% | 83.9016% | **+1.6151** |
| FabWave min10 | 分类 | Accuracy | 97.9540% | 97.9540% | 0.0000 |
|  |  | Macro-F1 | 99.4929% | 99.4929% | 0.0000 |

对应 best epochs：

| 数据集 | 开启 | 关闭 |
|---|---:|---:|
| BRepPreDiff | 74 | 16 |
| Fusion360Seg | 196 | 153 |
| MFCAD++ | 130 | 195 |
| TMCAD | 21 | 173 |
| FabWave min10 | 16 | 7 |

## MFCAD++ 补跑说明

原始“离散损失关闭”MFCAD++ run 在 epoch 121 开始时中断，已有 epoch 120 checkpoint。2026-08-20 从该 checkpoint 恢复优化器与模型状态并完成 epoch 121–200，最终 validation 最佳为 epoch 195：

- run：`runs/discrete_loss_encoder_ablation/finetune/20260820-113631_discrete_loss_discrete_off_mfcadpp_seg_resume120_legacy_features_20260820`
- best checkpoint SHA-256：`e8b69ccd540b17efebc070d10742d6189c2dd61d628efeac829a2c4cd6c5401f`
- test JSON：该 run 下的 `test_metrics.json`

原 611/3 特征缓存后来被 711/63 OCC v2 缓存覆盖。新缓存的前 6 个 face-grid 通道与旧 `[xyz, normal]` 完全相同，只额外插入 trim mask；edge 前 3 个通道也与旧缓存相同。因此补跑通过显式 `brep.feature_schema: legacy` 投影删除 mask、截取 edge 前 3 列，恢复旧模型输入，并用单元测试验证通道逐元素保持。

限制：checkpoint 保存模型和 optimizer，但不保存 RNG 状态。因此恢复后 epoch 121–200 的 minibatch 随机序列不能逐位复原为原本不中断时的序列。所有可配置变量、seed、数据、输入通道、优化器状态和训练预算均保持一致；该限制只影响对“逐 batch 完全确定性”的声明。

