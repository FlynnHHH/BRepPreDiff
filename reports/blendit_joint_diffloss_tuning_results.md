# Blendit 联合无 coarse-label 基模 DiffLoss 调参结果

生成日期：2026-07-24

## 结论

使用 `all_splits_no_coarse` 联合自监督预训练基模，在同一 Blendit
train/validation/test 划分上，调优后的 DiffLoss 已超过现有 MLP 基线。

| Head | Accuracy | Macro-F1 | Macro-IoU | Weighted-F1 | Weighted-IoU |
|---|---:|---:|---:|---:|---:|
| MLP | 0.981403 | 0.963957 | 0.931435 | 0.981600 | 0.964574 |
| DiffLoss（最终） | **0.982055** | **0.965953** | **0.935136** | **0.982190** | **0.965671** |
| DiffLoss - MLP | **+0.065 pp** | **+0.200 pp** | **+0.370 pp** | **+0.059 pp** | **+0.110 pp** |

最终 DiffLoss 的三个类别 F1 均高于 MLP：

| 类别 | MLP F1 | DiffLoss F1 | 差值 |
|---|---:|---:|---:|
| NonTransition | 0.990183 | **0.990414** | +0.023 pp |
| VBF | 0.969131 | **0.973187** | +0.406 pp |
| EBF | 0.932558 | **0.934257** | +0.170 pp |

## 对照协议

- 预训练 checkpoint：
  `runs/pretrain/20260723-213828_joint_all_splits_no_coarse/checkpoints/last.pt`
- Blendit 数据：6,127 train / 766 validation / 766 test；
- 测试集：41,459 faces；
- MLP 对照：
  `runs/finetune/20260723-232008_joint_blendit_mlp/checkpoints/best.pt`，
  best epoch 21；
- DiffLoss 与 MLP 使用相同 encoder 架构、相同预训练 checkpoint、相同数据划分；
- 所有训练 checkpoint 均按 validation Macro-F1 选择；
- 推理参数和 score bias 只在 validation 上搜索；只有超过已有 validation
  门槛的训练变体才进入 test。

## 最终 DiffLoss 参数

| 参数 | 值 |
|---|---|
| Prediction type | `x_start` |
| Train timesteps / schedule | 1,000 / cosine |
| Head width / depth / dropout | 128 / 3 / 0.0 |
| Noise samples per token | 4 |
| Encoder / head learning rate | `3e-4` / `3e-4` |
| Weight decay | `1e-4` |
| Batch size / epochs | 256 / 100 |
| Class weights | `[0.67, 1.37, 0.96]` |
| DDIM sampling steps | 1 |
| Sampling temperature | 0.75 |
| Inference samples | 1 |
| Class score bias | `[0.75, 0.0, 0.0]` |

最终推荐配置为
`configs/finetune_joint_blendit_diffloss.yaml`。已验证 checkpoint 来自：

`runs/finetune/20260724-003104_joint_blendit_diff_s2_xstart_mildcw/checkpoints/best.pt`

该 checkpoint 为 epoch 99；其训练时保存的原始配置仍完整保存在对应 run
目录。最终的 1-step、temperature 和 class-score bias 通过评估命令覆盖，见下方
复现命令。

## 搜索过程

第一轮固定同一预训练基模并并行比较预测目标、head 容量和学习率；明显落后的
分支提前剪枝。第二轮比较类别权重、噪声重复数和 encoder/head 分离学习率。

| 变体 | 状态 / Best epoch | 原始 Val Macro-F1 | 调优推理 Val Macro-F1 | Test Macro-F1 |
|---|---:|---:|---:|---:|
| `x_start_epsilon`, epsilon weight 0.5 | epoch 26 剪枝 | 0.937060 | — | — |
| `x_start`, 128×3, lr `3e-4` | 100 / best 86 | 0.948600 | 0.950729 | 0.964303 |
| `x_start`, 256×4 | epoch 46 剪枝 | 0.945780 | — | — |
| `x_start`, lr `1e-4` | epoch 66 剪枝 | 0.944190 | — | — |
| `x_start` + mild class weights | 100 / best 99 | **0.949450** | **0.951351** | **0.965953** |
| `x_start`, noise samples 8 | 100 / best 93 | 0.949330 | 0.950959 | 未过门槛 |
| `x_start`, encoder/head lr `1e-4/3e-4` | 100 / best 91 | 0.947010 | 未过门槛 | — |

### 推理参数消融

以下结果使用无类别权重的 `x_start` best checkpoint，全部在 validation 上计算：

| Sampling | Val Macro-F1 |
|---|---:|
| 25 steps, temperature 1.0 | 0.948600 |
| 5 steps, temperature 1.0 | 0.948804 |
| 2 steps, temperature 0.75 | 0.949775 |
| 1 step, temperature 1.0 | 0.949961 |
| 1 step, temperature 0.5 | 0.950191 |
| 1 step, temperature 0.75 | 0.950258 |
| 1 step, temperature 0.75, 4-sample ensemble | 0.949983 |
| 上一行单样本 + NonTransition score bias 0.75 | **0.950729** |

多步 DDIM 在这个 `x_start` 分类任务上会累积误差；直接在最高噪声时刻进行一次
条件预测更有效，也显著降低推理成本。4-sample 集成没有改善指标。

## 最终逐类结果

最终 DiffLoss confusion matrix（行是真值，列是预测）：

| GT / Pred | NonTransition | VBF | EBF |
|---|---:|---:|---:|
| NonTransition | 33,733 | 2 | 431 |
| VBF | 2 | 1,724 | 36 |
| EBF | 218 | 55 | 5,258 |

| 类别 | Support | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| NonTransition | 34,166 | 0.993520 | 0.987327 | 0.990414 | 0.981010 |
| VBF | 1,762 | 0.967996 | 0.978434 | 0.973187 | 0.947774 |
| EBF | 5,531 | 0.918428 | 0.950642 | 0.934257 | 0.876626 |

Transition binary F1 为 `0.955875`，MLP 为 `0.955089`。

## 实现改动

- 新增联合基模 Blendit DiffLoss 配置；
- CLI override 支持 list/dict，便于搜索 class weights 和 score bias；
- fine-tune optimizer 支持可选的 encoder/head 分离学习率；
- DiffLoss 推理支持逐类 `class_score_bias`；
- 对上述功能增加单元测试，默认配置行为保持不变。

## 产物与复现

| 产物 | SHA-256 |
|---|---|
| Pretrain `last.pt` | `1689ab00fe1d9e110b6a38f052a773a241b2765dd7c5addee1cef28fc8e3d464` |
| MLP `best.pt` | `f1a234cbe402eb788c8eaafd84e1c641f6d755dabd325a6864c99a3b3f5c4bb3` |
| 最终 DiffLoss `best.pt` | `9457620dc0358dc4a574e1cda994aac721cb3c86f1be5d17cb6924cbdb2b5abc` |

最终 test 指标文件：

`runs/finetune/20260724-003104_joint_blendit_diff_s2_xstart_mildcw/test_tuned_inference.json`

精确复现最终 checkpoint 的 test 评估：

```bash
CUDA_VISIBLE_DEVICES=0 /home/hhfeng/miniconda3/envs/blendit/bin/python \
  -m blendit.training.evaluate \
  --checkpoint runs/finetune/20260724-003104_joint_blendit_diff_s2_xstart_mildcw/checkpoints/best.pt \
  --split test \
  --output runs/finetune/20260724-003104_joint_blendit_diff_s2_xstart_mildcw/test_tuned_inference.json \
  --batch-size 512 \
  --num-workers 8 \
  --device cuda \
  --override label_diffusion.sampling_steps=1 \
  --override label_diffusion.sampling_temperature=0.75 \
  --override label_diffusion.class_score_bias=[0.75,0.0,0.0]
```

## 解释边界

本结果证明在当前固定数据协议和固定 MLP 基线下，经过 validation 调参的 DiffLoss
可以取得更高 test 指标。DiffLoss 获得了比 MLP 更多的超参数搜索预算，因此这不是
对两个 head 进行完全对称预算的算法比较；它回答的是“能否把该基模上的 DiffLoss
调到超过现有 MLP”这一问题。调参过程中曾查看一次未校准 DiffLoss test，后续
score bias 和类别权重门槛仍只依据 validation 选择，报告保留这一过程以避免把结果
描述成一次性盲测。
