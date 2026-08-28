# Edge Update Attention + Mean+Max 分类实验：TMCAD 与 FabWave

> 实验日期：2026-08-11  
> 硬件：NVIDIA TITAN RTX，TMCAD 使用物理 GPU 1，FabWave 使用物理 GPU 2  
> 随机种子：模型 seed 42、DataLoader seed 42  
> 每组 200 epochs，按 validation Accuracy 选择 `best.pt`，最后只在正式 test split 上评估该 checkpoint

## 1. 实验目的

在项目默认的 **Mean+Max graph pooling + MLP 分类头**不变时，将先前分类实验的
4-layer B-Rep FFN baseline encoder 换为 4-layer、4-head **Edge Update Attention encoder**，
重新评估 TMCAD 10 类和 FabWave min10 40 类识别任务。

Edge Update Attention 的预训练权重必须与其架构匹配。本次使用 333,476 图七来源联合无标签预训练
checkpoint（150 epochs），SHA-256：
`647be213de85ea393e94ec96e6fb33f4b4c87ed2e6709f8f8d041d7ba64c89ae`。
全部 123 个 encoder 张量均成功加载，`skipped=0`。

需要注意：作为参照的 baseline encoder + Mean+Max v3 使用另一个架构匹配 checkpoint，
其七来源预训练语料包含 334,036 图，SHA-256
`f9e39e2a22d65d50318f031ce1cdf76c6a399ce41f7eb25d05f9523a067c52b9`。
因此下述差值是 **encoder + 对应预训练 checkpoint 的架构实验**，不是只改变 encoder 参数的严格单变量消融。

## 2. 下游协议

- Encoder：4-layer Edge Update Attention，hidden dim 128，4 heads，full fine-tuning。
- Pooling / head：Mean+Max / MLP。
- 优化：AdamW，lr `3e-4`，weight decay `1e-4`，200 epochs。
- Batch：micro-batch 64、梯度累积 4，等效 batch 256；测试 batch 64。
- 数据：TMCAD 8,709/1,090/1,087；FabWave min10 3,191/407/391。
- 归一化：per-graph feature normalization。
- 选模：每个 epoch 验证，按最高 validation Accuracy 保存 `best.pt`。

联合无标签预训练包含各下游 split 的几何，因此属于 transductive self-supervised 协议。

## 3. 正式测试结果

| Dataset | GPU | Params | Best epoch | Best val Acc | Test samples | Test Acc | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TMCAD（10 类） | 1 | 1,874,074 | 141 | 85.3211% | 1,087 | **86.1086%** | 85.7738% | 85.9091% | **85.7531%** | 86.0946% | **75.6592%** |
| FabWave min10（40 类） | 2 | 1,877,944 | 29 | 97.0516% | 391 | **97.9540%** | 99.5556% | 99.5238% | **99.4929%** | 97.9461% | **99.0794%** |

TMCAD 正确识别 936/1,087 个 test 模型；FabWave 正确识别 383/391 个 test 模型。

## 4. 与 baseline encoder + Mean+Max v3 对比

| Dataset | Encoder | Params | Test Acc | Macro-F1 | Weighted-F1 | mIoU |
|---|---|---:|---:|---:|---:|---:|
| TMCAD | 4-layer FFN baseline | 882,314 | 85.5566% | 85.2855% | 85.5820% | 75.0315% |
| TMCAD | **Edge Update Attention** | 1,874,074 | **86.1086%** | **85.7531%** | **86.0946%** | **75.6592%** |
| FabWave min10 | 4-layer FFN baseline | 886,184 | 97.9540% | 99.4929% | 97.9461% | 99.0794% |
| FabWave min10 | **Edge Update Attention** | 1,877,944 | **97.9540%** | **99.4929%** | **97.9461%** | **99.0794%** |

- **TMCAD**：Edge Update 架构实验相对 baseline encoder + Mean+Max 提高
  **0.5520 pp Acc**、**0.4676 pp Macro-F1**、**0.5127 pp Weighted-F1** 和
  **0.6277 pp mIoU**；参数量增加 **112.40%**（约 2.124×）。
- **FabWave min10**：全部 test 指标完全持平；参数量增加 **111.91%**（约 2.119×），
  没有观察到额外识别收益。

## 5. 结论

- TMCAD 上，Edge Update Attention + Mean+Max 得到新的单种子最佳结果：Test Acc
  **86.1086%**。提升幅度为 0.5520 pp，建议在决定是否升级默认 encoder 前补做多种子复验。
- FabWave min10 已接近饱和，Edge Update 没有改善 391 个 test 样本上的任何汇总指标；
  若重视效率，baseline encoder 更合适。
- 当前证据支持继续保留 **Mean+Max** 作为分类 pooling 默认值，但不足以把 Edge Update
  设为所有分类实验的默认 encoder。

## 6. 运行产物

- TMCAD：`runs/edge_update_meanmax_cls/finetune/20260811-115407_edge_update_meanmax_tmcad_20260811-115348`
  - `best.pt` epoch 141
  - SHA-256 `20b7ba265a99270d31371500f2a9defbcebdc7b7b82e10cbab810cf61ac91918`
- FabWave min10：`runs/edge_update_meanmax_cls/finetune/20260811-115407_edge_update_meanmax_fabwave_20260811-115348`
  - `best.pt` epoch 29
  - SHA-256 `1efcca26711779520379cf94eeb775072de0a4f96a820824904897a75dfde64d`
- 启动日志：`runs/edge_update_meanmax_cls/launch_logs/20260811-115348/`
- 复现入口：`scripts/run_edge_update_meanmax_cls_titan.sh`

代码验证：`112 passed`，另有 1 条 PyTorch `scatter_reduce` beta API warning。
