# 全量预训练 Linear probe 与目标排除预训练消融

按用户澄清执行两条独立实验线。目标任务为 Fusion360Seg（8 类面分割）、Blend / BRepPreDiff（3 类面分割）、TMCAD（10 类整模型分类）。

| 实验线 | GPU | 预训练初始化 | 下游训练 | 顺序 |
|---|---:|---|---|---|
| 全量预训练 Linear probe | 2 | 复用完整 Inductive9、100-epoch 编码器 | 冻结全部 encoder，仅训练单个 Linear，200 epochs | Fusion360Seg → Blend → TMCAD |
| Leave-one-dataset-out | 1（Fusion360Seg/Blend）、0（TMCAD） | 每个目标单独排除后，从随机初始化预训练 100 epochs | 全量微调 MLP，200 epochs | Fusion360Seg 预训练/微调/测试 → Blend → TMCAD |

这里的全量预训练沿用现有完整 Inductive9（九个来源的 train-only 无标签几何，不增加 val/test）；三个 downstream 均使用各自完整 train split，不做 few-shot。两条线独立，GPU 2 不依赖 GPU 1。总计新增 3 次预训练、3 次全量微调、3 次纯线性 probe，不增加随机 probe。

## 排除协议

- Fusion360Seg 排除 `fusion360seg_s2_0_1` 整个预训练来源；Blend 排除 `brepprediff` 来源；TMCAD 排除 `tmcad` 来源。Blend 预训练池与其下游不完全相同，这是保守的来源整体排除。
- 对各自目标 train/val/test 的几何缓存生成禁止指纹集合，进一步删除其余来源的精确重复样本；不使用标签或测试指标。覆盖范围强于只删除目标 train。
- SHA-256 对 `face_cont, face_surface_type, edge_index, edge_cont, edge_type, edge_relation` 的名称、shape、dtype、原始数组字节计算，排除标签。
- 该审计识别完全相同的几何缓存，不能排除旋转、面/边重编号、重参数化或不同精度提取后的等价 CAD。不能将“精确重叠为零”表述成几何等价体绝对不存在。
- `exclusion_audit.json` 记录来源级原始/保留/移除样本数和剩余精确重叠数；`splits/` 固化每次实际训练使用的缓存清单。原始数据不改动。
- LODO 不使用已有全量预训练权重初始化；重启仅可恢复本实验自身的 checkpoint。

## 模型与协议

复制 `inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final` 实际保存配置：完整 EUA，128 hidden、4 layers、4 heads，seed 42、50% SO(3) augmentation，原有下游 split、batch/累积、损失、学习率及 validation accuracy 选优规则。预训练为 100 epochs、batch 128、LR 1e-4、constant scheduler；下游统一 200 epochs。所有下游均以 validation 最佳 checkpoint 在 test 评估，不根据 test 选模型。

分割 probe：`Linear(128, C)`。TMCAD 保留配置 `graph_pooling: mean_max`，但严格线性 probe 采用固定 `[mean(h); max(h)]` 拼接后 `Linear(256, 10)`，不训练 pooling 内的非线性投影。原 MLP/DiffLoss 的 Mean+Max 和旧配置默认 mean 行为不改变。

冻结方式：所有 encoder 参数 `requires_grad=False`，训练期间 encoder.eval() 关闭 dropout。仅线性权重与偏置可训练。三项 probe 使用同一完整 Inductive9 预训练 checkpoint，不使用微调过的 encoder。

GPU 使用项目共享锁，并等待空闲显存不少于 50 GB，不终止其他实验。

## 结果解释

- 全量预训练 probe 测量完整预训练表示的线性可分性；预训练含目标来源几何，因此它本身不证明对未见数据集的迁移。
- LODO 微调测量未见目标来源时的迁移能力，与现有完整预训练微调基线比较。
- 二者共同提供不同角度证据；本轮没有 LODO probe 或随机冻结 probe，不能据此单独声称“未见目标数据集的冻结表示显著优于随机表示”。
- 删除来源同时减少预训练语料量，不属于等样本量或等 optimizer-step 控制。单 seed 42 不提供统计显著性结论。
- 指标为 Accuracy、Macro-F1、mIoU；TMCAD 以 Accuracy/Macro-F1 为主，两个分割任务同时关注 mIoU。

## 产物及启动

根目录：`runs/lodo_transfer_pre100_ft200_seed42_20260918/`。

`probe_manifest.json` 为 GPU 2 的 3 个 probe；`manifest.json` 为 GPU 1 的 6 个预训练/微调阶段。`configs/` 固定配置，`source/` 源码快照，`*provenance.json` 保存依赖哈希，`state/` 与 `logs/` 保存状态及日志，`results/` 保存 test 指标。

```bash
PYTHONPATH=src conda run --no-capture-output -n blendit python scripts/run_lodo_transfer.py --prepare-probes
conda run --no-capture-output -n blendit python scripts/run_lodo_transfer.py --worker probe
PYTHONPATH=src conda run --no-capture-output -n blendit python scripts/run_lodo_transfer.py --prepare-only
conda run --no-capture-output -n blendit python scripts/run_lodo_transfer.py --worker finetune
```

验证：linear-head/probability 路由、只有 Linear 参数更新、冻结参数保持不变、checkpoint strict round-trip，以及原分类和 encoder freezing 测试，共 23 项通过；另有 1 项几何指纹标签无关性测试通过。实际 Fusion360Seg probe 第 34 epoch 最佳 checkpoint 的 123 个 encoder 张量与预训练权重逐位一致，只有 1,032 个线性头参数。

## 排除审计结果

原始预训练样本数 488,099。

| 排除目标 | 保留样本 | 总移除 | 整体来源外额外精确重复 | 剩余精确重叠 |
|---|---:|---:|---:|---:|
| Fusion360Seg | 463,125 | 24,974 | 10 | 0 |
| Blend | 315,808 | 172,291 | 4 | 0 |
| TMCAD | 479,390 | 8,709 | 0 | 0 |

自动汇总：[运行状态和测试结果](transfer_ablation_results_20260918.md)。

## GPU 调度更新

按用户要求，TMCAD 的 LODO 预训练 → 微调 → test 已单独移至 GPU 0；GPU 1 仅继续 Fusion360Seg、Blend。GPU 2 的三个完整预训练 linear probe 不变。GPU 1 现有预训练子进程原样接管，不重新开始。训练配置和排除划分没有更改。

恢复调度时使用：

```bash
conda run --no-capture-output -n blendit python scripts/run_transfer_gpu_queue.py --gpu 0 --targets tmcad
conda run --no-capture-output -n blendit python scripts/run_transfer_gpu_queue.py --gpu 1 --targets fusion360seg blend
```

不要再使用旧的 GPU 1 全三任务队列命令，以免重复调度 TMCAD。
