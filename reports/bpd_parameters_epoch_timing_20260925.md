# BPD 参数量与 epoch 用时（2026-09-25）

按 BRepPreDiff 默认预训练模型统计：edge_update_attention，hidden_dim=128，4 层，4 heads，面/边输入维度 711/63。CPU 实例化当前代码，直接累加 Parameter.numel()，不计 buffer、梯度及优化器状态。未修改已有模型代码或训练配置。

总参数：**2,249,696**；全部可训练。Encoder：**1,826,832**；预训练其他模块：**422,864**。batch size 不影响参数量；下游模型需按具体 head 和类别数另计。

| 模块 | 参数量 |
|---|---:|
| time_mlp | 33,024 |
| encoder | 1,826,832 |
| face_noise_head | 108,231 |
| face_recon_head | 108,231 |
| surface_head | 20,640 |
| edge_context | 65,792 |
| edge_noise_head | 24,639 |
| edge_recon_head | 24,639 |
| edge_type_head | 20,640 |
| relation_head | 17,028 |

## 历史实测耗时

| 每卡 batch | 卡数 | 全局 batch | 数据 | 纯训练时间/epoch |
|---:|---:|---:|---|---|
| 32 | 4 | 128 | 旧七源预训练 | checkpoint 保留，耗时日志缺失 |
| 64 | 4 | 256 | 旧七源预训练 | checkpoint 保留，耗时日志缺失 |
| 128 | 1 | 128 | 九源 train-only，488,099 个样本 | 242.62 ± 7.76 秒 |

batch 128 来自单卡 NVIDIA A800-SXM4-80GB 的完整 100 epoch 历史运行；硬件类型由该 run 的 wandb-metadata.json 确认，实际卡数由训练日志 world_size=1 确认。每轮 3,814 batches，num_workers=16。剔除首轮后统计 epoch 2–100，共 99 轮。首轮 244 秒；中位数 241 秒；范围 236–314 秒。标准差为样本标准差。

时间为日志 `split=train start` 到对应训练指标行之差，日志精度为秒，包含数据加载及训练循环，不包含启动、建模、验证、checkpoint 保存。该 run 未启用验证。CSV 同时提供相邻 epoch 开始时间差，以包含轮间保存和日志开销；最后一轮没有该值。此为历史墙钟时间，不能排除共享服务器负载影响。

来源：`runs/inductive_lr1e4_encoder/pretrain/20260901-115311_inductive9_no_fab_encoder_lr1e4_e100_seed42_20260901/logs/pretrain.log`。

旧 BS32/64 的配置证据见 `reports/encoder_pretrain_batchsize_downstream_comparison_2026-08-26.md`。当前配置默认四卡、每卡 64，但不可将这里的单卡 batch 128 耗时套用到默认四卡配置。九源与七源数据规模、卡数不同，不构成 batch size 速度消融。

## 缺失与限制

当前环境没有 `/dev/nvidia*`，`nvidia-smi` 无法连接驱动，因此未补跑 batch 16/32/64/128/256 的受控 GPU 测速，也未根据 batch size 线性外推耗时。现有 BS32 单卡日志仅有第一轮开始记录，不能计算完整 epoch 时间。不同 batch size 的完整对照仍需在 GPU 环境、固定数据和设备条件下实测。
