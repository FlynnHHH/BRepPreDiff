# BPD 预训练和下游微调参数量、epoch 用时

统计日期：2026-09-25。此报告汇总历史运行，不是重新进行的受控 batch-size 测速。

## 统计口径

- 从每个 run 的 config.yaml 在 CPU 实例化模型，参数量逐一与历史日志核对一致；不加载权重，不修改配置。
- Encoder 为 edge_update_attention，128 hidden、4 层、4 heads、711/63 输入。下游为全量微调；分类保留原配置 mean_max；全部下游运行 200 epochs。
- 每个 run 剔除其第一轮。纯训练和验证统计 epoch 2–200（预训练 2–100）；整轮统计相邻 start-to-start，最后一轮无下一轮起点，因此下游为 2–199。
- 训练时间包含数据加载、前后向和更新；训练+验证从训练开始到验证结束；整轮还含轮间 checkpoint 和记录开销。时间单位均为秒，原日志精度为秒。
- batch 指每卡图数量；有效更新 batch 约为 batch × 卡数 × 梯度累积，末尾 batch 可能不足。

## 参数量

| 数据集 | Head | 类别数 | Encoder | Head/其他 | 总参数 |
|---|---|---:|---:|---:|---:|
| 九源 train-only | pretrain |  | 1,826,832 | 422,864 | 2,249,696 |
| BRepPreDiff | mlp | 3 | 1,826,832 | 16,899 | 1,843,731 |
| BRepPreDiff | diffusion | 3 | 1,826,832 | 348,291 | 2,175,123 |
| Fusion360Seg | mlp | 8 | 1,826,832 | 17,544 | 1,844,376 |
| Fusion360Seg | diffusion | 8 | 1,826,832 | 349,576 | 2,176,408 |
| MFCAD++ | mlp | 25 | 1,826,832 | 19,737 | 1,846,569 |
| MFCAD++ | diffusion | 25 | 1,826,832 | 353,945 | 2,180,777 |
| TMCAD | mlp | 10 | 1,826,832 | 67,722 | 1,894,554 |
| TMCAD | diffusion | 10 | 1,826,832 | 400,010 | 2,226,842 |
| SolidLetters | mlp | 26 | 1,826,832 | 69,786 | 1,896,618 |
| SolidLetters | diffusion | 26 | 1,826,832 | 404,122 | 2,230,954 |
| CADSynth | mlp | 25 | 1,826,832 | 19,737 | 1,846,569 |
| CADSynth | diffusion | 25 | 1,826,832 | 353,945 | 2,180,777 |
| MFInstSeg | mlp | 25 | 1,826,832 | 19,737 | 1,846,569 |
| MFInstSeg | diffusion | 25 | 1,826,832 | 353,945 | 2,180,777 |

## 预训练

九源 488,099 个样本，单卡 A800-SXM4-80GB，batch 128，平均纯训练 **242.62 ± 7.76 秒/epoch**，整轮 **242.64 秒**。没有验证。
旧七源四卡 batch 32、64 仅保留 checkpoint，未保留可用耗时日志。不能填入实测时间，不能与九源单卡结果作为速度消融。

## 下游微调耗时

两套 cohort 分开列出，避免混合运行平均。A：inductive_lr1e4_pre100_full_ft200；B：inductive9_rotate_mix。两者预训练 checkpoint 不同，B 还使用旋转增强及梯度累积，因此不可将差异全归于 batch size。

| 数据集 | Head | 组 | Batch | 累积 | 旋转概率 | 训练样本 | 训练 | 验证 | 训练+验证 | 整轮 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BRepPreDiff | mlp | A | 256 | 1 | 0 | 14,295 | 6.92 | 1.32 | 8.24 | 8.25 |
| BRepPreDiff | diffusion | A | 256 | 1 | 0 | 14,295 | 10.88 | 1.98 | 12.86 | 12.86 |
| Fusion360Seg | mlp | A | 256 | 1 | 0 | 24,964 | 7.59 | 1.55 | 9.14 | 9.16 |
| Fusion360Seg | diffusion | A | 256 | 1 | 0 | 24,964 | 8.26 | 2.14 | 10.39 | 10.41 |
| MFCAD++ | mlp | A | 256 | 1 | 0 | 41,766 | 15.61 | 3.13 | 18.73 | 18.75 |
| MFCAD++ | diffusion | A | 256 | 1 | 0 | 41,766 | 18.02 | 4.60 | 22.61 | 22.62 |
| TMCAD | mlp | A | 256 | 1 | 0 | 8,709 | 14.99 | 2.89 | 17.88 | 17.88 |
| TMCAD | diffusion | A | 256 | 1 | 0 | 8,709 | 15.24 | 3.17 | 18.41 | 18.42 |
| SolidLetters | mlp | A | 256 | 1 | 0 | 69,660 | 28.71 | 3.04 | 31.75 | 31.77 |
| SolidLetters | diffusion | A | 256 | 1 | 0 | 69,660 | 55.84 | 7.97 | 63.81 | 63.70 |
| CADSynth | mlp | A | 512 | 1 | 0 | 79,941 | 22.09 | 3.31 | 25.40 | 25.44 |
| CADSynth | diffusion | A | 512 | 1 | 0 | 79,941 | 16.70 | 2.98 | 19.68 | 19.69 |
| MFInstSeg | mlp | A | 512 | 1 | 0 | 49,996 | 13.45 | 2.39 | 15.84 | 15.89 |
| MFInstSeg | diffusion | A | 512 | 1 | 0 | 49,996 | 10.57 | 2.36 | 12.92 | 12.93 |
| BRepPreDiff | diffusion | B | 128 | 2 | 0.5 | 14,295 | 11.15 | 1.95 | 13.10 | 13.13 |
| BRepPreDiff | mlp | B | 128 | 2 | 0.5 | 14,295 | 10.25 | 1.61 | 11.86 | 11.87 |
| Fusion360Seg | diffusion | B | 128 | 2 | 0.5 | 24,964 | 9.61 | 2.25 | 11.86 | 11.88 |
| Fusion360Seg | mlp | B | 128 | 2 | 0.5 | 24,964 | 9.14 | 1.78 | 10.92 | 10.94 |
| MFCAD++ | diffusion | B | 128 | 2 | 0.5 | 41,766 | 19.37 | 4.31 | 23.68 | 23.73 |
| MFCAD++ | mlp | B | 128 | 2 | 0.5 | 41,766 | 18.97 | 3.43 | 22.41 | 22.42 |
| TMCAD | diffusion | B | 128 | 2 | 0.5 | 8,709 | 14.92 | 3.05 | 17.96 | 17.99 |
| TMCAD | mlp | B | 128 | 2 | 0.5 | 8,709 | 14.57 | 2.72 | 17.30 | 17.32 |
| SolidLetters | mlp | B | 128 | 2 | 0.5 | 69,660 | 33.46 | 3.16 | 36.61 | 36.63 |
| SolidLetters | diffusion | B | 128 | 2 | 0.5 | 69,660 | 35.64 | 4.35 | 39.98 | 40.03 |
| CADSynth | diffusion | B | 256 | 2 | 0.5 | 79,941 | 39.75 | 6.68 | 46.43 | 46.60 |
| CADSynth | mlp | B | 256 | 2 | 0.5 | 79,941 | 37.77 | 5.47 | 43.24 | 43.27 |
| MFInstSeg | diffusion | B | 256 | 2 | 0.5 | 49,996 | 29.93 | 5.41 | 35.34 | 35.44 |
| MFInstSeg | mlp | B | 256 | 2 | 0.5 | 49,996 | 27.36 | 4.49 | 31.84 | 31.90 |

## 复现与限制

所有选取的日志 world_size 均为 1。历史运行可能受到共享 GPU、并发任务、缓存状态影响；硬件元数据按 run 记录在 CSV，缺失时不推定型号。DiffLoss 验证包含采样推理，具体采样配置见各 run 的 config.yaml。
这里的 DiffLoss 使用各运行嵌入配置；prediction_type 和 sampling_steps 已导出至 CSV，不能将其耗时套用到其他采样步数。训练+验证和整轮的统计 epoch 集合不同，均值偶尔可能出现整轮略小于训练+验证。
当前无 /dev/nvidia*，无法补跑 batch 16/32/64/128/256/512 的同硬件、同配置测速。现有全量下游表覆盖 128/256/512；few-shot 的 16/32 不混入全量训练对比。
运行：`PYTHONPATH=src /home/nvme03/hhfeng/miniconda3/envs/meshpred/bin/python scripts/summarize_bpd_epoch_timing.py`。
汇总 CSV 和 JSON 包含配置路径、日志来源、硬件、参数量、耗时及标准差；`_epochs.csv` 保留全部逐轮起止时间。

## 日志来源

- 九源 train-only / pretrain / batch 128：`runs/inductive_lr1e4_encoder/pretrain/20260901-115311_inductive9_no_fab_encoder_lr1e4_e100_seed42_20260901/logs/pretrain.log`
- BRepPreDiff / mlp / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-200155_full_brepprediff_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- BRepPreDiff / diffusion / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-202953_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- Fusion360Seg / mlp / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-211317_full_fusion360seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- Fusion360Seg / diffusion / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-214421_full_fusion360seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- MFCAD++ / mlp / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-221937_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- MFCAD++ / diffusion / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260901-232240_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- TMCAD / mlp / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-003838_full_tmcad_cls_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- TMCAD / diffusion / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-013844_full_tmcad_cls_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- SolidLetters / mlp / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-024034_full_solidletters_cls_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- SolidLetters / diffusion / batch 256：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-042701_full_solidletters_cls_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- CADSynth / mlp / batch 512：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-080006_full_cadsynth_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- CADSynth / diffusion / batch 512：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-092556_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- MFInstSeg / mlp / batch 512：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-103203_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- MFInstSeg / diffusion / batch 512：`runs/inductive_lr1e4_pre100_full_ft200/finetune/20260902-112521_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901/logs/finetune.log`
- BRepPreDiff / diffusion / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- BRepPreDiff / mlp / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-045201_full_brepprediff_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- Fusion360Seg / diffusion / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- Fusion360Seg / mlp / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-053603_full_fusion360seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- MFCAD++ / diffusion / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- MFCAD++ / mlp / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-061558_full_mfcadpp_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- TMCAD / diffusion / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- TMCAD / mlp / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-073527_full_tmcad_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- SolidLetters / mlp / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-083543_full_solidletters_cls_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- SolidLetters / diffusion / batch 128：`runs/inductive9_rotate_mix/finetune/20260909-083544_full_solidletters_cls_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- CADSynth / diffusion / batch 256：`runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- CADSynth / mlp / batch 256：`runs/inductive9_rotate_mix/finetune/20260909-104933_full_cadsynth_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- MFInstSeg / diffusion / batch 256：`runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_diffloss_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
- MFInstSeg / mlp / batch 256：`runs/inductive9_rotate_mix/finetune/20260909-112027_full_mfinstseg_seg_mlp_ft200_seed42_inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final/logs/finetune.log`
