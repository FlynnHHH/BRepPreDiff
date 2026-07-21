# MFCAD++ fine-tuning results

Run date: 2026-07-21

## Dataset and protocol

- Dataset root: `/data/hhfeng/MFCAD++`
- Classes: 24 machining-feature classes (`0..23`) plus Stock/background (`24`), for 25 segmentation outputs
- Official splits: 41,766 train / 8,950 validation / 8,949 test models
- Test faces: 268,982
- Feature cache: 59,665 validated NPZ graphs, `uv_grid_size: 10`
- Initialization: Blendit 150-epoch pretrained encoder at
  `runs/pretrain/20260720-212145_pretrain_resume_20260720_205223/checkpoints/last.pt`
- Fine-tuning: 100 epochs, two GPUs per method, batch size 256 per GPU, AdamW, learning rate `3e-4`
- Checkpoint selection: highest validation Macro-F1
- Evaluation: exact face-level confusion matrix over the complete cached test split

Both methods selected epoch 100 as their best validation checkpoint.

## Test metrics

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro IoU | Weighted F1 | Weighted IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiffLoss baseline | 0.985334 | 0.974424 | 0.977523 | 0.975851 | 0.953785 | 0.985368 | 0.971833 |
| MLP | **0.989661** | **0.983963** | **0.983604** | **0.983761** | **0.968433** | **0.989657** | **0.979816** |

MLP improves test Accuracy by 0.433 percentage points, Macro-F1 by 0.791 points, and Macro-IoU by 1.465 points over the DiffLoss baseline.

## Lowest per-class F1

| Method | Class | Support | F1 | IoU |
|---|---|---:|---:|---:|
| DiffLoss baseline | 6 Rectangular through slot | 4,097 | 0.909221 | 0.833551 |
| DiffLoss baseline | 8 Rectangular through step | 7,017 | 0.935125 | 0.878155 |
| DiffLoss baseline | 10 Slanted through step | 7,039 | 0.936555 | 0.880681 |
| MLP | 6 Rectangular through slot | 4,097 | 0.944364 | 0.894593 |
| MLP | 10 Slanted through step | 7,039 | 0.957203 | 0.917920 |
| MLP | 8 Rectangular through step | 7,017 | 0.957744 | 0.918915 |

## Artifacts

- Baseline run: `runs/finetune/20260721-144209_mfcad_baseline`
- Baseline full metrics: `runs/finetune/20260721-144209_mfcad_baseline/test_metrics.json`
- MLP run: `runs/finetune/20260721-144209_mfcad_mlp`
- MLP full metrics: `runs/finetune/20260721-144209_mfcad_mlp/test_metrics.json`
