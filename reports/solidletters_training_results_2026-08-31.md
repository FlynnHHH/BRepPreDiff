# SolidLetters classification training results (2026-08-31)

## Summary

The 26-class SolidLetters whole-model classification experiment completed all 100 epochs on
physical GPU 4 without CUDA OOM or numerical failures. The DiffLoss classifier selected its best
checkpoint at epoch 84 and achieved **96.8028% test accuracy**, **96.8736% macro-F1**, and
**94.2194% macro-IoU** on 19,392 clean test models.

| Measure | Result |
|---|---:|
| Best validation epoch | 84 |
| Best validation accuracy | 97.1590% |
| Best validation macro-F1 | 97.2130% |
| Test accuracy | 96.8028% |
| Test macro-precision | 96.9113% |
| Test macro-recall | 96.8787% |
| Test macro-F1 | 96.8736% |
| Test macro-IoU | 94.2194% |
| Test weighted-F1 | 96.7939% |
| Test weighted-IoU | 94.0854% |
| Correct / total test predictions | 18,772 / 19,392 |

## Data protocol

- Labels are inferred from the leading filename letter: `a_...step` through `z_...step` map to
  class IDs 0 through 25.
- The dataset-provided test split is preserved. Ten percent of the official training split is
  deterministically reserved for validation with seed 42 and per-letter stratification.
- Clean split sizes are 69,660 train, 7,744 validation, and 19,392 test models.
- OCC extraction rejected 64 source models (52 train, 0 validation, 12 test). These models are
  excluded by the clean splits and retained in `data/splits/solidletters_*_invalid.txt` and
  `cache/features/solidletters_invalid.jsonl` for audit.
- One source model, `b_Fredericka the Great_upper`, is absent from both official split files and
  was not added to an experiment split.
- The 96,796 generated OCC-grid-v2 caches were scanned after extraction; no cached array contained
  NaN/Inf or unreadable data.

## Training configuration

| Setting | Value |
|---|---|
| Task / head | Whole-model classification / label diffusion (DiffLoss) |
| Encoder | 4-layer, hidden dim 128, 4-head `edge_update_attention` |
| Graph pooling | Mean + Max (`mean_max`) |
| Classes | 26 |
| Face / edge input dimensions | 711 / 63 |
| Initialization | Seven-source OCC-grid-v2 pretrained encoder, 123 tensors loaded and 0 skipped |
| Diffusion prediction | `x_start_epsilon`, weights 1.0 / 0.5 |
| Training / sampling steps | 1,000 / 25 |
| Epochs | 100 |
| Batch size | 256 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4 |
| Feature preprocessing | Per-graph normalization |
| Checkpoint selection | Maximum validation accuracy |
| Seed | 42 |
| Device | Physical GPU 4, NVIDIA A800-SXM4-80GB |

The pretrained checkpoint was
`runs/pretrain/20260824-114039_seven_source_711_4gpu_wandb_20260824-113934/checkpoints/last.pt`.
Training ran from 14:55:32 to 15:54:41 (Asia/Shanghai), approximately 59 minutes 9 seconds.

## Batch-size and shared-GPU check

GPU 4 already had another process using approximately 36.6 GiB. Real forward/backward/AdamW steps
were tested sequentially with the final edge-update DiffLoss model before the full run:

| Batch size | Peak allocated | Peak reserved | Outcome |
|---:|---:|---:|---|
| 16 | 203 MiB | 224 MiB | Passed |
| 32 | 286 MiB | 324 MiB | Passed |
| 64 | 567 MiB | 696 MiB | Passed |
| 128 | 1,023 MiB | 1,142 MiB | Passed |
| 256 | 1,986 MiB | 2,182 MiB | Passed |

At formal-training startup, `nvidia-smi` reported approximately 4.15 GiB for the SolidLetters
process and 40.5 GiB still free on GPU 4. The default batch size 256 subsequently completed all
epochs without OOM.

## Optimization trajectory

| Epoch | Train accuracy | Validation accuracy |
|---:|---:|---:|
| 1 | 46.962% | 10.331% |
| 10 | 98.271% | 92.962% |
| 20 | 99.022% | 95.054% |
| 40 | 99.484% | 96.823% |
| 60 | 99.493% | 96.630% |
| 80 | 99.645% | 97.120% |
| 84 | 99.603% | **97.159%** |
| 100 | 99.613% | 97.146% |

At epoch 100, training loss was 0.00257 and validation loss was 0.00531. Validation accuracy
plateaued near 97% while training accuracy approached 99.6%, but the final validation result was
only 0.013 percentage points below the epoch-84 maximum. This indicates mild late-stage
overfitting rather than an unstable collapse.

## Per-class findings

Most letters have test F1 above 97%. The five lowest-F1 classes are:

| Class | Support | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| `l` | 789 | 86.876% | 77.186% | 81.745% | 69.126% |
| `i` | 800 | 80.138% | 87.250% | 83.543% | 71.737% |
| `g` | 733 | 94.188% | 97.271% | 95.705% | 91.763% |
| `j` | 785 | 96.491% | 98.089% | 97.284% | 94.711% |
| `e` | 749 | 99.034% | 95.861% | 97.422% | 94.974% |

The dominant confusions are `l -> i` (157 models) and `i -> l` (83 models). Together they account
for 240 of 620 test errors (38.7%). The next-largest directed confusion is `q -> g` with 12
models. This suggests that future improvements should focus on preserving the subtle B-Rep shape
differences between slender glyphs, especially `i` and lowercase `l`, rather than broadly changing
the classifier.

## Artifacts and reproduction

- Run directory:
  `runs/finetune/20260831-145532_solidletters_diffloss_gpu4_bs256_20260831`
- Best model: `checkpoints/best.pt` (epoch 84)
- Final model: `checkpoints/last.pt` (epoch 100)
- Training log: `logs/finetune.log`
- Embedded resolved configuration: `config.yaml`
- Full test metrics, confusion matrix, and per-class metrics: `test_metrics.json`

Re-evaluate the selected model with:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src \
  /home/nvme03/hhfeng/miniconda3/envs/blendit/bin/python \
  -m brepprediff.training.evaluate \
  --checkpoint runs/finetune/20260831-145532_solidletters_diffloss_gpu4_bs256_20260831/checkpoints/best.pt \
  --split test \
  --output runs/finetune/20260831-145532_solidletters_diffloss_gpu4_bs256_20260831/test_metrics.json \
  --batch-size 256 --num-workers 8 --device cuda
```

## Conclusion

The SolidLetters adaptation and DiffLoss training run are valid and reproducible. The default
Mean+Max pooling and 100-epoch downstream budget work well, and batch size 256 is safe even while
sharing GPU 4 under the observed memory load. Overall classification performance is strong; the
remaining error is disproportionately caused by the visually and topologically similar `i`/`l`
pair.
