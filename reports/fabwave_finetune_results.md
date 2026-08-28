# FabWave classification fine-tuning results

## Setup

- Dataset root: `/data/hhfeng/FabWave`
- Task: whole-model classification (`task: cls`)
- Source CAD: 4,575 STEP/STP models in 44 non-empty source categories
- OCC-valid CAD: 4,549 models
- Clean split sizes: 3,639 train, 463 validation, 447 test
- Pretrained encoder:
  `<joint-pretrain-run>/checkpoints/last.pt`
- MLP: 100 epochs; DiffLoss: 200 epochs
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- Effective training batch size: 512 (256 per GPU, two GPUs per head)
- Selection: highest validation Macro-F1
- Test evaluation: exact single-process pass over all 447 test models

Both heads fine-tune the same pretrained encoder. DiffLoss uses bipolar one-hot targets, four
noise samples per graph token, `x_start` prediction, and deterministic one-step DDIM inference.
MLP uses cross-entropy and softmax inference.

## Results

| Head | Best epoch | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test weighted-F1 | Test Macro-IoU | Test weighted-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 30 | 86.21% | 89.68% | 85.01% | 90.67% | 82.91% | 89.51% | 77.81% |
| DiffLoss | 196 | 85.99% | 90.16% | 86.13% | 91.50% | 86.23% | 90.03% | 80.48% |
| DiffLoss - MLP | — | -0.22 pp | +0.48 pp | +1.12 pp | +0.84 pp | +3.32 pp | +0.52 pp | +2.66 pp |

DiffLoss is better on every aggregate test metric. The largest gain is weighted-F1 (+3.32
percentage points), driven mainly by improved separation of the three shaft/washer-like classes.
MLP converges much faster, reaching its selected checkpoint at epoch 30 versus epoch 196 for
DiffLoss.

## Test classes with errors

All supported test classes not listed below have F1 = 100% for both heads.

| Class | Support | MLP F1 | DiffLoss F1 | DiffLoss - MLP |
|---|---:|---:|---:|---:|
| Keyway_Shaft | 25 | 66.67% | 40.00% | -26.67 pp |
| O_Rings | 37 | 63.79% | 64.65% | +0.86 pp |
| Rotary_Shaft | 25 | 0.00% | 50.91% | +50.91 pp |
| Washers | 72 | 58.82% | 70.59% | +11.76 pp |

The configured 44-class Macro-F1 includes two zero-support test classes (`Miter Gears` and
`Webbing Guide`) as zero. Restricting the macro average to the 42 supported test classes gives
94.98% for MLP and 95.86% for DiffLoss. The primary table deliberately retains the repository's
standard configured-class metric so it matches the saved evaluator output.

## Data audit

- `Fixed Cap Flange` is present as a directory in the 25-45 group but contains no STEP/STP file,
  including in the source ZIP. It is excluded before class mapping.
- `Webbing Guide` contains 26 STEP files, but OCC rejects all 26 while mapping faces. They are
  retained in the 44-class mapping for auditability and excluded from the clean splits.
- `Miter Gears` contains only one STEP file. Stratified allocation places it in training, so it has
  no validation or test support.
- The other 4,549 cache files passed the full finite-value cache scan.

## Artifacts

- Dataset config: `data/fabwave.yaml`
- Class map: `data/splits/fabwave_class_map.json`
- Invalid STEP audit: `/data/hhfeng/brepprediff/cache/features/fabwave_invalid.jsonl`
- MLP checkpoint:
  `runs/finetune/20260804-152415_joint_fabwave_mlp/checkpoints/best.pt`
- MLP metrics:
  `runs/finetune/20260804-152415_joint_fabwave_mlp/test_metrics.json`
- DiffLoss checkpoint:
  `runs/finetune/20260804-153506_joint_fabwave_diffloss_200/checkpoints/best.pt`
- DiffLoss metrics:
  `runs/finetune/20260804-153506_joint_fabwave_diffloss_200/test_metrics.json`

DiffLoss was resumed from its epoch-90 periodic checkpoint with `train.num_workers=0` after the
original run's DataLoader workers became orphaned at epoch 99. The optimizer, model, and epoch
state were restored from the periodic checkpoint; the completed result is the resumed run above.
