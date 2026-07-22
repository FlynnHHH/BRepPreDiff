# TMCAD classification fine-tuning results

## Setup

- Dataset root: `/data/hhfeng/TMCAD`
- Task: whole-model classification (`task: cls`)
- Classes: 10, in alphabetical order: bearing, bolt, bracket, coupling, flange, gear, nut,
  pulley, screw, shaft
- Source models / generated one-hot `.cls` files: 10,897 / 10,897
- Valid split sizes: 8,709 train, 1,090 validation, 1,087 test (10,886 total)
- Invalid STEP audit: 9 train, 1 validation, 1 test (11 total)
- Pretrained encoder:
  `runs/pretrain/20260720-212145_pretrain_resume_20260720_205223/checkpoints/last.pt`
- Fine-tuning: 100 epochs, AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size 256
  per GPU, 4 TITAN RTX GPUs
- Selection: highest validation Macro-F1
- Test evaluation: exact single-process pass over all 1,087 test models

DiffLoss uses bipolar one-hot targets, `x_start_epsilon` prediction, four noise samples per graph
token, and deterministic 25-step DDIM inference. MLP uses cross-entropy and softmax inference.
Both heads load the same pretrained encoder and randomly initialize their graph-level heads.

## Results

| Head | Best epoch | Val accuracy | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test weighted-F1 | Test Macro-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| DiffLoss | 100 | 75.28% | 75.05% | 74.43% | 74.08% | 74.71% | 60.19% |
| MLP | 42 | 83.70% | 83.40% | 81.88% | 81.34% | 81.74% | 69.43% |
| MLP - DiffLoss | — | +8.42 pp | +8.35 pp | +7.45 pp | +7.26 pp | +7.04 pp | +9.23 pp |

MLP is better on every reported aggregate test metric. It also converges substantially faster:
its validation Macro-F1 reaches 70.53% by epoch 10, while DiffLoss reaches 8.72% at the same epoch
and improves gradually through epoch 100.

## Per-class test F1

| Class | Support | DiffLoss | MLP | MLP - DiffLoss |
|---|---:|---:|---:|---:|
| bearing | 110 | 68.97% | 78.86% | +9.90 pp |
| bolt | 151 | 90.85% | 92.15% | +1.30 pp |
| bracket | 112 | 77.36% | 85.45% | +8.09 pp |
| coupling | 107 | 51.81% | 64.29% | +12.47 pp |
| flange | 100 | 84.88% | 89.76% | +4.88 pp |
| gear | 101 | 76.24% | 84.88% | +8.64 pp |
| nut | 101 | 84.51% | 89.22% | +4.71 pp |
| pulley | 101 | 55.35% | 68.72% | +13.37 pp |
| screw | 100 | 77.97% | 83.41% | +5.45 pp |
| shaft | 104 | 72.92% | 76.70% | +3.78 pp |

The largest remaining confusions are coupling and pulley. MLP improves those categories by 12.47
and 13.37 percentage points respectively, but they remain the lowest-F1 classes.

## Artifacts

- DiffLoss checkpoint:
  `runs/finetune/20260721-172132_tmcad_diffloss/checkpoints/best.pt`
- DiffLoss full test metrics:
  `runs/finetune/20260721-172132_tmcad_diffloss/test_metrics.json`
- MLP checkpoint:
  `runs/finetune/20260721-173614_tmcad_mlp/checkpoints/best.pt`
- MLP full test metrics:
  `runs/finetune/20260721-173614_tmcad_mlp/test_metrics.json`

The JSON metric files contain the complete 10x10 confusion matrices and precision, recall, F1,
and IoU for every class. Run artifacts are intentionally local and excluded from Git.

## Invalid STEP audit

The following source models retain their generated `.cls` labels but are excluded from training
because OCC could not produce a finite B-Rep graph:

```text
coupling/822.stp
gear/622.stp
nut/181.stp
pulley/377.stp
pulley/59.stp
pulley/642.stp
pulley/658.stp
pulley/672.stp
pulley/948.stp
coupling/763.stp
pulley/113.stp
```
