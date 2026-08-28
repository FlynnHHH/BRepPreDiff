# Finetune baseline and ablation results

Generated from the completed suite at `runs/finetune_suite/20260720-205223_full`.
The suite completed on 2026-07-21 and contains one baseline plus 10 retained ablation experiments.

## Evaluation protocol

- Task: three-class face segmentation (`NonTransition`, `VBF`, `EBF`)
- Train split: `data/splits/finetune_train.txt` (6,127 models)
- Validation split: `data/splits/finetune_val.txt` (766 models)
- Test split: `data/splits/finetune_test.txt` (766 models, 41,459 faces)
- Batch size: 512 per rank
- DataLoader workers: 16 per rank
- Best-checkpoint selection: validation Macro-F1
- Test metrics: exact face-level confusion matrix from each experiment's best checkpoint

## Summary

| Experiment | Best epoch | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test mIoU | Transition F1 | Δ Macro-F1 vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline_default** | 43 | 0.944927 | 0.978509 | 0.954612 | 0.914494 | 0.951011 | +0.000000 |
| ablation_epsilon_encoder_all | 99 | 0.638596 | 0.867291 | 0.649302 | 0.519467 | 0.679766 | -0.305311 |
| ablation_epsilon_encoder_partial | 100 | 0.861719 | 0.944958 | 0.863970 | 0.769203 | 0.876156 | -0.090642 |
| ablation_epsilon_weight_0_1 | 31 | 0.945697 | 0.978581 | 0.952322 | 0.910371 | 0.952709 | -0.002290 |
| ablation_epsilon_weight_0_25 | 27 | 0.944592 | 0.976386 | 0.947276 | 0.901483 | 0.948462 | -0.007336 |
| ablation_epsilon_weight_1_0 | 71 | 0.945368 | 0.977279 | 0.952606 | 0.910789 | 0.943692 | -0.002006 |
| **ablation_head_mlp_full** | 98 | **0.953323** | 0.981234 | **0.964880** | **0.933230** | 0.953749 | **+0.010268** |
| ablation_mlp_encoder_all | 100 | 0.898494 | 0.960660 | 0.903100 | 0.828301 | 0.916718 | -0.051512 |
| ablation_mlp_encoder_partial | 96 | 0.946666 | **0.981283** | 0.964395 | 0.932313 | **0.954352** | +0.009783 |
| ablation_prediction_epsilon_full | 100 | 0.895533 | 0.956945 | 0.902749 | 0.827687 | 0.900921 | -0.051863 |
| ablation_prediction_x_start_full | 87 | 0.942850 | 0.978605 | 0.959097 | 0.922722 | 0.947877 | +0.004485 |

## Baseline details

The baseline uses the DiffLoss head, joint `x_start_epsilon` prediction with weights `1.0/0.5`,
and a fully trainable encoder.

| Metric | Value |
|---|---:|
| Best epoch | 43 |
| Test faces | 41,459 |
| Accuracy | 0.978509 |
| Macro precision | 0.948725 |
| Macro recall | 0.960948 |
| Macro-F1 | 0.954612 |
| Macro-IoU | 0.914494 |
| Weighted-F1 | 0.978751 |
| Weighted-IoU | 0.959376 |
| Transition F1 | 0.951011 |
| Transition IoU | 0.906598 |

Baseline per-class results:

| Class | Support | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| NonTransition | 34,166 | 0.993593 | 0.985014 | 0.989285 | 0.978798 |
| VBF | 1,762 | 0.954416 | 0.950624 | 0.952516 | 0.909338 |
| EBF | 5,531 | 0.898166 | 0.947207 | 0.922034 | 0.855347 |

Baseline confusion matrix (rows are ground truth, columns are predictions):

| GT / Pred | NonTransition | VBF | EBF |
|---|---:|---:|---:|
| NonTransition | 33,654 | 5 | 507 |
| VBF | 0 | 1,675 | 87 |
| EBF | 217 | 75 | 5,239 |

## Main findings

- Replacing DiffLoss with a fully trainable MLP head gives the best test Macro-F1,
  `0.964880`, improving on the baseline by 1.027 percentage points.
- Partially freezing two encoder layers with the MLP head remains competitive at
  `0.964395` Macro-F1 and gives the highest transition F1 (`0.954352`).
- Predicting `x_start` alone improves test Macro-F1 by 0.448 points; predicting only
  `epsilon` reduces it by 5.186 points.
- Changing the epsilon loss weight from the baseline value `0.5` to `0.1`, `0.25`, or
  `1.0` does not improve Macro-F1.
- Freezing the complete encoder is consistently harmful, especially for the epsilon-only
  DiffLoss configuration.

## Source artifacts

- Full per-experiment report, including every confusion matrix:
  `runs/finetune_suite/20260720-205223_full/RESULTS.md`
- Machine-readable summary: `runs/finetune_suite/20260720-205223_full/results.json`
- Spreadsheet-friendly summary: `runs/finetune_suite/20260720-205223_full/results.csv`
- Individual test metrics:
  `runs/finetune_suite/20260720-205223_full/test_evaluations/`
