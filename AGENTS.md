# BRepPreDiff project memory

## Whole-model classification pooling

- Use `model.graph_pooling: mean_max` by default for all new or repeated `task: cls`
  experiments, including both MLP and DiffLoss heads.
- Keep another pooling method only when the user explicitly requests a pooling ablation or when
  reproducing/evaluating an older checkpoint with an embedded configuration.
- Do not reinterpret legacy classification configs or checkpoints that omit `graph_pooling`;
  the code-level fallback remains `mean` for backward compatibility.
- Evidence: `reports/cls_pooling_ablation_2026-08-11.md`. On TMCAD, Mean+Max improved test
  accuracy by 2.9439 percentage points; on saturated FabWave min10 it tied Mean, so this is the
  project-wide experiment default rather than a claim of universal superiority.

## Downstream fine-tuning budget

- Use `train.epochs: 200` by default for all new or repeated downstream classification and
  segmentation experiments.
- Keep another epoch budget only when the user explicitly requests a training-budget ablation or
  when reproducing/evaluating an older run with its original protocol.
