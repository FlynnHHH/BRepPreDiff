# MFInstSeg / MFCAD++ / CADSynth multi-supervision-ratio design

- Ratios: 0.1%, 0.5%, 1%, 1.5%, 2%, 3%, and 100% of labeled training CAD models.
- Selection: seed 42, exact ceiling sample counts, nested subsets. A deterministic greedy prefix
  first covers all 25 semantic classes; remaining models follow a stable SHA-256 order.
- Validation/test: original complete splits, never subsampled. CADSynth uses the established clean
  train/validation/test splits because 59 train models have invalid OCC/label correspondence.
- Model: 4-layer Edge Update Attention encoder, hidden size 128, 4 heads, MLP segmentation head;
  all encoder/head parameters are fine-tuned.
- Initialization: full-loss inductive-9 encoder, pretraining LR 1e-4, constant schedule, 100 epochs,
  seed 42.
- Fine-tuning: 200 epochs, seed/DataLoader seed 42, LR 3e-4, weight decay 1e-4, Dice weight 0.3.
  The full validation split is evaluated every 5 epochs; validation accuracy selects `best.pt`,
  which is evaluated once on the unchanged test set.
- Parallelism: one sequential ratio queue per benchmark on physical GPUs 1, 2, and 3.
- Metrics: face accuracy, Macro-F1, Weighted-F1, and Macro-IoU.
- The 100% endpoints reuse the matching completed full-data MLP runs from
  `runs/inductive_lr1e4_pre100_full_ft200`; those runs used the same seed, encoder checkpoint,
  data splits, hyperparameters, and 200-epoch budget, with denser validation (every epoch).

The launcher is `scripts/run_supervision_ratio_three_gpu.sh`; deterministic split audits are stored
beside the splits under `data/splits/supervision_ratios/seed42` and copied into each result folder.
