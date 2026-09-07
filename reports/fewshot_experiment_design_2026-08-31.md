# 50-epoch encoder 10/20-shot downstream experiment

## Scope

- Local encoder pretraining: seven-source transductive unlabeled corpus, 50 epochs, seed 42.
- Encoder: 4-layer Edge Update Attention, hidden dimension 128, 4 attention heads.
- Physical device: GPU 4 (NVIDIA A800 80GB).
- Pretraining batch size: 128 on one GPU, matching the earlier 4 x 32 global batch.
- Pretraining learning rate: `5e-4`; checkpoints saved every 5 epochs.
- Downstream head: MLP; encoder is fully fine-tuned.
- Downstream budget: 100 epochs for both classification and segmentation.
- Classification pooling: Mean+Max.

## K-shot definition

The selection unit is one labeled CAD model. For classification, K distinct models are selected
per graph class. For semantic segmentation, K distinct models containing each face class are
selected independently and their union forms the training set. A model containing several face
classes can therefore support several classes. The 10-shot selection is nested in the 20-shot
selection. Selection uses deterministic SHA-256 ranks derived from seed 42, class ID, and sample
ID. Validation and test splits remain complete and unchanged.

| Task | Type | Classes | 10-shot unique CADs | 20-shot unique CADs |
|---|---|---:|---:|---:|
| Blendit/BRepPreDiff | segmentation | 3 | 30 | 60 |
| Fusion360Seg | segmentation | 8 | 80 | 160 |
| MFCAD++ | segmentation | 25 | 249 | 496 |
| TMCAD | classification | 10 | 100 | 200 |

FabWave-min10 is excluded: class 1 has only 9 unique models in the training split, so the current
dataset cannot support a strict 10-shot or 20-shot experiment without replacement or a changed
taxonomy.

## Evaluation

- Fine-tuning seed and DataLoader seed: 42.
- Full validation split evaluated every 5 epochs.
- `best.pt` selected by validation accuracy.
- The selected checkpoint is evaluated once on the unchanged full test split.
- Report accuracy, macro F1, and macro IoU.

## Paths

- Encoder run: `runs/fewshot/pretrain/20260831-112232_fewshot_encoder_pretrain50_b128_gpu4_20260831`
- Few-shot splits: `data/splits/fewshot/seed42`
- Fine-tuning/result root: `runs/fewshot`
- Final report tag: `fewshot_pre50_b128_seed42_20260831`

The suite is launched with `scripts/run_fewshot_gpu4.sh`; split creation and result summarization
are implemented by `scripts/create_fewshot_splits.py` and `scripts/summarize_fewshot.py`.
