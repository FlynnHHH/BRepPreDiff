# BRepPreDiff

BRepPreDiff learns reusable B-Rep representations through geometry diffusion pretraining,
then fine-tunes an MLP head for face segmentation or whole-model classification.

STEP models are converted into attributed graphs: faces are nodes, adjacency relations are
edges, and continuous geometry is augmented with surface and edge attributes. Pretraining
keeps topology fixed, denoises continuous features, and reconstructs discrete attributes.

## Installation

Requirements: Python 3.9–3.12, PyTorch 1.12 or newer, and `pythonocc-core` for STEP extraction
and mesh export. The Conda environment includes these dependencies.

Run commands from the repository root:

```bash
conda env create -f environment.yml
conda activate brepprediff
python -m pip install --no-deps -e .
```

## Configuration

- `data/*.yaml` defines dataset paths, splits, feature extraction, and label mappings.
- `configs/*.yaml` defines models, optimizers, checkpoints, and training parameters.

Training configurations reference a data YAML relative to their own location. Dataset and
output paths are relative to the repository root. Edit the data YAMLs to match your local
files, or pass `--data-config` and `--override key=value` on the command line.

The default encoder uses `edge_update_attention` with 128 hidden channels, four layers,
and four attention heads. Default pretraining runs for 150 epochs on the nine-source
inductive corpus defined in [data/pretrain_joint_inductive.yaml](data/pretrain_joint_inductive.yaml).
Only training splits contribute to this corpus, and their labels are stripped.

Downstream configurations use an MLP head and 200 epochs. Classification configurations
explicitly select `model.graph_pooling: mean_max`; configurations embedded in older checkpoints
that omit this field retain the original `mean` fallback.

The default feature schema has 711 face channels and 63 edge channels, using a 10×10 face
UV grid and a 10-point edge grid. Each face sample contains XYZ, a unit normal, and a
trimming mask; each edge sample contains XYZ and a unit tangent. Pretraining and downstream
feature schemas must match when transferring an encoder.

## Data preparation

Place datasets under `data/raw/` or update their configured paths. Datasets and generated
caches are not included in the source repository. Split files contain STEP filenames,
relative paths, stems, or cache filenames, one per line.

For one dataset, prepare its configured splits and cache:

```bash
brepprediff-prepare-data --config data/finetune.yaml --workers 8
```

This command generates missing splits, builds missing graph caches, removes invalid caches,
and checks that every configured split is complete. Training reuses the cache with
`data.prepare_on_start: false`. Repeat preparation after changing source files or extraction
settings. For the default multi-source pretraining corpus, prepare each source using its
individual data YAML before starting training.

Segmentation accepts per-face SEG or supported JSON labels. The default transition-face
mapping is raw label `4` → EBF/class `2`, raw label `6` → VBF/class `1`, and all other values
→ non-transition/class `0`. For datasets that already use class IDs, configure
`labels.raw_to_class_map: null` and `labels.default_class: null`.

Classification uses `task: cls` and one one-hot `.cls` label per model. Label preparation
commands are available for TMCAD, FabWave, and SolidLetters:

```bash
brepprediff-prepare-tmcad-cls --dataset-root data/raw/tmcad
brepprediff-prepare-fabwave-cls --dataset-root data/raw/fabwave
brepprediff-prepare-solidletters-cls --dataset-root data/raw/solidletters
```

Use the split paths in the corresponding data YAML. TMCAD writes generated splits under
`<dataset-root>/brepprediff_splits/`; FabWave and SolidLetters write under `data/splits/`.
Where a data YAML selects an OCC-cleaned split, filter the generated split using the
invalid-cache log before assigning it to that path:

```bash
brepprediff-filter-split --split data/splits/solidletters_train.txt \
  --invalid-log cache/features/solidletters_invalid.jsonl \
  --output data/splits/solidletters_train_clean.txt
```

Generate CADSynth splits from its official split lists and deterministic MFInstSeg splits:

```bash
python scripts/prepare_cadsynth_mfinstseg_splits.py \
  --cadsynth-root data/raw/cadsynth \
  --mfinstseg-root data/raw/mfinstseg
```

MFCAD++ stores face labels in STEP entity names. Convert them into SEG files with:

```bash
python -m brepprediff.data.mfcad_seg \
  --dataset-root data/raw/mfcad \
  --step-root data/raw/mfcad/step \
  --seg-root data/raw/mfcad/seg \
  --brepprediff-splits-dir data/splits --workers 8
```

## Pretraining

Use the default four-GPU launcher:

```bash
bash scripts/pretrain.sh
```

Set `NPROC_PER_NODE` to the number of GPUs to use. A single-process launch is also available
through the console command:

```bash
brepprediff-pretrain --config configs/pretrain.yaml
```

For a single prepared dataset, override the data configuration:

```bash
brepprediff-pretrain --config configs/pretrain.yaml --data-config data/pretrain.yaml
```

Adjust batch size and worker count to your hardware with `--override train.batch_size=32`
and `--override train.num_workers=8`.

## MLP fine-tuning

The bundled [encoder checkpoint](ckpt/encoder_pretrain.pt) is the Inductive9 rotate-mix
baseline trained for 100 epochs with seed 42, learning rate `1e-4`, batch size 128, and
50% random SO(3) rotation augmentation. It is compatible with all MLP configurations below.
The file preserves the original pretraining checkpoint, including its training configuration;
fine-tuning loads only the encoder weights. Its 100-epoch training budget differs from the
150-epoch budget currently configured for new runs in `configs/pretrain.yaml`.

Use the bundled checkpoint or replace this path with your own completed pretraining run:

```bash
PRETRAIN_CHECKPOINT=ckpt/encoder_pretrain.pt
bash scripts/finetune_mlp.sh \
  --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
```

Verify the bundled file from the repository root with
`sha256sum -c ckpt/encoder_pretrain.pt.sha256`.

The launcher defaults to `configs/finetune.yaml`. Select another dataset with `--config`:

| Dataset / task | Training configuration |
| --- | --- |
| Transition-face segmentation | [configs/finetune.yaml](configs/finetune.yaml) |
| Fusion360Seg segmentation | [configs/finetune_fusion360seg_mlp.yaml](configs/finetune_fusion360seg_mlp.yaml) |
| MFCAD++ segmentation | [configs/finetune_mfcad_mlp.yaml](configs/finetune_mfcad_mlp.yaml) |
| CADSynth segmentation | [configs/finetune_cadsynth.yaml](configs/finetune_cadsynth.yaml) |
| MFInstSeg segmentation | [configs/finetune_mfinstseg.yaml](configs/finetune_mfinstseg.yaml) |
| TMCAD classification | [configs/finetune_tmcad_mlp.yaml](configs/finetune_tmcad_mlp.yaml) |
| FabWave classification | [configs/finetune_fabwave_mlp.yaml](configs/finetune_fabwave_mlp.yaml) |
| SolidLetters classification | [configs/finetune_solidletters_mlp.yaml](configs/finetune_solidletters_mlp.yaml) |

For example, fine-tune TMCAD on four GPUs:

```bash
NPROC_PER_NODE=4 bash scripts/finetune_mlp.sh \
  --config configs/finetune_tmcad_mlp.yaml \
  --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
```

The direct equivalent for a single process is:

```bash
brepprediff-finetune --config configs/finetune_tmcad_mlp.yaml \
  --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
```

Fine-tuning updates the full encoder by default and selects `best.pt` using validation
accuracy. Resume a run with `--override train.resume=PATH`; `train.epochs` is the target
total epoch count. Encoder freezing is controlled by `train.encoder_freeze_mode`
(`none`, `all`, or `partial`) and `train.encoder_frozen_layers`.

Training records metrics with Weights & Biases. Run `wandb login` once, set `WANDB_MODE=offline`
for local logging, or add `--override wandb.enabled=false` to disable tracking.

## Evaluation and inference

The bundled [Blend checkpoint](ckpt/blend_best.pt) contains the fine-tuned encoder and MLP
segmentation head from epoch 135 of a 200-epoch run (seed 42). It has the highest saved
validation accuracy among 109 locally available Blend checkpoints: **98.8433%**. Its encoder
was initialized from the 100-epoch Inductive9 rotate-mix run with discrete reconstruction
losses disabled. Test accuracy is **98.9144%**, macro F1 is **97.1606%**, and mean IoU is
**94.5318%** on 1,787 models (102,980 faces).

The file retains the original model weights, configuration, epoch, and validation metrics;
optimizer state is omitted. It loads directly for evaluation and inference. See
[checkpoint metadata](ckpt/blend_best.json) for provenance, full metrics, and split checksums.
Verify the download with `sha256sum -c ckpt/blend_best.pt.sha256`.

Evaluate the bundled checkpoint on the prepared Blend test split, or substitute your own
downstream checkpoint:

```bash
brepprediff-evaluate \
  --checkpoint ckpt/blend_best.pt \
  --split test --output outputs/metrics.json
```

Evaluation uses the configuration saved in the checkpoint and reports accuracy, F1, IoU,
per-class metrics, and the confusion matrix. Use `--config` to supply a configuration when
it is not embedded in the checkpoint.

For transition-face segmentation, convert STEP files into SEG predictions:

```bash
brepprediff-infer-seg data/raw/example.step \
  --checkpoint ckpt/blend_best.pt \
  --output-dir outputs/predictions
```

The output maps non-transition, EBF, and VBF predictions to SEG values `0`, `4`, and `6`.
For colored PLY exports from segmentation checkpoints:

```bash
brepprediff-infer-visual \
  --checkpoint ckpt/blend_best.pt \
  --step data/raw/example.step \
  --output-dir outputs/visualization
```

## Repository layout

```text
configs/                 default pretraining and dataset-specific MLP configurations
ckpt/                    bundled encoder and Blend segmentation weights, checksums, and metadata
data/                    data configurations and dataset split lists
scripts/                 default launchers and data preparation utilities
src/brepprediff/
  brep/                  OpenCascade feature extraction
  data/                  graph datasets, labels, caching, and preparation
  inference/             SEG prediction and PLY export
  models/                shared encoder, geometry pretraining, and downstream heads
  training/              training, evaluation, checkpointing, and logging
```

Runs are written to `runs/<stage>/<timestamp_name>/` with a resolved `config.yaml`, logs,
and checkpoints. Generated caches, run outputs, checkpoints, and packaged artifacts are ignored
by Git; `ckpt/encoder_pretrain.pt` and `ckpt/blend_best.pt` are explicitly included.
