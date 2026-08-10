# Blendit

Diffusion-based B-Rep representation learning for face segmentation and whole-model classification.

Blendit parses STEP models into attributed B-Rep graphs, caches graph features, pretrains a
geometry-denoising encoder, and fine-tunes it for face-level classification. The default label
space contains three classes:

- `0`: non-transition face
- `1`: vertex blend face (VBF)
- `2`: edge blend face (EBF)

## Highlights

- Direct STEP/STP parsing with `pythonocc-core`.
- Independent STEP and SEG/JSON source directories.
- Automatic split generation, cache completion, and invalid-cache removal.
- Diffusion denoising pretraining over continuous face and edge features.
- Segmentation (`task: seg`) and classification (`task: cls`) downstream tasks.
- MLP and label-diffusion fine-tuning heads for both task types.
- Single- and multi-GPU training with checkpoint resume support.
- STEP-to-SEG inference, PLY visualization, and a Windows inference package builder.

## Pipeline

```text
STEP + SEG/JSON
      │
      ▼
B-Rep graph extraction ──► NPZ feature cache
      │
      ▼
geometry diffusion pretraining
      │
      ▼
face segmentation fine-tuning
      │
      ├──► SEG predictions
      └──► colored PLY visualization
```

The topology remains fixed during pretraining. Gaussian noise is applied to continuous geometry
features such as area, centroid, normals, curvature, UV samples, edge length, and dihedral angle.
Surface type, edge type, topology relation, and optional coarse labels are reconstructed through
auxiliary heads.

## Requirements

- Python 3.9–3.12
- PyTorch 1.12 or newer
- `pythonocc-core` for STEP extraction and mesh export
- NumPy, PyYAML, tqdm

The checked-in Conda file reproduces the locally tested Python/PyTorch environment and includes a
Python 3.9-compatible `pythonocc-core` build for STEP extraction and PLY export.

## Installation

```bash
git clone <your-github-repository-url>
cd Blendit
conda env create -f environment.yml
conda activate blendit
python -m pip install --no-deps -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Configuration

Data preparation and model training use separate YAML files.

```text
data/*.yaml       STEP/SEG paths, split paths, cache paths, B-Rep extraction, label mapping
configs/*.yaml    model, diffusion, optimizer, checkpoint, and run parameters
```

Each training config points to a data config relative to its own location:

```yaml
# configs/pretrain.yaml
data_config: ../data/pretrain.yaml
```

Training and inference merge both files at runtime. Use `--data-config` to select another data
configuration without editing the training YAML.

### Data configuration

Start from [data/default.yaml](data/default.yaml):

```yaml
data:
  steps_dir: data/raw/steps
  segs_dir: data/raw/segs
  cache_dir: data/cache/features
  train_split: data/splits/train.txt
  val_split: data/splits/val.txt
  test_split: data/splits/test.txt
  split_seed: 42
  split_ratios: {train: 0.8, val: 0.1, test: 0.1}

brep:
  uv_grid_size: 4
  smooth_angle_degrees: 5.0

labels:
  raw_to_class_map: {4: 2, 6: 1}
  default_class: 0
  ignore_index: -100
```

`steps_dir` and `segs_dir` are independent; neither must be a child of a shared root. Split files
contain one STEP filename, relative path, stem, or cache filename per line.

The default ABC/BrepDit mapping is:

```text
raw 4       -> EBF (class 2)
raw 6       -> VBF (class 1)
other value -> non-transition (class 0)
```

Set `labels.raw_to_class_map: null` and `labels.default_class: null` when labels already contain
model class IDs, as in [data/filletrec.yaml](data/filletrec.yaml).

Downstream fine-tuning defaults to face segmentation. Set the top-level task to `cls` for
whole-model classification:

```yaml
task: cls
data:
  steps_dir: /path/to/steps
  labels_dir: /path/to/one_hot_cls_files
labels:
  num_classes: 10
```

For `seg`, every graph label tensor has one class ID per face. For `cls`, each `.cls` source file
is a whitespace-separated one-hot vector; it is strictly validated and converted to one graph-level
target. The encoder's face embeddings are mean-pooled before either the MLP or DiffLoss head.

## Prepare data

```bash
blendit-prepare-data --config data/default.yaml --workers 8
```

The command performs one idempotent preparation pass:

1. Read configured train/validation/test split files.
2. Generate missing split files from paired STEP and label files.
3. Build every cache entry required by the splits.
4. Scan all configured cache directories.
5. Delete unreadable caches and caches containing NaN or infinity.
6. Fail if any split is still incomplete.

This preparation pass is the authoritative full-cache validation. Run it again whenever the
STEP/SEG inputs, split files, B-Rep extraction settings, or cached NPZ files change.

Training startup skips the full preparation and cache scan by default:

```yaml
data:
  prepare_on_start: false
```

With this setting, training only resolves each split entry to an existing cache path while building
the Dataset, then loads NPZ files on demand for each batch. It does not scan every cache for
NaN/Inf, rebuild caches, or parse STEP files inside DataLoader workers. If a prepared cache is
later corrupted, training reports the error when that sample is loaded.

The recommended workflow is therefore:

```bash
# Run once, and repeat after any source, split, extraction-setting, or cache change.
blendit-prepare-data --config data/pretrain.yaml --workers 16

# Reuse the validated cache without another full scan.
torchrun --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config configs/pretrain.yaml
```

Set `data.prepare_on_start: true` only when training should run the same full preparation
automatically. Under distributed training, rank 0 then performs the preparation while the other
ranks wait for it to finish.

Advanced data commands are exposed through the same module:

```bash
# Build one split only
blendit-cache --config data/default.yaml --split train --workers 8

# Scan an existing cache
blendit-scan-cache --cache-dir data/cache/features \
  --invalid-log data/cache/invalid.jsonl

# Remove failed samples from a split
blendit-filter-split --split data/splits/train.txt \
  --invalid-log data/cache/invalid.jsonl \
  --output data/splits/train_clean.txt
```

For MFCAD++, the face class is stored as the name of each STEP `ADVANCED_FACE` entity. Extract
one label per line into Blendit-compatible SEG files, while preserving the train/validation/test
directory layout, with:

```bash
python -m blendit.data.mfcad_seg \
  --dataset-root /data/hhfeng/MFCAD++ \
  --step-root /data/hhfeng/MFCAD++/step \
  --seg-root /data/hhfeng/MFCAD++/seg \
  --blendit-splits-dir /data/hhfeng/MFCAD++/blendit_splits \
  --workers 16
```

The dataset contains 24 machining-feature categories (`0..23`) plus the Stock/background label
(`24`), so face segmentation uses 25 output classes. The matching source-data configuration is
`data/mfcad.yaml`; its UV grid matches the pretrained Blendit encoder.

Fine-tune the default DiffLoss baseline or the MLP head with the dedicated configurations:

```bash
blendit-finetune --config configs/finetune_mfcad_baseline.yaml \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/last.pt
blendit-finetune --config configs/finetune_mfcad_mlp.yaml \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/last.pt
```

### TMCAD model classification

TMCAD contains ten model categories stored as top-level directories. Generate a one-hot `.cls`
file beside every STEP/STP model, plus deterministic class-stratified 80/10/10 splits:

```bash
blendit-prepare-tmcad-cls --dataset-root /data/hhfeng/TMCAD
blendit-prepare-data --config data/tmcad.yaml --workers 16
```

The stable alphabetical class mapping is recorded in
`/data/hhfeng/TMCAD/blendit_splits/class_map.json`. With the bundled TMCAD data configuration it is:

```text
0 bearing, 1 bolt, 2 bracket, 3 coupling, 4 flange,
5 gear, 6 nut, 7 pulley, 8 screw, 9 shaft
```

The checked-in `data/splits/tmcad_*.txt` clean splits retain 10,886 parseable models. Eleven source
STEP files rejected by OCC (invalid/empty/non-finite geometry) are listed in the matching
`data/splits/tmcad_*_invalid.txt` audit files; their `.cls` files remain in the source dataset.
The completed DiffLoss/MLP experiment and per-class metrics are recorded in
[reports/tmcad_finetune_results.md](reports/tmcad_finetune_results.md).

Fine-tune and evaluate both heads from the same pretrained encoder:

```bash
PRETRAIN=runs/pretrain/<run>/checkpoints/last.pt

blendit-finetune --config configs/finetune_tmcad_diffloss.yaml \
  --override train.pretrain_checkpoint=$PRETRAIN
blendit-evaluate \
  --checkpoint runs/finetune/<tmcad_diffloss_run>/checkpoints/best.pt \
  --split test --output runs/finetune/<tmcad_diffloss_run>/test_metrics.json

blendit-finetune --config configs/finetune_tmcad_mlp.yaml \
  --override train.pretrain_checkpoint=$PRETRAIN
blendit-evaluate \
  --checkpoint runs/finetune/<tmcad_mlp_run>/checkpoints/best.pt \
  --split test --output runs/finetune/<tmcad_mlp_run>/test_metrics.json
```

## Pretraining

```bash
blendit-pretrain --config configs/pretrain.yaml
```

Common overrides:

```bash
blendit-pretrain --config configs/pretrain.yaml \
  --override train.device=cuda \
  --override train.batch_size=32 \
  --override train.epochs=100
```

Multi-GPU training:

```bash
torchrun --standalone --nproc_per_node=4 -m blendit.training.pretrain \
  --config configs/pretrain.yaml \
  --override train.device=cuda
```

Resume to a target total epoch count:

```bash
blendit-pretrain --config configs/pretrain.yaml \
  --override train.resume=runs/pretrain/<run>/checkpoints/last.pt \
  --override train.epochs=150
```

Use [configs/pretrain_no_coarse.yaml](configs/pretrain_no_coarse.yaml) for the no-coarse-label-head
ablation.

## Fine-tuning

MLP segmentation head:

```bash
blendit-finetune --config configs/finetune.yaml \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/last.pt
```

Label-diffusion head:

```bash
blendit-finetune --config configs/finetune_diffloss.yaml \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/last.pt
```

For the tuned Blendit DiffLoss setup initialized from the joint
`all_splits_no_coarse` checkpoint, use
[configs/finetune_joint_blendit_diffloss.yaml](configs/finetune_joint_blendit_diffloss.yaml).
Its controlled MLP comparison, parameter search, and exact test metrics are recorded in
[reports/blendit_joint_diffloss_tuning_results.md](reports/blendit_joint_diffloss_tuning_results.md).

Multi-GPU fine-tuning:

```bash
torchrun --standalone --nproc_per_node=4 -m blendit.training.finetune \
  --config configs/finetune.yaml \
  --override train.device=cuda \
  --override train.pretrain_checkpoint=runs/pretrain/<run>/checkpoints/last.pt
```

Use `configs/finetune_diffloss.yaml` in the command above to train the label-diffusion head on
multiple GPUs.

`train.encoder_freeze_mode` supports `none`, `all`, and `partial`. For partial freezing,
`train.encoder_frozen_layers` controls how many message-passing layers are frozen.

Fine-tuning checkpoints are selected using validation accuracy by default.

## Evaluation

Evaluate one STEP/SEG pair, or pass two directories with matching relative file paths to evaluate a
dataset:

```bash
blendit-evaluate \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --step /path/to/step-or-directory \
  --seg /path/to/seg-or-directory \
  --output evaluation_metrics.json
```

The command uses the configuration embedded in the checkpoint by default. Use `--config` when
evaluating an older checkpoint without an embedded configuration. The JSON result includes
accuracy, macro and weighted F1/IoU, per-class metrics, the confusion matrix, and binary transition
metrics with VBF and EBF merged.

## Inference

### STEP to SEG

```bash
blendit-infer-seg /path/to/step-or-directory \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --config configs/finetune.yaml
```

The output preserves the input directory structure and converts model class IDs back to SEG values:

```text
non-transition -> 0
EBF            -> 4
VBF            -> 6
```

### PLY visualization and evaluation

```bash
blendit-infer-visual \
  --config configs/finetune.yaml \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt \
  --step /path/to/model.step \
  --seg /path/to/model.seg \
  --output-dir tools/visualize/results
```

To view generated PLY files:

```bash
cd tools/visualize
python viewer_server.py
```

Open `http://localhost:8061`. Additional viewer options are documented in
[tools/visualize/README_visualize.md](tools/visualize/README_visualize.md).

## Outputs

```text
runs/
├── pretrain/<timestamp_name>/
│   ├── config.yaml
│   ├── logs/pretrain.log
│   └── checkpoints/{last.pt,epoch_*.pt}
└── finetune/<timestamp_name>/
    ├── config.yaml
    ├── logs/finetune.log
    └── checkpoints/{best.pt,last.pt,epoch_*.pt}
```

Data caches, run directories, checkpoints, visualization results, and release archives are ignored
by Git. Publish large models as GitHub Release assets or through a model registry instead of adding
them to source control.

## Repository layout

```text
Blendit/
├── configs/                 training configurations
├── data/*.yaml              data-preparation configurations
├── scripts/                 dataset, ablation, and package utilities
├── src/blendit/
│   ├── brep/                OpenCascade feature extraction
│   ├── data/
│   │   ├── segmentation.py   SEG/JSON matching and face-label extraction
│   │   ├── classification.py CLS one-hot parsing and model-label extraction
│   │   └── dataset.py        shared dataset lifecycle, cache, and task routing
│   ├── inference/           SEG and visualization inference
│   ├── models/
│   │   ├── segmentation.py   face-level models, losses, and metrics
│   │   ├── classification.py model-level models, losses, and metrics
│   │   ├── downstream.py     shared encoder and label-diffusion machinery
│   │   └── finetune.py       task router used by training and evaluation
│   └── training/            shared runtime, pretraining, and fine-tuning
├── tests/                   CPU unit and regression tests
└── tools/                   browser viewer and Windows package assets
```

## Testing

```bash
pytest -q
```

The test suite uses synthetic graphs and temporary caches, so it does not require the training
dataset. Tests that exercise OpenCascade-specific behavior are isolated from ordinary CPU tests.

A GitHub Actions workflow runs the test suite for pushes and pull requests.

## Windows inference package

Build a source-based Windows package containing a selected checkpoint:

```bash
python scripts/build_windows_inference_package.py \
  --checkpoint runs/finetune/<run>/checkpoints/best.pt
```

The generated ZIP and SHA-256 file are written to `dist/`. Runtime instructions are in
[tools/windows_inference/README_zh-CN.md](tools/windows_inference/README_zh-CN.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and pull-request checklist.

## License

No open-source license has been selected yet. Add a `LICENSE` file before making the repository
public so that reuse and redistribution terms are explicit.
