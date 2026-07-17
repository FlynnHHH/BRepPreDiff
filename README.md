# Blendit

Diffusion-based B-Rep representation learning and face segmentation for blend-face recognition.

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
- MLP and label-diffusion fine-tuning heads.
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

The checked-in Conda file reproduces the locally tested Python/PyTorch environment. That local
environment does not contain `pythonocc-core`; cached-data training and the unit tests work without
it, while STEP extraction and PLY export require installing it separately.

## Installation

```bash
git clone <your-github-repository-url>
cd Blendit
conda env create -f environment.yml
conda activate blendit
python -m pip install --no-deps -e .
```

For STEP extraction, install a `pythonocc-core` build compatible with Python 3.9 in the environment
before running data preparation.

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

Pretraining and fine-tuning run the same preparation automatically before creating DataLoaders.
Training always reads cached NPZ graphs; it does not parse STEP inside DataLoader workers.

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

`train.encoder_freeze_mode` supports `none`, `all`, and `partial`. For partial freezing,
`train.encoder_frozen_layers` controls how many message-passing layers are frozen.

Fine-tuning checkpoints are selected using validation Macro-F1 by default. FilletRec binary
fine-tuning uses positive-class F1 through
[configs/finetune_filletrec_diffloss.yaml](configs/finetune_filletrec_diffloss.yaml).

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
│   ├── data/                graph IO, datasets, preparation, cache validation
│   ├── inference/           SEG and visualization inference
│   ├── models/              encoder, diffusion, and segmentation models
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
