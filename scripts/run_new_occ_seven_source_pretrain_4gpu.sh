#!/usr/bin/env bash
# Rebuild the remaining legacy caches, then pretrain the default 711/63
# Edge Update encoder on the inductive train-only joint corpus, including
# SolidLetters/CADSynth/MFInstSeg while holding out every downstream val/test split.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
BREPPREDIFF_CACHE_BIN="${BREPPREDIFF_CACHE_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/brepprediff-cache}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="new_occ_seven_source_edge_update_${RUN_TAG}"
LAUNCH_DIR="$ROOT_DIR/runs/launch_logs/$RUN_NAME"

mkdir -p "$LAUNCH_DIR"
printf '%s\n' "$$" >"$LAUNCH_DIR/launcher.pid"
exec > >(tee -a "$LAUNCH_DIR/launcher.log") 2>&1
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$BREPPREDIFF_CACHE_BIN" ]]; then
  echo "brepprediff-cache executable not found: $BREPPREDIFF_CACHE_BIN" >&2
  exit 2
fi

# These three source caches were still on the legacy 611/3 schema.
cache_configs=(
  data/fusion360seg_s2_0_1_pretrain.yaml
  data/fusion360rec_r1_0_1_pretrain.yaml
  data/fusion360ass_j1_0_0_pretrain.yaml
)
for config in "${cache_configs[@]}"; do
  for split in train val test; do
    echo "[$(date --iso-8601=seconds)] rebuilding OCC-grid-v2 cache config=$config split=$split"
    "$BREPPREDIFF_CACHE_BIN" \
      --config "$config" \
      --split "$split" \
      --workers 16 \
      --overwrite-cache
  done
done

echo "[$(date --iso-8601=seconds)] caches ready; waiting for CUDA driver"
while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 5 minutes"
  sleep 300
done

echo "[$(date --iso-8601=seconds)] starting run=$RUN_NAME gpu_ids=$GPU_IDS"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m brepprediff.training.pretrain \
  --config "$ROOT_DIR/configs/pretrain.yaml" \
  --override "run.name=$RUN_NAME" \
  --override "run.show_progress=false" \
  --override "wandb.enabled=false"

echo "[$(date --iso-8601=seconds)] completed run=$RUN_NAME"
