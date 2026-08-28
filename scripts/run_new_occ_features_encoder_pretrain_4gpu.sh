#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/hhfeng/miniconda3/envs/blendit/bin/python"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="new_occ_features_edge_update_${RUN_TAG}"
LAUNCH_DIR="$ROOT_DIR/runs/launch_logs/$RUN_NAME"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-null}"

mkdir -p "$LAUNCH_DIR"
printf '%s\n' "$$" > "$LAUNCH_DIR/launcher.pid"
exec > >(tee -a "$LAUNCH_DIR/launcher.log") 2>&1
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "[$(date --iso-8601=seconds)] queued run=$RUN_NAME gpu_ids=$GPU_IDS resume=$RESUME_CHECKPOINT"
while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 5 minutes"
  sleep 300
done

echo "[$(date --iso-8601=seconds)] CUDA ready; starting four-GPU encoder pretraining"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config "$ROOT_DIR/configs/pretrain_joint_all_splits.yaml" \
  --override "run.name=$RUN_NAME" \
  --override "model.encoder_type=edge_update_attention" \
  --override "model.num_heads=4" \
  --override "brep.edge_u_grid_size=10" \
  --override "train.batch_size=64" \
  --override "train.num_workers=16" \
  --override "train.resume=$RESUME_CHECKPOINT" \
  --override "wandb.enabled=false" \
  --override "run.show_progress=false"

echo "[$(date --iso-8601=seconds)] encoder pretraining completed run=$RUN_NAME"
