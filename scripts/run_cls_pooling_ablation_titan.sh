#!/usr/bin/env bash
# Compare graph-level pooling methods on TMCAD and FabWave using TITAN GPUs 1-3.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/cls_pooling_ablation}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/launch_logs/$RUN_TAG}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt}"

TMCAD_CONFIG="$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml"
FABWAVE_CONFIG="$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

wait_for_driver() {
  while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
    echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 30 seconds."
    sleep 30
  done
}

run_experiment() {
  local dataset="$1"
  local pooling="$2"
  local gpu="$3"
  local config="$4"
  local run_name="pooling_${dataset}_${pooling}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${dataset}_${pooling}.log"

  echo "[$(date --iso-8601=seconds)] Starting $dataset/$pooling on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.graph_pooling=$pooling" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.epochs=200" \
    --override "train.dataloader_seed=42" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $dataset/$pooling: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 256 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $dataset/$pooling: $run_dir"
}

worker_gpu_1() {
  run_experiment tmcad mean 1 "$TMCAD_CONFIG"
  run_experiment fabwave mean 1 "$FABWAVE_CONFIG"
  run_experiment fabwave mean_max 1 "$FABWAVE_CONFIG"
}

worker_gpu_2() {
  run_experiment tmcad mean_max 2 "$TMCAD_CONFIG"
  run_experiment tmcad residual_attention 2 "$TMCAD_CONFIG"
}

worker_gpu_3() {
  run_experiment tmcad mean_std 3 "$TMCAD_CONFIG"
  run_experiment fabwave mean_std 3 "$FABWAVE_CONFIG"
  run_experiment fabwave residual_attention 3 "$FABWAVE_CONFIG"
}

wait_for_driver
worker_gpu_1 & pid_gpu_1=$!
worker_gpu_2 & pid_gpu_2=$!
worker_gpu_3 & pid_gpu_3=$!

status=0
for pid in "$pid_gpu_1" "$pid_gpu_2" "$pid_gpu_3"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one pooling experiment failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] Pooling ablation completed: $RUN_ROOT"
