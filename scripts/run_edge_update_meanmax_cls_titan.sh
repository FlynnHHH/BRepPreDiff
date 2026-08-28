#!/usr/bin/env bash
# Fine-tune Edge Update Attention + Mean+Max on TMCAD and FabWave using TITAN GPUs 1 and 2.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/edge_update_meanmax_cls}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/launch_logs/$RUN_TAG}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/edge_update_new_joint/pretrain/20260807-123426_edge_update_new_joint_20260807-123259/checkpoints/last.pt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Edge Update pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
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
  local gpu="$2"
  local config="$3"
  local run_name="edge_update_meanmax_${dataset}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${dataset}.log"

  echo "[$(date --iso-8601=seconds)] Starting $dataset on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.graph_pooling=mean_max" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.epochs=200" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    --override "train.dataloader_seed=42" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $dataset: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $dataset: $run_dir"
}

wait_for_driver
run_experiment tmcad 1 "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml" & pid_tmcad=$!
run_experiment fabwave 2 "$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml" & pid_fabwave=$!

status=0
for pid in "$pid_tmcad" "$pid_fabwave"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one Edge Update + Mean+Max experiment failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] Edge Update + Mean+Max experiments completed: $RUN_ROOT"
