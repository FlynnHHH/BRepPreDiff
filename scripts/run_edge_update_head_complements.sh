#!/usr/bin/env bash
# Complete the Edge Update Attention benchmark matrix:
# segmentation with DiffLoss and classification with an MLP head.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_WORKTREE="${EDGE_WORKTREE:-/tmp/brepprediff-encoder-edge-update}"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/edge_update_new_joint}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/edge_update_head_complements_$RUN_TAG}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$RUN_ROOT/pretrain/20260807-123426_edge_update_new_joint_20260807-123259/checkpoints/last.pt}"

if [[ ! -d "$EDGE_WORKTREE/src/brepprediff" ]]; then
  echo "Edge Update worktree not found: $EDGE_WORKTREE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$EDGE_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 60 seconds."
  sleep 60
done

run_downstream() {
  local task_name="$1"
  local gpu="$2"
  local config="$3"
  local epochs="$4"
  local run_name="edge_update_${task_name}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${task_name}.log"

  echo "[$(date --iso-8601=seconds)] Starting $task_name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.epochs=$epochs" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $task_name: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $task_name"
}

# Match the established task-specific budgets: segmentation 100 epochs and
# classification 200 epochs. All runs select best.pt by validation accuracy.
pids=()
run_downstream brepprediff_seg_diffloss "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_diffloss_200.yaml" 100 & pids+=("$!")
run_downstream fusion360seg_diffloss "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_diffloss_200.yaml" 100 & pids+=("$!")
run_downstream mfcadpp_seg_diffloss "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_diffloss_200.yaml" 100 & pids+=("$!")
run_downstream tmcad_cls_mlp "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml" 200 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

run_downstream fabwave_cls_mlp "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml" 200

echo "[$(date --iso-8601=seconds)] Complementary head experiments completed: $RUN_ROOT"
