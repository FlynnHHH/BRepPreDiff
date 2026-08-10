#!/usr/bin/env bash
# Continue all 3 segmentation benchmarks x 3 heads from epoch 100 to 200.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_WORKTREE="${EDGE_WORKTREE:-/tmp/blendit-encoder-edge-update}"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/edge_update_new_joint}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/edge_update_seg_heads_extend200_$RUN_TAG}"

if [[ ! -d "$EDGE_WORKTREE/src/blendit" ]]; then
  echo "Edge Update worktree not found: $EDGE_WORKTREE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
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

run_extension() {
  local task_name="$1"
  local variant="$2"
  local gpu="$3"
  local source_name="$4"
  local source_dir="$RUN_ROOT/finetune/$source_name"
  local source_config="$source_dir/config.yaml"
  local resume_checkpoint="$source_dir/checkpoints/last.pt"
  local run_name="edge_update_extend200_${task_name}_${variant}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${task_name}_${variant}.log"

  if [[ ! -f "$source_config" || ! -f "$resume_checkpoint" ]]; then
    echo "Incomplete epoch-100 source run: $source_dir" >&2
    return 2
  fi

  echo "[$(date --iso-8601=seconds)] Starting $task_name/$variant epochs 101-200 on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$source_config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "train.resume=$resume_checkpoint" \
    --override "train.epochs=200" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Late-period best checkpoint not found for $task_name/$variant: $checkpoint" >&2
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
  echo "[$(date --iso-8601=seconds)] Completed $task_name/$variant"
}

# Balance each GPU by expected runtime: three MFCAD++ jobs, three Fusion360Seg
# jobs, and three short Blendit jobs are distributed as four sequential queues.
(
  run_extension mfcadpp mlp "${GPUS[0]}" "20260807-200941_edge_update_mfcadpp_seg_20260807-123259"
  run_extension blendit mlp "${GPUS[0]}" "20260807-200941_edge_update_blendit_seg_20260807-123259"
) & worker0=$!
(
  run_extension mfcadpp xstart "${GPUS[1]}" "20260808-163205_edge_update_mfcadpp_seg_diffloss_20260808-head-complements-titan"
  run_extension blendit xstart "${GPUS[1]}" "20260808-163205_edge_update_blendit_seg_diffloss_20260808-head-complements-titan"
) & worker1=$!
(
  run_extension mfcadpp xse "${GPUS[2]}" "20260808-183835_edge_update_mfcadpp_seg_diffloss_xse_20260808-xse-titan"
  run_extension blendit xse "${GPUS[2]}" "20260808-183835_edge_update_blendit_seg_diffloss_xse_20260808-xse-titan"
) & worker2=$!
(
  run_extension fusion360seg mlp "${GPUS[3]}" "20260807-200941_edge_update_fusion360seg_20260807-123259"
  run_extension fusion360seg xstart "${GPUS[3]}" "20260808-163205_edge_update_fusion360seg_diffloss_20260808-head-complements-titan"
  run_extension fusion360seg xse "${GPUS[3]}" "20260808-183835_edge_update_fusion360seg_diffloss_xse_20260808-xse-titan"
) & worker3=$!

status=0
for worker in "$worker0" "$worker1" "$worker2" "$worker3"; do
  if ! wait "$worker"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] All nine segmentation head extensions completed"
