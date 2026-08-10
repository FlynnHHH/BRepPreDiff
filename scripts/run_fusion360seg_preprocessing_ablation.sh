#!/usr/bin/env bash
# Fine-tune and test three preprocessing-matched pretrained models in parallel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "GPU_IDS must contain exactly three comma-separated GPU IDs." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
summary_log="$LOG_DIR/${timestamp}_fusion360seg_preprocessing_ablation.summary.log"
names=(per_graph none typewise_global)
run_names=(
  fusion360seg_ablation_per_graph
  fusion360seg_ablation_none
  fusion360seg_ablation_typewise_global
)
configs=(
  configs/finetune_fusion360seg_ablation_per_graph.yaml
  configs/finetune_fusion360seg_ablation_none.yaml
  configs/finetune_fusion360seg_ablation_typewise_global.yaml
)

run_one() {
  local gpu="$1"
  local name="$2"
  local run_name="$3"
  local config="$4"
  local train_log="$LOG_DIR/${timestamp}_fusion360seg_${name}_train.log"

  echo "$(date --iso-8601=seconds) starting $name gpu=$gpu" | tee -a "$summary_log"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 "$PYTHON_BIN" \
    -m blendit.training.finetune --config "$config" >"$train_log" 2>&1

  local run_dir
  run_dir="$(find runs/finetune -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$run_dir" || ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "No best checkpoint found for $name." >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 "$PYTHON_BIN" \
    -m blendit.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 512 \
    --num-workers 8 \
    --device cuda >"$run_dir/test_evaluate.log" 2>&1
  echo "$(date --iso-8601=seconds) completed $name run_dir=$run_dir" | tee -a "$summary_log"
}

pids=()
for index in "${!names[@]}"; do
  run_one "${GPUS[$index]}" "${names[$index]}" "${run_names[$index]}" "${configs[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
