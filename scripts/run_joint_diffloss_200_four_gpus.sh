#!/usr/bin/env bash
# Run the four 200-epoch joint-pretraining DiffLoss fine-tuning experiments
# concurrently, one process per GPU.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$available_gpus" -lt 4 ]]; then
  echo "Four visible GPUs are required; found: $available_gpus" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"

names=(blendit fusion360seg mfcadpp tmcad)
configs=(
  configs/finetune_joint_blendit_diffloss_200.yaml
  configs/finetune_joint_fusion360seg_diffloss_200.yaml
  configs/finetune_joint_mfcadpp_diffloss_200.yaml
  configs/finetune_joint_tmcad_diffloss_200.yaml
)

pids=()
launch() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local log_file="$LOG_DIR/${timestamp}_${name}_diffloss_200.log"

  echo "Launching $name on GPU $gpu; log: $log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" >"$log_file" 2>&1 &
  pids+=("$!")
}

for index in "${!names[@]}"; do
  launch "${GPUS[$index]}" "${names[$index]}" "${configs[$index]}"
done

cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed: ${names[$index]}"
  else
    echo "Failed: ${names[$index]}; see $LOG_DIR/${timestamp}_${names[$index]}_diffloss_200.log" >&2
    status=1
  fi
done

trap - INT TERM
exit "$status"
