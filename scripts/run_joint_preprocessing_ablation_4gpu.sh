#!/usr/bin/env bash
# Run three joint-pretraining preprocessing ablations sequentially on four GPUs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"

if [[ ! -x "$PYTHON_BIN" || ! -x "$TORCHRUN_BIN" ]]; then
  echo "BRepPreDiff Python/torchrun executable is unavailable." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs." >&2
  exit 2
fi

for gpu in "${GPUS[@]}"; do
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu")"
  if [[ "${gpu_name^^}" != *TITAN* ]]; then
    echo "GPU $gpu is not a Titan GPU: $gpu_name" >&2
    exit 2
  fi
done

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
summary_log="$LOG_DIR/${timestamp}_joint_preprocessing_ablation.summary.log"
configs=(
  configs/pretrain_joint_ablation_per_graph.yaml
  configs/pretrain_joint_ablation_none.yaml
  configs/pretrain_joint_ablation_typewise_global.yaml
)
names=(per_graph none typewise_global)

status=0
for index in "${!configs[@]}"; do
  config="${configs[$index]}"
  name="${names[$index]}"
  log_file="$LOG_DIR/${timestamp}_joint_pretrain_${name}.log"
  echo "$(date --iso-8601=seconds) starting $name log=$log_file" | tee -a "$summary_log"
  if CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS=1 "$TORCHRUN_BIN" \
    --standalone --nproc_per_node=4 \
    -m brepprediff.training.pretrain --config "$config" >"$log_file" 2>&1; then
    echo "$(date --iso-8601=seconds) completed $name" | tee -a "$summary_log"
  else
    result=$?
    echo "$(date --iso-8601=seconds) failed $name exit=$result" | tee -a "$summary_log"
    status=1
  fi
done

exit "$status"
