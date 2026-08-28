#!/usr/bin/env bash
# Fine-tune the full-unlabeled-corpus encoder with MLP heads on four benchmarks,
# one task per GPU, then evaluate every best checkpoint on its test split.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PIPELINE_TAG="${PIPELINE_TAG:-20260805-unlabeled-v2}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
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

mkdir -p "$LOG_DIR"

names=(mfcadpp tmcad fusion360seg_s2_0_0 fabwave)
configs=(
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_fabwave_mlp.yaml
)

pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${GPUS[$index]}"
  config="${configs[$index]}"
  run_name="${name}_mlp_200_${PIPELINE_TAG}"
  train_log="$LOG_DIR/${run_name}.log"
  echo "[$(date --iso-8601=seconds)] Starting $name MLP fine-tuning on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune \
    --config "$config" \
    --override "run.name=${run_name}" \
    --override "train.epochs=200" \
    --override "train.pretrain_checkpoint=${PRETRAIN_CHECKPOINT}" \
    >"$train_log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[$(date --iso-8601=seconds)] Fine-tuning completed: ${names[$index]}"
  else
    echo "[$(date --iso-8601=seconds)] Fine-tuning failed: ${names[$index]}" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] Evaluating four MLP best checkpoints"
pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${GPUS[$index]}"
  run_name="${name}_mlp_200_${PIPELINE_TAG}"
  run_dir="$(find runs/finetune -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "Best checkpoint not found for $name: $run_dir" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 512 \
    --num-workers 8 \
    --device cuda \
    >"$run_dir/test_evaluate.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[$(date --iso-8601=seconds)] Test evaluation completed: ${names[$index]}"
  else
    echo "[$(date --iso-8601=seconds)] Test evaluation failed: ${names[$index]}" >&2
    status=1
  fi
done
exit "$status"
