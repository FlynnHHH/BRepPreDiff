#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-runs/pretrain/20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2/checkpoints/last.pt}"
TAG="${TAG:-20260806-max-acc}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"

IFS=',' read -r -a gpus <<< "$GPU_IDS"
if [[ ${#gpus[@]} -ne 4 ]]; then
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

names=(mfcadpp tmcad fusion360seg_s2_0_0 fabwave_min10)
configs=(
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml
)

train_pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${gpus[$index]}"
  config="${configs[$index]}"
  run_name="${name}_mlp_200_${TAG}"
  log_path="$LOG_DIR/${run_name}.train.log"
  echo "[$(date --iso-8601=seconds)] Starting $name MLP training on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "train.epochs=200" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    >"$log_path" 2>&1 &
  train_pids+=("$!")
done

status=0
for index in "${!train_pids[@]}"; do
  if wait "${train_pids[$index]}"; then
    echo "[$(date --iso-8601=seconds)] Training completed: ${names[$index]}"
  else
    echo "[$(date --iso-8601=seconds)] Training failed: ${names[$index]}" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

eval_pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${gpus[$index]}"
  run_name="${name}_mlp_200_${TAG}"
  run_dir="$(find runs/finetune -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $name: $run_dir" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] Starting $name test evaluation on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 512 \
    --num-workers 8 \
    --device cuda \
    >"$run_dir/test_evaluate.log" 2>&1 &
  eval_pids+=("$!")
done

status=0
for index in "${!eval_pids[@]}"; do
  if wait "${eval_pids[$index]}"; then
    echo "[$(date --iso-8601=seconds)] Test evaluation completed: ${names[$index]}"
  else
    echo "[$(date --iso-8601=seconds)] Test evaluation failed: ${names[$index]}" >&2
    status=1
  fi
done
exit "$status"
