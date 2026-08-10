#!/usr/bin/env bash
# Prepare every requested STEP source, pretrain the original MLP/FFN encoder on
# four GPUs, then run four DiffLoss fine-tunes concurrently (one per GPU).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS="${WORKERS:-16}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"
PIPELINE_TAG="${PIPELINE_TAG:-$(date +%Y%m%d-%H%M%S)}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" data/splits

prepare_and_filter() {
  local config="$1"
  local prefix="$2"
  local invalid_log="$3"
  shift 3
  local raw_splits=("$@")

  local raw_overrides=()
  local split_names=(train val test)
  local index
  for index in "${!split_names[@]}"; do
    raw_overrides+=(--override "data.${split_names[$index]}_split=${raw_splits[$index]}")
  done

  # Extraction failures are recorded and removed from the clean splits below.
  if ! "$PYTHON_BIN" -m blendit.data.load_data \
      --config "$config" --workers "$WORKERS" "${raw_overrides[@]}"; then
    echo "Initial preparation found invalid STEP files for $prefix; filtering them."
  fi

  for index in "${!split_names[@]}"; do
    "$PYTHON_BIN" -m blendit.data.load_data filter-split \
      --split "${raw_splits[$index]}" \
      --invalid-log "$invalid_log" \
      --output "data/splits/${prefix}_${split_names[$index]}_clean.txt" \
      --removed-output "data/splits/${prefix}_${split_names[$index]}_invalid.txt"
  done

  "$PYTHON_BIN" -m blendit.data.load_data --config "$config" --workers "$WORKERS"
}

echo "[$(date --iso-8601=seconds)] Preparing Fusion360Seg s2.0.1"
prepare_and_filter \
  data/fusion360seg_s2_0_1_pretrain.yaml \
  fusion360seg_s2_0_1 \
  data/cache/features/fusion360seg_s2_0_1_invalid.jsonl \
  data/splits/fusion360seg_train.txt \
  data/splits/fusion360seg_val.txt \
  data/splits/fusion360seg_test.txt

echo "[$(date --iso-8601=seconds)] Preparing Fusion360Rec r1.0.1"
prepare_and_filter \
  data/fusion360rec_r1_0_1_pretrain.yaml \
  fusion360rec_r1_0_1 \
  data/cache/features/fusion360rec_r1_0_1_unlabeled_invalid.jsonl \
  data/splits/fusion360rec_r1_0_1_train_raw.txt \
  data/splits/fusion360rec_r1_0_1_val_raw.txt \
  data/splits/fusion360rec_r1_0_1_test_raw.txt

echo "[$(date --iso-8601=seconds)] Preparing Fusion360Ass j1.0.0"
prepare_and_filter \
  data/fusion360ass_j1_0_0_pretrain.yaml \
  fusion360ass_j1_0_0 \
  data/cache/features/fusion360ass_j1_0_0_unlabeled_invalid.jsonl \
  data/splits/fusion360ass_j1_0_0_train_raw.txt \
  data/splits/fusion360ass_j1_0_0_val_raw.txt \
  data/splits/fusion360ass_j1_0_0_test_raw.txt

echo "[$(date --iso-8601=seconds)] Validating FabWave cache"
"$PYTHON_BIN" -m blendit.data.load_data --config data/fabwave.yaml --workers "$WORKERS"

PRETRAIN_NAME="joint_fusion_gallery_mlp_all_unlabeled_${PIPELINE_TAG}"
echo "[$(date --iso-8601=seconds)] Starting four-GPU joint pretraining: $PRETRAIN_NAME"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config configs/pretrain_joint_fusion_gallery_mlp_all_splits.yaml \
  --override "run.name=${PRETRAIN_NAME}"

PRETRAIN_RUN="$(find runs/pretrain -mindepth 1 -maxdepth 1 -type d \
  -name "*_${PRETRAIN_NAME}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
PRETRAIN_CHECKPOINT="${PRETRAIN_RUN}/checkpoints/last.pt"
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi

names=(mfcadpp tmcad fusion360seg_s2_0_0 fabwave)
configs=(
  configs/finetune_joint_mfcadpp_diffloss_200.yaml
  configs/finetune_joint_tmcad_diffloss_200.yaml
  configs/finetune_joint_fusion360seg_diffloss_200.yaml
  configs/finetune_joint_fabwave_diffloss_200.yaml
)

pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${GPUS[$index]}"
  config="${configs[$index]}"
  fine_name="${name}_diffloss_${PIPELINE_TAG}"
  fine_log="$LOG_DIR/${PIPELINE_TAG}_${name}_diffloss.log"
  echo "[$(date --iso-8601=seconds)] Starting $name DiffLoss on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=${fine_name}" \
    --override "train.pretrain_checkpoint=${PRETRAIN_CHECKPOINT}" \
    >"$fine_log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[$(date --iso-8601=seconds)] Completed: ${names[$index]}"
  else
    echo "[$(date --iso-8601=seconds)] Failed: ${names[$index]}" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] Evaluating four best checkpoints on test splits"
pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  gpu="${GPUS[$index]}"
  fine_name="${name}_diffloss_${PIPELINE_TAG}"
  fine_run="$(find runs/finetune -mindepth 1 -maxdepth 1 -type d \
    -name "*_${fine_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ ! -f "$fine_run/checkpoints/best.pt" ]]; then
    echo "Best checkpoint not found for $name: $fine_run" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$fine_run/checkpoints/best.pt" \
    --split test \
    --output "$fine_run/test_metrics.json" \
    --batch-size 512 \
    --num-workers 8 \
    --device cuda \
    >"$fine_run/test_evaluate.log" 2>&1 &
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
