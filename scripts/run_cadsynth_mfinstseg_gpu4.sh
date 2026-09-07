#!/usr/bin/env bash
# Build both OCC caches in parallel, then fine-tune and evaluate sequentially on physical GPU 4.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nvme03/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_ID="${GPU_ID:-4}"
CACHE_WORKERS="${CACHE_WORKERS:-16}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/cadsynth_mfinstseg}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/cadsynth_mfinstseg_$RUN_TAG}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/fewshot/pretrain/20260831-141228_fewshot_encoder_pretrain50_b128_resume_e010_gpu4_20260831/checkpoints/last.pt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Building CADSynth and MFInstSeg caches"
prepare_cache() {
  local dataset="$1"
  local config="data/${dataset}.yaml"
  local invalid_log="cache/features/${dataset}_invalid.jsonl"
  local log="$LOG_ROOT/${dataset}_cache.log"

  if "$PYTHON_BIN" -m brepprediff.data.load_data \
    --config "$config" --workers "$CACHE_WORKERS" >"$log" 2>&1; then
    return 0
  fi

  echo "[$(date --iso-8601=seconds)] Filtering invalid $dataset samples" | tee -a "$log"
  for split in train val test; do
    "$PYTHON_BIN" -m brepprediff.data.load_data filter-split \
      --split "data/splits/${dataset}_${split}.txt" \
      --invalid-log "$invalid_log" \
      --output "data/splits/${dataset}_${split}_clean.txt" \
      --removed-output "data/splits/${dataset}_${split}_removed.txt" \
      >>"$log" 2>&1
  done
  "$PYTHON_BIN" -m brepprediff.data.load_data \
    --config "$config" --workers "$CACHE_WORKERS" \
    --override "data.train_split=data/splits/${dataset}_train_clean.txt" \
    --override "data.val_split=data/splits/${dataset}_val_clean.txt" \
    --override "data.test_split=data/splits/${dataset}_test_clean.txt" \
    >>"$log" 2>&1
  touch "$LOG_ROOT/${dataset}_uses_clean_splits"
}

prepare_cache cadsynth &
cadsynth_cache_pid=$!
prepare_cache mfinstseg &
mfinstseg_cache_pid=$!

cache_status=0
wait "$cadsynth_cache_pid" || cache_status=1
wait "$mfinstseg_cache_pid" || cache_status=1
if [[ "$cache_status" -ne 0 ]]; then
  echo "Cache preparation failed; inspect $LOG_ROOT/*_cache.log" >&2
  exit "$cache_status"
fi

while ! timeout 30s nvidia-smi -i "$GPU_ID" >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] GPU $GPU_ID unavailable; retrying in 60 seconds"
  sleep 60
done

run_dataset() {
  local dataset="$1"
  local config="$2"
  local run_name="${dataset}_mlp_default_gpu${GPU_ID}_${RUN_TAG}"
  local train_log="$LOG_ROOT/${dataset}_train.log"
  local split_overrides=()
  if [[ -f "$LOG_ROOT/${dataset}_uses_clean_splits" ]]; then
    split_overrides=(
      --override "data.train_split=data/splits/${dataset}_train_clean.txt"
      --override "data.val_split=data/splits/${dataset}_val_clean.txt"
      --override "data.test_split=data/splits/${dataset}_test_clean.txt"
    )
  fi

  echo "[$(date --iso-8601=seconds)] Fine-tuning $dataset on physical GPU $GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m brepprediff.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    "${split_overrides[@]}" \
    >"$train_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $dataset: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 512 \
    --num-workers 16 \
    --device cuda \
    >>"$train_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $dataset: $run_dir"
}

run_dataset cadsynth "$ROOT_DIR/configs/finetune_cadsynth.yaml"
run_dataset mfinstseg "$ROOT_DIR/configs/finetune_mfinstseg.yaml"
echo "[$(date --iso-8601=seconds)] CADSynth and MFInstSeg pipeline completed"
