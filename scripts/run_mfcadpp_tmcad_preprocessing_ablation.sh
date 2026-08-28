#!/usr/bin/env bash
# Fine-tune and test preprocessing-matched checkpoints on MFCAD++ and TMCAD.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
LOG_DIR="${LOG_DIR:-runs/launch_logs}"
mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"
summary_log="$LOG_DIR/${timestamp}_mfcadpp_tmcad_preprocessing_ablation.summary.log"

checkpoint_for() {
  case "$1" in
    per_graph) echo "runs/pretrain/20260803-155054_joint_ablation_per_graph/checkpoints/last.pt" ;;
    none) echo "runs/pretrain/20260803-173114_joint_ablation_none/checkpoints/last.pt" ;;
    typewise_global) echo "runs/pretrain/20260803-190220_joint_ablation_typewise_global/checkpoints/last.pt" ;;
    *) return 2 ;;
  esac
}

run_job() {
  local gpu="$1"
  local dataset="$2"
  local mode="$3"
  local base_config run_name normalize checkpoint train_log
  if [[ "$dataset" == "mfcadpp" ]]; then
    base_config="configs/finetune_joint_mfcadpp_mlp.yaml"
  else
    base_config="configs/finetune_joint_tmcad_mlp.yaml"
  fi
  run_name="${dataset}_ablation_${mode}"
  checkpoint="$(checkpoint_for "$mode")"
  normalize=false
  if [[ "$mode" == "per_graph" ]]; then
    normalize=true
  fi
  train_log="$LOG_DIR/${timestamp}_${dataset}_${mode}_train.log"

  echo "$(date --iso-8601=seconds) starting dataset=$dataset mode=$mode gpu=$gpu" | tee -a "$summary_log"
  command=(
    "$PYTHON_BIN" -m brepprediff.training.finetune
    --config "$base_config"
    --override "run.name=$run_name"
    --override "train.pretrain_checkpoint=$checkpoint"
    --override "train.normalize_per_graph=$normalize"
    --override "train.feature_preprocessing.mode=$mode"
  )
  if [[ "$mode" == "typewise_global" ]]; then
    command+=(--override "train.feature_preprocessing.stats_path=runs/feature_stats/joint_all_splits_train.json")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 "${command[@]}" >"$train_log" 2>&1

  local run_dir
  run_dir="$(find runs/finetune -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  if [[ -z "$run_dir" || ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "No best checkpoint found for dataset=$dataset mode=$mode." >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 "$PYTHON_BIN" \
    -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 512 \
    --num-workers 8 \
    --device cuda >"$run_dir/test_evaluate.log" 2>&1
  echo "$(date --iso-8601=seconds) completed dataset=$dataset mode=$mode run_dir=$run_dir" | tee -a "$summary_log"
}

# Balance historical single-GPU runtimes across four GPU queues.
(run_job 0 mfcadpp per_graph; run_job 0 tmcad none) & queue0=$!
(run_job 1 mfcadpp none; run_job 1 tmcad typewise_global) & queue1=$!
run_job 2 mfcadpp typewise_global & queue2=$!
run_job 3 tmcad per_graph & queue3=$!

status=0
for pid in "$queue0" "$queue1" "$queue2" "$queue3"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
