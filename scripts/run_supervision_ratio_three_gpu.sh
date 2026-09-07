#!/usr/bin/env bash
# One benchmark per GPU; seven nested labeled-data ratios per benchmark.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nvme03/hhfeng/miniconda3/envs/blendit/bin/python}"
GPUS=(${GPUS:-1 2 3})
RUN_TAG="${RUN_TAG:-supervision_ratios_mlp200_seed42_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/supervision_ratios}"
RESULT_ROOT="$RUN_ROOT/results/$RUN_TAG"
LOG_ROOT="$RUN_ROOT/launch_logs/$RUN_TAG"
SPLIT_ROOT="$ROOT_DIR/data/splits/supervision_ratios/seed42"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/inductive_lr1e4_encoder/pretrain/20260901-115311_inductive9_no_fab_encoder_lr1e4_e100_seed42_20260901/checkpoints/last.pt}"
RATIOS=(0.1 0.5 1 1.5 2 3 100)

if [[ ${#GPUS[@]} -ne 3 ]]; then
  echo "Exactly three GPU IDs are required" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" || ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Missing Python or pretrain checkpoint" >&2
  exit 2
fi
mkdir -p "$RESULT_ROOT" "$LOG_ROOT" "$SPLIT_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
printf '%s\n' "$RUN_TAG" >"$LOG_ROOT/run_tag"
printf '%s\n' "$REPORT_PATH" >"$LOG_ROOT/report_path"
printf '%s\n' "$PRETRAIN_CHECKPOINT" >"$LOG_ROOT/pretrain_checkpoint"

tasks=(mfinstseg mfcadpp cadsynth)
display_names=(MFInstSeg MFCAD++ CADSynth)
configs=(configs/finetune_mfinstseg.yaml configs/finetune_joint_mfcadpp_mlp.yaml configs/finetune_cadsynth.yaml)
source_splits=(data/splits/mfinstseg_train.txt data/splits/mfcad_train.txt data/splits/cadsynth_train_clean.txt)
batches=(512 256 512)
val_splits=(data/splits/mfinstseg_val.txt data/splits/mfcad_val.txt data/splits/cadsynth_val_clean.txt)
test_splits=(data/splits/mfinstseg_test.txt data/splits/mfcad_test.txt data/splits/cadsynth_test_clean.txt)

for i in "${!tasks[@]}"; do
  "$PYTHON_BIN" scripts/create_supervision_ratio_splits.py \
    --config "${configs[$i]}" --task-name "${display_names[$i]}" \
    --source-split "${source_splits[$i]}" --ratios "${RATIOS[@]}" \
    --seed 42 --output-dir "$SPLIT_ROOT"
done

latest_run() {
  local pattern="$1"
  if [[ ! -d "$RUN_ROOT/finetune" ]]; then
    return 0
  fi
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_benchmark() {
  local i="$1" gpu="${GPUS[$1]}" task="${tasks[$1]}" display="${display_names[$1]}"
  local config="${configs[$1]}" batch="${batches[$1]}" ratio tag split run_name run_dir result_dir log
  for ratio in "${RATIOS[@]}"; do
    tag="${ratio//./p}"
    split="$SPLIT_ROOT/${display}_r${tag}_train.txt"
    run_name="${task}_mlp_r${tag}_ft200_seed42_${RUN_TAG}"
    result_dir="$RESULT_ROOT/$task/r${tag}"
    log="$LOG_ROOT/${task}_r${tag}.log"
    mkdir -p "$result_dir"
    cp "${split%.txt}.audit.json" "$result_dir/selection.audit.json"
    run_dir="$(latest_run "*_${run_name}")"
    if [[ -f "$result_dir/test_metrics.json" ]]; then
      echo "[$(date --iso-8601=seconds)] skip completed task=$task ratio=$ratio%"
      continue
    fi
    args=(
      --config "$config"
      --override "run.name=$run_name"
      --override "run.output_dir=$RUN_ROOT"
      --override "run.save_every_epochs=10"
      --override "run.show_progress=false"
      --override "seed=42"
      --override "data.train_split=$split"
      --override "data.val_split=${val_splits[$i]}"
      --override "data.test_split=${test_splits[$i]}"
      --override "model.encoder_type=edge_update_attention"
      --override "model.num_heads=4"
      --override "model.finetune_head=mlp"
      --override "brep.edge_u_grid_size=10"
      --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
      --override "train.epochs=200"
      --override "train.batch_size=$batch"
      --override "train.gradient_accumulation_steps=1"
      --override "train.num_workers=16"
      --override "train.dataloader_seed=42"
      --override "train.validate_every_epochs=5"
      --override "wandb.enabled=false"
    )
    resume_checkpoint=""
    if [[ -n "$run_dir" ]]; then
      if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
        resume_checkpoint="$run_dir/checkpoints/last.pt"
      else
        resume_checkpoint="$(find "$run_dir/checkpoints" -maxdepth 1 -type f \
          -name 'epoch_*.pt' | sort | tail -n 1)"
      fi
    fi
    if [[ -n "$resume_checkpoint" ]]; then
      args+=(--override "train.resume=$resume_checkpoint")
    else
      args+=(--override "train.resume=null")
    fi
    echo "[$(date --iso-8601=seconds)] train task=$task ratio=$ratio% gpu=$gpu"
    GPU_ID="$gpu" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" >"$log" 2>&1
    run_dir="$(latest_run "*_${run_name}")"
    if [[ -z "$run_dir" || ! -f "$run_dir/checkpoints/best.pt" ]]; then
      echo "Missing best checkpoint task=$task ratio=$ratio%" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n blendit \
      python -m brepprediff.training.evaluate --checkpoint "$run_dir/checkpoints/best.pt" \
      --split test --output "$result_dir/test_metrics.json" --batch-size "$batch" \
      --num-workers 16 --device cuda >>"$log" 2>&1
    cp "$result_dir/test_metrics.json" "$run_dir/test_metrics.json"
    echo "[$(date --iso-8601=seconds)] complete task=$task ratio=$ratio% run=$run_dir"
  done
}

pids=()
for i in "${!tasks[@]}"; do
  run_benchmark "$i" >"$LOG_ROOT/${tasks[$i]}_queue.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "At least one benchmark queue failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi
"$PYTHON_BIN" scripts/summarize_supervision_ratios.py \
  --results-root "$RESULT_ROOT" --csv "$RESULT_ROOT/results.csv" \
  --markdown "$REPORT_PATH" --pretrain-checkpoint "$PRETRAIN_CHECKPOINT"
echo "[$(date --iso-8601=seconds)] suite complete report=$REPORT_PATH"
