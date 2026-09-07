#!/usr/bin/env bash
# Run DiffLoss on the existing nested 10/20-shot splits and compare with MLP.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/fewshot/pretrain/20260831-141228_fewshot_encoder_pretrain50_b128_resume_e010_gpu4_20260831/checkpoints/last.pt}"
RUN_TAG="${RUN_TAG:-fewshot_pre50_b128_resume_e010_seed42_20260831}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/fewshot}"
RESULT_ROOT="$RUN_ROOT/results/$RUN_TAG"
LOG_ROOT="$RUN_ROOT/launch_logs/$RUN_TAG"
SPLIT_ROOT="$ROOT_DIR/data/splits/fewshot/seed42"
REPORT_PATH="$ROOT_DIR/reports/fewshot_pre50_b128_resume_e010_seed42_20260831.md"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretrain checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

tasks=(blendit_seg fusion360seg mfcadpp_seg tmcad_cls solidletters_cls)
configs=(
  configs/finetune_joint_brepprediff_diffloss_200.yaml
  configs/finetune_joint_fusion360seg_diffloss_200.yaml
  configs/finetune_joint_mfcadpp_diffloss_200.yaml
  configs/finetune_joint_tmcad_diffloss_200.yaml
  configs/finetune_solidletters_diffloss.yaml
)
batches=(32 32 32 16 256)

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_task() {
  local shot="$1" index="$2"
  local task="${tasks[$index]}"
  local config="${configs[$index]}"
  local batch="${batches[$index]}"
  local split="$SPLIT_ROOT/${task}_${shot}shot_train.txt"
  local run_name="${task}_diffloss_${shot}shot_pre50_ft100_seed42_${RUN_TAG}"
  local result_dir="$RESULT_ROOT/$task/diffloss/${shot}shot"
  local task_log="$LOG_ROOT/${task}_diffloss_${shot}shot.log"
  local run_dir

  if [[ ! -f "$split" ]]; then
    echo "Few-shot split not found: $split" >&2
    return 1
  fi
  mkdir -p "$result_dir"
  cp "${split%.txt}.audit.json" "$result_dir/selection.audit.json"
  echo "[$(date --iso-8601=seconds)] train task=$task head=DiffLoss shot=$shot gpu=$GPU_ID"
  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "run.save_every_epochs=10" \
    --override "run.show_progress=false" \
    --override "seed=42" \
    --override "data.train_split=$split" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.finetune_head=diffusion" \
    --override "model.graph_pooling=mean_max" \
    --override "brep.edge_u_grid_size=10" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.resume=null" \
    --override "train.epochs=100" \
    --override "train.batch_size=$batch" \
    --override "train.gradient_accumulation_steps=1" \
    --override "train.num_workers=8" \
    --override "train.dataloader_seed=42" \
    --override "train.validate_every_epochs=5" \
    --override "wandb.enabled=false" \
    >"$task_log" 2>&1

  run_dir="$(latest_run "*_${run_name}")"
  if [[ -z "$run_dir" ]] || [[ ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "Best checkpoint not found for task=$task head=DiffLoss shot=$shot" >&2
    return 1
  fi
  echo "[$(date --iso-8601=seconds)] evaluate task=$task head=DiffLoss shot=$shot"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test \
    --output "$result_dir/test.json" \
    --batch-size "$batch" \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
}

for shot in 10 20; do
  pids=()
  for index in "${!tasks[@]}"; do
    run_task "$shot" "$index" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "At least one ${shot}-shot DiffLoss task failed; inspect $LOG_ROOT" >&2
    exit "$status"
  fi
done

conda run --no-capture-output -n blendit python scripts/summarize_fewshot.py \
  --results-root "$RESULT_ROOT" \
  --csv "$RESULT_ROOT/results.csv" \
  --markdown "$REPORT_PATH"
echo "[$(date --iso-8601=seconds)] DiffLoss few-shot suite complete report=$REPORT_PATH"
