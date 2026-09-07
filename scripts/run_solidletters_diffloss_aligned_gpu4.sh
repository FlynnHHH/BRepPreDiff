#!/usr/bin/env bash
# SolidLetters DiffLoss using the same x_start/1-step settings as the other tasks.
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
CONFIG="$ROOT_DIR/configs/finetune_solidletters_diffloss.yaml"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_shot() {
  local shot="$1"
  local split="$SPLIT_ROOT/solidletters_cls_${shot}shot_train.txt"
  local run_name="solidletters_cls_diffloss_aligned_${shot}shot_pre50_ft100_seed42_${RUN_TAG}"
  local result_dir="$RESULT_ROOT/solidletters_cls/diffloss_aligned/${shot}shot"
  local task_log="$LOG_ROOT/solidletters_cls_diffloss_aligned_${shot}shot.log"
  local run_dir

  mkdir -p "$result_dir"
  cp "${split%.txt}.audit.json" "$result_dir/selection.audit.json"
  echo "[$(date --iso-8601=seconds)] train task=solidletters_cls head=DiffLoss-aligned shot=$shot"
  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune \
    --config "$CONFIG" \
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
    --override "label_diffusion.prediction_type=x_start" \
    --override "label_diffusion.x_start_loss_weight=1.0" \
    --override "label_diffusion.epsilon_loss_weight=0.0" \
    --override "label_diffusion.sampling_steps=1" \
    --override "label_diffusion.sampling_temperature=0.75" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.resume=null" \
    --override "train.epochs=100" \
    --override "train.batch_size=256" \
    --override "train.gradient_accumulation_steps=1" \
    --override "train.num_workers=8" \
    --override "train.dataloader_seed=42" \
    --override "train.validate_every_epochs=5" \
    --override "wandb.enabled=false" \
    >"$task_log" 2>&1

  run_dir="$(latest_run "*_${run_name}")"
  if [[ -z "$run_dir" ]] || [[ ! -f "$run_dir/checkpoints/best.pt" ]]; then
    echo "Best checkpoint not found for SolidLetters aligned DiffLoss ${shot}-shot" >&2
    return 1
  fi
  echo "[$(date --iso-8601=seconds)] evaluate task=solidletters_cls head=DiffLoss-aligned shot=$shot"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test \
    --output "$result_dir/test.json" \
    --batch-size 256 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
}

pids=()
for shot in 10 20; do
  run_shot "$shot" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "SolidLetters aligned DiffLoss failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

conda run --no-capture-output -n blendit python scripts/summarize_fewshot.py \
  --results-root "$RESULT_ROOT" \
  --csv "$RESULT_ROOT/results.csv" \
  --markdown "$REPORT_PATH"
echo "[$(date --iso-8601=seconds)] SolidLetters aligned DiffLoss complete report=$REPORT_PATH"
