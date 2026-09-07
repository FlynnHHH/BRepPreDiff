#!/usr/bin/env bash
# Sequential 10/20-shot fine-tuning on physical GPU 4 after local pretraining.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/scripts/run_blendit_gpu4.sh}"
GPU_ID="${GPU_ID:-4}"
PRETRAIN_RUN="${PRETRAIN_RUN:-$ROOT_DIR/runs/fewshot/pretrain/20260831-111759_fewshot_encoder_pretrain50_gpu4_20260831}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$PRETRAIN_RUN/checkpoints/last.pt}"
RUN_TAG="${RUN_TAG:-fewshot_pre50_seed42_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/fewshot}"
SPLIT_ROOT="$ROOT_DIR/data/splits/fewshot/seed42"
RESULT_ROOT="$RUN_ROOT/results/$RUN_TAG"
LOG_ROOT="$RUN_ROOT/launch_logs/$RUN_TAG"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"

mkdir -p "$SPLIT_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GPU 4 launcher not found: $PYTHON_BIN" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] waiting for pretrain checkpoint=$PRETRAIN_CHECKPOINT"
while [[ ! -f "$PRETRAIN_CHECKPOINT" ]] || ! grep -q "finished pretraining" "$PRETRAIN_RUN/logs/pretrain.log" 2>/dev/null; do
  sleep 60
done
echo "[$(date --iso-8601=seconds)] pretraining complete"

tasks=(blendit_seg fusion360seg mfcadpp_seg tmcad_cls)
configs=(
  configs/finetune_joint_brepprediff_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
)
batches=(32 32 32 16)

for index in "${!tasks[@]}"; do
  conda run --no-capture-output -n blendit python scripts/create_fewshot_splits.py \
    --config "${configs[$index]}" \
    --task-name "${tasks[$index]}" \
    --shots 10 20 \
    --seed 42 \
    --output-dir "$SPLIT_ROOT"
done

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
    local run_name="${task}_${shot}shot_pre50_ft100_seed42_${RUN_TAG}"
    local task_result="$RESULT_ROOT/$task/${shot}shot"
    local task_log="$LOG_ROOT/${task}_${shot}shot.log"
    local run_dir
    mkdir -p "$task_result"
    cp "${split%.txt}.audit.json" "$task_result/selection.audit.json"

    echo "[$(date --iso-8601=seconds)] train task=$task shot=$shot gpu=$GPU_ID"
    GPU_ID="$GPU_ID" "$PYTHON_BIN" finetune \
      --config "$config" \
      --override "run.name=$run_name" \
      --override "run.output_dir=$RUN_ROOT" \
      --override "run.save_every_epochs=10" \
      --override "run.show_progress=false" \
      --override "seed=42" \
      --override "data.train_split=$split" \
      --override "model.encoder_type=edge_update_attention" \
      --override "model.num_heads=4" \
      --override "model.finetune_head=mlp" \
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
      echo "Best checkpoint not found for task=$task shot=$shot" >&2
      exit 1
    fi
    echo "[$(date --iso-8601=seconds)] evaluate task=$task shot=$shot checkpoint=$run_dir/checkpoints/best.pt"
    CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
      python -m brepprediff.training.evaluate \
      --checkpoint "$run_dir/checkpoints/best.pt" \
      --split test \
      --output "$task_result/test.json" \
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
    echo "At least one ${shot}-shot task failed; inspect $LOG_ROOT" >&2
    exit "$status"
  fi
done

conda run --no-capture-output -n blendit python scripts/summarize_fewshot.py \
  --results-root "$RESULT_ROOT" \
  --csv "$RESULT_ROOT/results.csv" \
  --markdown "$REPORT_PATH"
echo "[$(date --iso-8601=seconds)] few-shot suite complete report=$REPORT_PATH"
