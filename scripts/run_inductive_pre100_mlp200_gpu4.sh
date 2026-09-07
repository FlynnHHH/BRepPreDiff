#!/usr/bin/env bash
# Automated GPU-4 pipeline: resume inductive pretraining from epoch 50 to 100,
# then run full-data 200-epoch MLP fine-tuning/evaluation with W&B enabled.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
RUN_TAG="${RUN_TAG:-inductive9_no_fab_pre100_mlp200_seed42_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/inductive_pre100_mlp200}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/$RUN_TAG}"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"
BASE_PRETRAIN_CHECKPOINT="${BASE_PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/inductive_full/pretrain/20260831-213902_inductive9_no_fab_pre50_seed42_inductive9_no_fab_full_pre50_ft100_seed42_20260831/checkpoints/last.pt}"
PRETRAIN_NAME="inductive9_no_fab_pre100_seed42_$RUN_TAG"
WANDB_PROJECT="${WANDB_PROJECT:-brepprediff}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset WANDB_MODE

printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
printf '%s\n' "$RUN_TAG" >"$LOG_ROOT/run_tag"
printf '%s\n' "$REPORT_PATH" >"$LOG_ROOT/report_path"

latest_run() {
  local stage="$1" pattern="$2"
  if [[ ! -d "$RUN_ROOT/$stage" ]]; then
    return 0
  fi
  find "$RUN_ROOT/$stage" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

if [[ ! -f "$BASE_PRETRAIN_CHECKPOINT" ]]; then
  echo "Base epoch-50 checkpoint not found: $BASE_PRETRAIN_CHECKPOINT" >&2
  exit 2
fi
while ! timeout 30s nvidia-smi -i "$GPU_ID" >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] GPU $GPU_ID unavailable; retrying in 60 seconds"
  sleep 60
done

pretrain_run="$(latest_run pretrain "*_${PRETRAIN_NAME}")"
if [[ -n "$pretrain_run" && -f "$pretrain_run/checkpoints/last.pt" ]] \
  && grep -q 'finished pretraining' "$pretrain_run/logs/pretrain.log" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] skipping completed pretrain run=$pretrain_run"
else
  resume_checkpoint="$BASE_PRETRAIN_CHECKPOINT"
  if [[ -n "$pretrain_run" && -f "$pretrain_run/checkpoints/last.pt" ]]; then
    resume_checkpoint="$pretrain_run/checkpoints/last.pt"
  fi
  echo "[$(date --iso-8601=seconds)] resuming inductive pretrain to epoch 100 checkpoint=$resume_checkpoint"
  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" pretrain \
    --config "$ROOT_DIR/configs/pretrain.yaml" \
    --override "run.name=$PRETRAIN_NAME" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "run.save_every_epochs=5" \
    --override "run.show_progress=false" \
    --override "seed=42" \
    --override "train.epochs=100" \
    --override "train.batch_size=128" \
    --override "train.num_workers=16" \
    --override "train.validate_every_epochs=1" \
    --override "train.resume=$resume_checkpoint" \
    --override "wandb.enabled=true" \
    --override "wandb.project=$WANDB_PROJECT" \
    --override "wandb.group=$RUN_TAG" \
    --override "wandb.name=$PRETRAIN_NAME" \
    --override "wandb.tags=[inductive,pretrain100,no_fabwave,gpu4]" \
    >"$LOG_ROOT/pretrain.log" 2>&1
  pretrain_run="$(latest_run pretrain "*_${PRETRAIN_NAME}")"
fi

PRETRAIN_CHECKPOINT="$pretrain_run/checkpoints/last.pt"
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Epoch-100 pretrain checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi
printf '%s\n' "$PRETRAIN_CHECKPOINT" >"$LOG_ROOT/pretrain_checkpoint"
echo "[$(date --iso-8601=seconds)] epoch-100 pretrain complete checkpoint=$PRETRAIN_CHECKPOINT"

tasks=(
  brepprediff_seg fusion360seg mfcadpp_seg tmcad_cls solidletters_cls
  cadsynth_seg mfinstseg_seg
)
configs=(
  configs/finetune_joint_brepprediff_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
  configs/finetune_solidletters_mlp.yaml
  configs/finetune_cadsynth.yaml
  configs/finetune_mfinstseg.yaml
)
batches=(256 256 256 256 256 512 512)
display_names=(BRepPreDiff Fusion360Seg MFCAD++ TMCAD SolidLetters CADSynth MFInstSeg)

run_downstream() {
  local index="$1"
  local task="${tasks[$index]}" config="${configs[$index]}" batch="${batches[$index]}"
  local run_name="full_${task}_mlp_ft200_pre100_seed42_$RUN_TAG"
  local task_log="$LOG_ROOT/${task}_mlp.log"
  local run_dir checkpoint
  local split_args=()
  run_dir="$(latest_run finetune "*_${run_name}")"

  if [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]]; then
    echo "[$(date --iso-8601=seconds)] skipping completed task=$task run=$run_dir"
    return
  fi
  if [[ "$task" == "cadsynth_seg" ]]; then
    split_args=(
      --override "data.train_split=data/splits/cadsynth_train_clean.txt"
      --override "data.val_split=data/splits/cadsynth_val_clean.txt"
      --override "data.test_split=data/splits/cadsynth_test_clean.txt"
    )
  fi

  args=(
    --config "$ROOT_DIR/$config"
    --override "run.name=$run_name"
    --override "run.output_dir=$RUN_ROOT"
    --override "run.save_every_epochs=10"
    --override "run.show_progress=false"
    --override "seed=42"
    --override "model.encoder_type=edge_update_attention"
    --override "model.num_heads=4"
    --override "model.finetune_head=mlp"
    --override "model.graph_pooling=mean_max"
    --override "brep.edge_u_grid_size=10"
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
    --override "train.epochs=200"
    --override "train.batch_size=$batch"
    --override "train.gradient_accumulation_steps=1"
    --override "train.num_workers=16"
    --override "train.dataloader_seed=42"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=true"
    --override "wandb.project=$WANDB_PROJECT"
    --override "wandb.group=$RUN_TAG"
    --override "wandb.name=$run_name"
    --override "wandb.tags=[full_data,mlp,finetune200,pretrain100,gpu4]"
    "${split_args[@]}"
  )
  if [[ -n "$run_dir" && -f "$run_dir/checkpoints/last.pt" ]]; then
    echo "[$(date --iso-8601=seconds)] resuming task=$task from $run_dir/checkpoints/last.pt"
    args+=(--override "train.resume=$run_dir/checkpoints/last.pt")
  else
    echo "[$(date --iso-8601=seconds)] training task=$task head=mlp epochs=200 gpu=$GPU_ID"
    args+=(--override "train.resume=null")
  fi

  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" \
    >"$task_log" 2>&1
  run_dir="$(latest_run finetune "*_${run_name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found task=$task path=$checkpoint" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" --split test \
    --output "$run_dir/test_metrics.json" --batch-size "$batch" \
    --num-workers 16 --device cuda >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed task=$task head=mlp run=$run_dir"
}

for index in "${!tasks[@]}"; do
  run_downstream "$index"
done

report_args=()
for index in "${!tasks[@]}"; do
  run_dir="$(latest_run finetune "*_full_${tasks[$index]}_mlp_ft200_pre100_seed42_$RUN_TAG")"
  report_args+=(--run "${display_names[$index]}" "$run_dir")
done
conda run --no-capture-output -n blendit python scripts/summarize_full_mlp.py \
  "${report_args[@]}" --pretrain-checkpoint "$PRETRAIN_CHECKPOINT" \
  --output "$REPORT_PATH" >"$LOG_ROOT/summary.log" 2>&1

echo "[$(date --iso-8601=seconds)] pipeline complete report=$REPORT_PATH"
