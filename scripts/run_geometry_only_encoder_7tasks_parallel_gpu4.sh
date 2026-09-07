#!/usr/bin/env bash
# Train the latest inductive9 encoder recipe without discrete pretraining losses,
# then launch MLP and DiffLoss fine-tuning for all seven downstream tasks in
# parallel on physical GPU 4.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
PRETRAIN_LR_SCHEDULER="${PRETRAIN_LR_SCHEDULER:-constant}"
PRETRAIN_MIN_LR="${PRETRAIN_MIN_LR:-0.0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
RUN_TAG="${RUN_TAG:-geometry_only_inductive9_${PRETRAIN_LR_SCHEDULER}_pre100_ft${FINETUNE_EPOCHS}_seed42_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/geometry_only_inductive9}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/$RUN_TAG}"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"
PRETRAIN_NAME="geometry_only_inductive9_no_fab_lr1e4_${PRETRAIN_LR_SCHEDULER}_e100_seed42_$RUN_TAG"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-brepprediff}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ "$WANDB_ENABLED" == "true" ]]; then
  unset WANDB_MODE
else
  export WANDB_MODE=disabled
fi

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

while ! timeout 30s nvidia-smi -i "$GPU_ID" >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] GPU $GPU_ID unavailable; retrying in 60 seconds"
  sleep 60
done

pretrain_run="$(latest_run pretrain "*_${PRETRAIN_NAME}")"
if [[ -n "$pretrain_run" && -f "$pretrain_run/checkpoints/last.pt" ]] \
  && grep -q 'finished pretraining' "$pretrain_run/logs/pretrain.log" 2>/dev/null; then
  echo "[$(date --iso-8601=seconds)] skipping completed pretrain run=$pretrain_run"
else
  pretrain_args=(
    --config "$ROOT_DIR/configs/pretrain.yaml"
    --override "run.name=$PRETRAIN_NAME"
    --override "run.output_dir=$RUN_ROOT"
    --override "run.save_every_epochs=5"
    --override "run.show_progress=false"
    --override "seed=42"
    --override "diffusion.categorical_loss_weight=0.0"
    --override "diffusion.relation_loss_weight=0.0"
    --override "train.epochs=100"
    --override "train.batch_size=128"
    --override "train.num_workers=16"
    --override "train.lr=1.0e-4"
    --override "train.lr_scheduler=$PRETRAIN_LR_SCHEDULER"
    --override "train.min_lr=$PRETRAIN_MIN_LR"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=$WANDB_ENABLED"
    --override "wandb.project=$WANDB_PROJECT"
    --override "wandb.group=$RUN_TAG"
    --override "wandb.name=$PRETRAIN_NAME"
    --override "wandb.tags=[inductive,geometry_only,pretrain100,no_fabwave,$PRETRAIN_LR_SCHEDULER,gpu4]"
  )
  if [[ -n "$pretrain_run" && -f "$pretrain_run/checkpoints/last.pt" ]]; then
    echo "[$(date --iso-8601=seconds)] resuming geometry-only pretrain checkpoint=$pretrain_run/checkpoints/last.pt"
    pretrain_args+=(--override "train.resume=$pretrain_run/checkpoints/last.pt")
  else
    echo "[$(date --iso-8601=seconds)] starting geometry-only pretrain epochs=100 gpu=$GPU_ID"
    pretrain_args+=(--override "train.resume=null")
  fi
  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" pretrain "${pretrain_args[@]}" \
    >"$LOG_ROOT/pretrain.log" 2>&1
  pretrain_run="$(latest_run pretrain "*_${PRETRAIN_NAME}")"
fi

PRETRAIN_CHECKPOINT="$pretrain_run/checkpoints/last.pt"
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretrain checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi
printf '%s\n' "$PRETRAIN_CHECKPOINT" >"$LOG_ROOT/pretrain_checkpoint"
echo "[$(date --iso-8601=seconds)] geometry-only pretrain complete checkpoint=$PRETRAIN_CHECKPOINT"

tasks=(
  brepprediff_seg fusion360seg mfcadpp_seg tmcad_cls solidletters_cls
  cadsynth_seg mfinstseg_seg
)
mlp_configs=(
  configs/finetune_joint_brepprediff_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
  configs/finetune_solidletters_mlp.yaml
  configs/finetune_cadsynth.yaml
  configs/finetune_mfinstseg.yaml
)
diff_configs=(
  configs/finetune_joint_brepprediff_diffloss_200.yaml
  configs/finetune_joint_fusion360seg_diffloss_200.yaml
  configs/finetune_joint_mfcadpp_diffloss_200.yaml
  configs/finetune_joint_tmcad_diffloss_200.yaml
  configs/finetune_solidletters_diffloss.yaml
  configs/finetune_cadsynth_diffloss.yaml
  configs/finetune_mfinstseg_diffloss.yaml
)
batches=(256 256 256 256 256 512 512)
display_names=(BRepPreDiff Fusion360Seg MFCAD++ TMCAD SolidLetters CADSynth MFInstSeg)

run_downstream() {
  local index="$1" head="$2"
  local task="${tasks[$index]}" config batch finetune_head run_name run_dir checkpoint resume_checkpoint
  local task_log="$LOG_ROOT/${task}_${head}.log"
  local split_args=()
  if [[ "$head" == "mlp" ]]; then
    config="${mlp_configs[$index]}"
    finetune_head="mlp"
  else
    config="${diff_configs[$index]}"
    finetune_head="diffusion"
  fi
  batch="${batches[$index]}"
  run_name="geometry_only_${task}_${head}_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG"
  run_dir="$(latest_run finetune "*_${run_name}")"

  if [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]]; then
    echo "[$(date --iso-8601=seconds)] skipping completed task=$task head=$head run=$run_dir"
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
    --override "model.finetune_head=$finetune_head"
    --override "model.graph_pooling=mean_max"
    --override "brep.edge_u_grid_size=10"
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
    --override "train.epochs=$FINETUNE_EPOCHS"
    --override "train.batch_size=$batch"
    --override "train.gradient_accumulation_steps=1"
    --override "train.num_workers=16"
    --override "train.dataloader_seed=42"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=$WANDB_ENABLED"
    --override "wandb.project=$WANDB_PROJECT"
    --override "wandb.group=$RUN_TAG"
    --override "wandb.name=$run_name"
    --override "wandb.tags=[full_data,$head,finetune${FINETUNE_EPOCHS},geometry_only_encoder,gpu4,parallel]"
    "${split_args[@]}"
  )
  if [[ "$head" == "diffloss" ]]; then
    args+=(
      --override "label_diffusion.prediction_type=x_start"
      --override "label_diffusion.x_start_loss_weight=1.0"
      --override "label_diffusion.epsilon_loss_weight=0.0"
      --override "label_diffusion.sampling_steps=1"
      --override "label_diffusion.sampling_temperature=0.75"
      --override "label_diffusion.seed=42"
    )
  fi
  resume_checkpoint=""
  if [[ -n "$run_dir" && -f "$run_dir/checkpoints/last.pt" ]]; then
    resume_checkpoint="$run_dir/checkpoints/last.pt"
  elif [[ -n "$run_dir" && -d "$run_dir/checkpoints" ]]; then
    resume_checkpoint="$(find "$run_dir/checkpoints" -maxdepth 1 -type f \
      -name 'epoch_*.pt' -print | sort -V | tail -n 1)"
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    echo "[$(date --iso-8601=seconds)] resuming task=$task head=$head checkpoint=$resume_checkpoint"
    args+=(--override "train.resume=$resume_checkpoint")
  else
    echo "[$(date --iso-8601=seconds)] training task=$task head=$head epochs=$FINETUNE_EPOCHS gpu=$GPU_ID"
    args+=(--override "train.resume=null")
  fi

  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" \
    >"$task_log" 2>&1
  run_dir="$(latest_run finetune "*_${run_name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found task=$task head=$head path=$checkpoint" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" --split test \
    --output "$run_dir/test_metrics.json" --batch-size "$batch" \
    --num-workers 16 --device cuda >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed task=$task head=$head run=$run_dir"
}

# The user explicitly requested all 14 downstream jobs to share GPU 4 in parallel.
pids=()
for index in "${!tasks[@]}"; do
  run_downstream "$index" mlp & pids+=("$!")
  run_downstream "$index" diffloss & pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one downstream run failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

report_args=()
for index in "${!tasks[@]}"; do
  mlp_run="$(latest_run finetune "*_geometry_only_${tasks[$index]}_mlp_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG")"
  diff_run="$(latest_run finetune "*_geometry_only_${tasks[$index]}_diffloss_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG")"
  report_args+=(--pair "${display_names[$index]}" "$mlp_run" "$diff_run")
done
conda run --no-capture-output -n blendit python scripts/compare_new_occ_all_heads.py \
  "${report_args[@]}" \
  --title "Geometry-only inductive 9-source encoder：7 个下游 MLP 与 DiffLoss 对照" \
  --epochs "$FINETUNE_EPOCHS" \
  --batch-description "task batch size 256/512、梯度累积 1、GPU 4 并行" \
  --pretrain-description "100-epoch inductive 9-source（无 FabWave）encoder；离散属性 loss 权重为 0；预训练 LR scheduler=$PRETRAIN_LR_SCHEDULER、min_lr=$PRETRAIN_MIN_LR" \
  --output "$REPORT_PATH" >"$LOG_ROOT/comparison.log" 2>&1

echo "[$(date --iso-8601=seconds)] pipeline complete report=$REPORT_PATH"
