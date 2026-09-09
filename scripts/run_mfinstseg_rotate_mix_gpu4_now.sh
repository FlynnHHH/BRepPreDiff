#!/usr/bin/env bash
# Pull the final MFInstSeg pair forward while the main GPU-4 pipeline waits on CADSynth.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/inductive9_rotate_mix}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/$RUN_TAG}"
GPU_ID="${GPU_ID:-4}"
LAUNCHER_PID="$(tr -d '[:space:]' < "$LOG_ROOT/launcher.pid")"
PRETRAIN_CHECKPOINT="$(tr -d '\n' < "$LOG_ROOT/pretrain_checkpoint")"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

resume_launcher() {
  if kill -0 "$LAUNCHER_PID" 2>/dev/null; then
    kill -CONT "$LAUNCHER_PID"
  fi
}
trap resume_launcher EXIT

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_head() {
  local head="$1" config finetune_head
  if [[ "$head" == "mlp" ]]; then
    config="configs/finetune_mfinstseg.yaml"
    finetune_head="mlp"
  else
    config="configs/finetune_mfinstseg_diffloss.yaml"
    finetune_head="diffusion"
  fi
  local run_name="full_mfinstseg_seg_${head}_ft200_seed42_${RUN_TAG}"
  local task_log="$LOG_ROOT/mfinstseg_seg_${head}.log"
  local args=(
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
    --override "train.epochs=200"
    --override "train.batch_size=256"
    --override "train.gradient_accumulation_steps=2"
    --override "train.rotation_augmentation_probability=0.5"
    --override "train.num_workers=16"
    --override "train.dataloader_seed=42"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=false"
    --override "train.resume=null"
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

  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" \
    >"$task_log" 2>&1
  local run_dir checkpoint
  run_dir="$(latest_run "*_${run_name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" --split test \
    --output "$run_dir/test_metrics.json" --batch-size 256 \
    --num-workers 16 --device cuda >>"$task_log" 2>&1
}

kill -STOP "$LAUNCHER_PID"
run_head mlp & mlp_pid=$!
run_head diffloss & diffloss_pid=$!
status=0
wait "$mlp_pid" || status=1
wait "$diffloss_pid" || status=1
exit "$status"
