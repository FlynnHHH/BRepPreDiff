#!/usr/bin/env bash
# Dispatch the unfinished lr=1e-3 downstream runs over every GPU except GPU 4.
# Each GPU executes its assigned queue sequentially to avoid same-device OOM.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-inductive9_lr1e3_constant_resume50_pre100_ft200_batched3_seed42_20260904}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/inductive_lr1e3_constant_pre100_ft200}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/${RUN_TAG}_7gpu}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$RUN_ROOT/pretrain/20260904-140032_inductive9_no_fab_full_lr1p0em3_constant_e100_seed42_${RUN_TAG}/checkpoints/last.pt}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-brepprediff}"
ORIGINAL_LAUNCHER_PID_FILE="${ORIGINAL_LAUNCHER_PID_FILE:-$ROOT_DIR/runs/launch_logs/$RUN_TAG/launcher.pid}"

mkdir -p "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"

if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretrain checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

latest_run() {
  local pattern="$1"
  if [[ ! -d "$RUN_ROOT/finetune" ]]; then
    return 0
  fi
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_task() {
  local gpu="$1" task="$2" head="$3" config="$4" batch="$5"
  local finetune_head run_name run_dir checkpoint resume_checkpoint task_log
  local split_args=() args=()

  run_name="full_${task}_${head}_ft${FINETUNE_EPOCHS}_seed42_${RUN_TAG}"
  task_log="$LOG_ROOT/${task}_${head}_gpu${gpu}.log"
  run_dir="$(latest_run "*_${run_name}")"
  if [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed task=$task head=$head gpu=$gpu run=$run_dir"
    return 0
  fi

  if [[ "$head" == "mlp" ]]; then
    finetune_head="mlp"
  else
    finetune_head="diffusion"
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
    "${split_args[@]}"
  )
  if [[ "$WANDB_ENABLED" == "true" ]]; then
    args+=(
      --override "wandb.project=$WANDB_PROJECT"
      --override "wandb.group=$RUN_TAG"
      --override "wandb.name=$run_name"
      --override "wandb.tags=[full_data,$head,finetune${FINETUNE_EPOCHS},gpu${gpu}]"
    )
  fi
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
    resume_checkpoint="$(find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' -print | sort -V | tail -n 1)"
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    echo "[$(date --iso-8601=seconds)] resume task=$task head=$head gpu=$gpu checkpoint=$resume_checkpoint"
    args+=(--override "train.resume=$resume_checkpoint")
  else
    echo "[$(date --iso-8601=seconds)] start task=$task head=$head gpu=$gpu epochs=$FINETUNE_EPOCHS"
    args+=(--override "train.resume=null")
  fi

  GPU_ID="$gpu" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" >"$task_log" 2>&1
  run_dir="$(latest_run "*_${run_name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found task=$task head=$head gpu=$gpu path=$checkpoint" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" --split test \
    --output "$run_dir/test_metrics.json" --batch-size "$batch" \
    --num-workers 16 --device cuda >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] complete task=$task head=$head gpu=$gpu run=$run_dir"
}

worker_gpu0() {
  run_task 0 fusion360seg diffloss configs/finetune_joint_fusion360seg_diffloss_200.yaml 256
  run_task 0 solidletters_cls mlp configs/finetune_solidletters_mlp.yaml 256
  run_task 0 solidletters_cls diffloss configs/finetune_solidletters_diffloss.yaml 256
}
worker_gpu1() {
  run_task 1 mfcadpp_seg mlp configs/finetune_joint_mfcadpp_mlp.yaml 256
  run_task 1 tmcad_cls mlp configs/finetune_joint_tmcad_mlp.yaml 256
}
worker_gpu2() {
  run_task 2 mfcadpp_seg diffloss configs/finetune_joint_mfcadpp_diffloss_200.yaml 256
  run_task 2 tmcad_cls diffloss configs/finetune_joint_tmcad_diffloss_200.yaml 256
}
worker_gpu3() { run_task 3 cadsynth_seg mlp configs/finetune_cadsynth.yaml 512; }
worker_gpu5() { run_task 5 cadsynth_seg diffloss configs/finetune_cadsynth_diffloss.yaml 512; }
worker_gpu6() { run_task 6 mfinstseg_seg mlp configs/finetune_mfinstseg.yaml 512; }
worker_gpu7() { run_task 7 mfinstseg_seg diffloss configs/finetune_mfinstseg_diffloss.yaml 512; }

declare -a worker_pids=()
status=0
for gpu in 0 1 2 3 5 6 7; do
  "worker_gpu${gpu}" >"$LOG_ROOT/gpu${gpu}.log" 2>&1 &
  worker_pids+=("$!")
done
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    status=1
  fi
done

# The original GPU-4 launcher was paused before this scheduler started. Once
# these queues are done, resume it so it skips successful runs, retries any
# missing ones, and generates the normal comparison report.
if [[ -f "$ORIGINAL_LAUNCHER_PID_FILE" ]]; then
  original_pid="$(cat "$ORIGINAL_LAUNCHER_PID_FILE")"
  if kill -0 "$original_pid" 2>/dev/null; then
    kill -CONT "$original_pid"
    echo "[$(date --iso-8601=seconds)] resumed original launcher pid=$original_pid"
  fi
fi

if (( status != 0 )); then
  echo "At least one seven-GPU worker failed; original launcher resumed for fallback" >&2
  exit "$status"
fi
echo "[$(date --iso-8601=seconds)] all seven-GPU queues complete"
