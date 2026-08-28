#!/usr/bin/env bash
# Exact-epoch ablation for the 2026-08-24 BS32/LR5e-4 encoder.
#
# Four 200-epoch trajectories per task are sufficient: fine-tuning has no
# total-epoch-dependent scheduler, and checkpoints are saved every 10 epochs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PRETRAIN_RUN="${PRETRAIN_RUN:-$ROOT_DIR/runs/pretrain/20260824-114039_seven_source_711_4gpu_wandb_20260824-113934}"
RUN_TAG="${RUN_TAG:-pretrain_finetune_epoch_grid_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/pretrain_finetune_epoch_grid}"
LOG_ROOT="$ROOT_DIR/runs/launch_logs/$RUN_TAG"
RESULT_ROOT="$RUN_ROOT/results/$RUN_TAG"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"
TASKS="${TASKS:-brepprediff_seg,fusion360seg,mfcadpp_seg,tmcad_cls,fabwave_cls}"

PRETRAIN_EPOCHS=(20 50 100 150)
FINETUNE_EPOCHS=(10 20 50 100 150 200)

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

for epoch in "${PRETRAIN_EPOCHS[@]}"; do
  checkpoint="$PRETRAIN_RUN/checkpoints/epoch_$(printf '%04d' "$epoch").pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Pretrain checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
exec > >(tee -a "$LOG_ROOT/launcher.log") 2>&1
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

task_enabled() {
  local wanted="$1"
  [[ ",$TASKS," == *",$wanted,"* ]]
}

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_task_pretrain() {
  local task="$1" gpu="$2" config="$3" pretrain_epoch="$4"
  local batch_size="$5" accumulation_steps="$6" num_workers="$7"
  local checkpoint run_name task_log run_dir finetune_epoch eval_checkpoint output

  if ! task_enabled "$task"; then
    echo "[$(date --iso-8601=seconds)] skipping task=$task pretrain_epoch=$pretrain_epoch"
    return
  fi

  checkpoint="$PRETRAIN_RUN/checkpoints/epoch_$(printf '%04d' "$pretrain_epoch").pt"
  run_name="epoch_grid_${task}_mlp_pre${pretrain_epoch}_ft200_${RUN_TAG}"
  task_log="$LOG_ROOT/${task}_pre${pretrain_epoch}.log"
  echo "[$(date --iso-8601=seconds)] training task=$task pretrain_epoch=$pretrain_epoch gpu=$gpu batch=$batch_size accumulation=$accumulation_steps"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "run.save_every_epochs=10" \
    --override "seed=42" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.finetune_head=mlp" \
    --override "model.graph_pooling=mean_max" \
    --override "brep.edge_u_grid_size=10" \
    --override "train.pretrain_checkpoint=$checkpoint" \
    --override "train.resume=null" \
    --override "train.epochs=200" \
    --override "train.batch_size=$batch_size" \
    --override "train.gradient_accumulation_steps=$accumulation_steps" \
    --override "train.num_workers=$num_workers" \
    --override "train.dataloader_seed=42" \
    --override "wandb.enabled=false" \
    >"$task_log" 2>&1

  run_dir="$(latest_run "*_${run_name}")"
  if [[ -z "$run_dir" ]]; then
    echo "Fine-tune run directory not found for run.name=$run_name" >&2
    return 1
  fi

  for finetune_epoch in "${FINETUNE_EPOCHS[@]}"; do
    eval_checkpoint="$run_dir/checkpoints/epoch_$(printf '%04d' "$finetune_epoch").pt"
    output="$RESULT_ROOT/$task/pretrain_${pretrain_epoch}/finetune_${finetune_epoch}.json"
    if [[ ! -f "$eval_checkpoint" ]]; then
      echo "Fine-tune checkpoint not found: $eval_checkpoint" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
      --checkpoint "$eval_checkpoint" \
      --split test \
      --output "$output" \
      --batch-size "$batch_size" \
      --num-workers "$num_workers" \
      --device cuda \
      >>"$task_log" 2>&1
    echo "[$(date --iso-8601=seconds)] evaluated task=$task pretrain=$pretrain_epoch finetune=$finetune_epoch"
  done
}

# Two workers share each GPU. TMCAD uses a smaller micro-batch because a batch
# of 64 can reserve essentially all 24 GiB on a TITAN RTX. Accumulation keeps
# the effective batch at 256 for every task: 16x16 for TMCAD and 64x4 for the
# lighter tasks. Four data-loader workers per process cap the eight-worker
# launch at 32 workers on the 56-core host.
pids=()

for index in "${!PRETRAIN_EPOCHS[@]}"; do
  run_task_pretrain \
    tmcad_cls "${GPUS[$index]}" "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml" \
    "${PRETRAIN_EPOCHS[$index]}" 16 16 4 &
  pids+=("$!")
done

(
  run_task_pretrain mfcadpp_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" 20 64 4 4
  run_task_pretrain mfcadpp_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" 100 64 4 4
  run_task_pretrain brepprediff_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_mlp.yaml" 20 64 4 4
) & pids+=("$!")
(
  run_task_pretrain mfcadpp_seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" 50 64 4 4
  run_task_pretrain mfcadpp_seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" 150 64 4 4
  run_task_pretrain brepprediff_seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_mlp.yaml" 50 64 4 4
) & pids+=("$!")
(
  for epoch in "${PRETRAIN_EPOCHS[@]}"; do
    run_task_pretrain fusion360seg "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_mlp.yaml" "$epoch" 64 4 4
  done
) & pids+=("$!")
(
  for epoch in "${PRETRAIN_EPOCHS[@]}"; do
    run_task_pretrain fabwave_cls "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml" "$epoch" 64 4 4
  done
  run_task_pretrain brepprediff_seg "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_mlp.yaml" 100 64 4 4
  run_task_pretrain brepprediff_seg "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_mlp.yaml" 150 64 4 4
) & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one task queue failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/summarize_pretrain_finetune_epoch_grid.py" \
  --results-root "$RESULT_ROOT" \
  --csv "$RESULT_ROOT/results.csv" \
  --markdown "$REPORT_PATH"

echo "[$(date --iso-8601=seconds)] epoch grid completed report=$REPORT_PATH csv=$RESULT_ROOT/results.csv"
