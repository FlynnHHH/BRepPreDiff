#!/usr/bin/env bash
# Run all five Edge Update Attention benchmarks with the joint
# L_x_start + 0.5 * L_epsilon DiffLoss objective.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/new_occ_features_downstreams}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/edge_update_xstart_epsilon_$RUN_TAG}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260818-173143_new_occ_features_edge_update_resume_e095_20260818-1729/checkpoints/last.pt}"
TASK_GROUP="${TASK_GROUP:-all}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 60 seconds."
  sleep 60
done

run_downstream() {
  local task_name="$1"
  local gpu="$2"
  local config="$3"
  local epochs="$4"
  local run_name="edge_update_${task_name}_diffloss_xse_${RUN_TAG}"
  local task_log="$LOG_ROOT/${task_name}.log"

  echo "[$(date --iso-8601=seconds)] Starting $task_name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.finetune_head=diffusion" \
    --override "brep.edge_u_grid_size=10" \
    --override "label_diffusion.prediction_type=x_start_epsilon" \
    --override "label_diffusion.x_start_loss_weight=1.0" \
    --override "label_diffusion.epsilon_loss_weight=0.5" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.epochs=$epochs" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    --override "wandb.enabled=false" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $task_name: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed $task_name"
}

if [[ "$TASK_GROUP" != "all" && "$TASK_GROUP" != "gpu0" && "$TASK_GROUP" != "fabwave" ]]; then
  echo "TASK_GROUP must be 'all', 'gpu0', or 'fabwave'; got: $TASK_GROUP" >&2
  exit 2
fi

pids=()
if [[ "$TASK_GROUP" == "all" || "$TASK_GROUP" == "gpu0" ]]; then
  (
    run_downstream blendit_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_blendit_diffloss_200.yaml" 200
    run_downstream fabwave_cls "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_diffloss_acc_200.yaml" 200
  ) & pids+=("$!")
else
  run_downstream fabwave_cls "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_diffloss_acc_200.yaml" 200 & pids+=("$!")
fi

if [[ "$TASK_GROUP" == "all" ]]; then
  run_downstream fusion360seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_diffloss_200.yaml" 200 & pids+=("$!")
  run_downstream mfcadpp_seg "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_diffloss_200.yaml" 200 & pids+=("$!")
  run_downstream tmcad_cls "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_diffloss_200.yaml" 200 & pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] Five x_start+0.5*epsilon DiffLoss experiments completed: $RUN_ROOT"
