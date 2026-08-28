#!/usr/bin/env bash
# Compare the two latest joint encoders with/without discrete pretraining losses
# on the five standard downstream benchmarks. Each dataset's pair runs on the
# same physical GPU, and every run trains for 200 epochs before test evaluation.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/discrete_loss_encoder_ablation}"
LOG_ROOT="${LOG_ROOT:-$RUN_ROOT/launch_logs/$RUN_TAG}"

DISCRETE_ON_CHECKPOINT="${DISCRETE_ON_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260813-183150_joint_fusion_gallery_mlp_all_splits/checkpoints/last.pt}"
DISCRETE_OFF_CHECKPOINT="${DISCRETE_OFF_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260814-160525_joint_fusion_gallery_mlp_all_splits/checkpoints/last.pt}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
for checkpoint in "$DISCRETE_ON_CHECKPOINT" "$DISCRETE_OFF_CHECKPOINT"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Pretraining checkpoint not found: $checkpoint" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 60 seconds."
  sleep 60
done

run_experiment() {
  local dataset="$1"
  local gpu="$2"
  local config="$3"
  local variant="$4"
  local checkpoint="$5"
  local run_name="discrete_loss_${variant}_${dataset}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${dataset}_${variant}.log"

  echo "[$(date --iso-8601=seconds)] Starting dataset=$dataset variant=$variant gpu=$gpu"
  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.graph_pooling=mean_max" \
    --override "train.pretrain_checkpoint=$checkpoint" \
    --override "train.epochs=200" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    --override "train.dataloader_seed=42" \
    >"$task_log" 2>&1; then
    echo "[$(date --iso-8601=seconds)] FAILED training dataset=$dataset variant=$variant; log=$task_log" >&2
    return 1
  fi

  local run_dir
  run_dir="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${run_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local best_checkpoint="$run_dir/checkpoints/best.pt"
  if [[ -z "$run_dir" || ! -f "$best_checkpoint" ]]; then
    echo "Best checkpoint not found for dataset=$dataset variant=$variant: $best_checkpoint" >&2
    return 1
  fi

  if ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$best_checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1; then
    echo "[$(date --iso-8601=seconds)] FAILED evaluation dataset=$dataset variant=$variant; log=$task_log" >&2
    return 1
  fi
  echo "[$(date --iso-8601=seconds)] Completed dataset=$dataset variant=$variant run=$run_dir"
}

run_pair() {
  local dataset="$1"
  local gpu="$2"
  local config="$3"
  local status=0

  run_experiment "$dataset" "$gpu" "$config" discrete_on "$DISCRETE_ON_CHECKPOINT" || status=1
  run_experiment "$dataset" "$gpu" "$config" discrete_off "$DISCRETE_OFF_CHECKPOINT" || status=1
  return "$status"
}

run_gpu0_pairs() {
  local status=0
  run_pair blendit_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_blendit_mlp.yaml" || status=1
  run_pair fabwave_cls "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml" || status=1
  return "$status"
}

# Keep each comparison pair on one GPU. GPU 0 handles the two shorter datasets
# sequentially while the other three GPUs each handle one dataset pair.
pids=()
run_gpu0_pairs & pids+=("$!")
run_pair fusion360seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_mlp.yaml" & pids+=("$!")
run_pair mfcadpp_seg "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" & pids+=("$!")
run_pair tmcad_cls "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] One or more comparisons failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi
echo "[$(date --iso-8601=seconds)] All ten downstream comparisons completed: $RUN_ROOT"
