#!/usr/bin/env bash
# Pretrain Edge Update Attention on the seven-source joint corpus, then fine-tune
# classification tasks with DiffLoss and segmentation tasks with an MLP head.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDGE_WORKTREE="${EDGE_WORKTREE:-/tmp/blendit-encoder-edge-update}"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/edge_update_new_joint}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/edge_update_new_joint_$RUN_TAG}"

if [[ ! -d "$EDGE_WORKTREE/src/blendit" ]]; then
  echo "Edge Update worktree not found: $EDGE_WORKTREE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$EDGE_WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# The host driver can temporarily be unavailable even though the training job is
# valid. Keep this launcher queued and begin only after the four GPUs respond.
while ! timeout 30s nvidia-smi -L >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 5 minutes."
  sleep 300
done

PRETRAIN_NAME="edge_update_new_joint_${RUN_TAG}"
echo "[$(date --iso-8601=seconds)] Starting Edge Update Attention pretraining: $PRETRAIN_NAME"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  -m blendit.training.pretrain \
  --config "$ROOT_DIR/configs/pretrain_joint_fusion_gallery_mlp_all_splits.yaml" \
  --data-config "$ROOT_DIR/data/pretrain_joint_fusion_gallery_filtered_fabwave_all_splits.yaml" \
  --override "run.name=$PRETRAIN_NAME" \
  --override "run.output_dir=$RUN_ROOT" \
  --override "model.encoder_type=edge_update_attention" \
  --override "model.num_heads=4" \
  --override "train.batch_size=32" \
  --override "train.gradient_accumulation_steps=8"

PRETRAIN_RUN="$(find "$RUN_ROOT/pretrain" -mindepth 1 -maxdepth 1 -type d \
  -name "*_${PRETRAIN_NAME}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
PRETRAIN_CHECKPOINT="$PRETRAIN_RUN/checkpoints/last.pt"
if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretraining checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 1
fi

run_downstream() {
  local task_name="$1"
  local gpu="$2"
  local config="$3"
  local fine_name="edge_update_${task_name}_${RUN_TAG}"
  local task_log="$LOG_ROOT/${task_name}.log"

  echo "[$(date --iso-8601=seconds)] Starting downstream task $task_name on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$fine_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    >"$task_log" 2>&1

  local fine_run
  fine_run="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_${fine_name}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  local checkpoint="$fine_run/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found for $task_name: $checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$fine_run/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] Completed downstream task $task_name"
}

# Segmentation uses MLP; classification uses DiffLoss.
pids=()
run_downstream blendit_seg "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_blendit_mlp.yaml" & pids+=("$!")
run_downstream fusion360seg "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_mlp.yaml" & pids+=("$!")
run_downstream mfcadpp_seg "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" & pids+=("$!")
run_downstream tmcad_cls "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_diffloss_200.yaml" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

# The fifth downstream task reuses GPU 0 after the first wave completes.
run_downstream fabwave_cls "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_fabwave_min10_diffloss_acc_200.yaml"

echo "[$(date --iso-8601=seconds)] Pipeline completed. Results: $RUN_ROOT"
