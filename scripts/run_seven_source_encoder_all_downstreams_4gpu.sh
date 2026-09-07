#!/usr/bin/env bash
# Wait for the 711/63 seven-source encoder, then run matched MLP and DiffLoss
# fine-tuning/evaluation on the four retained downstream benchmarks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PRETRAIN_RUN="${PRETRAIN_RUN:-$ROOT_DIR/runs/pretrain/20260820-163406_new_occ_seven_source_edge_update_20260820-151913}"
PRETRAIN_CHECKPOINT="$PRETRAIN_RUN/checkpoints/last.pt"
PRETRAIN_LOG="$PRETRAIN_RUN/logs/pretrain.log"
RUN_TAG="${RUN_TAG:-seven_source_711_all_downstreams_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/seven_source_711_downstreams}"
LOG_ROOT="$ROOT_DIR/runs/launch_logs/$RUN_TAG"
REPORT_PATH="$ROOT_DIR/reports/${RUN_TAG}.md"

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
exec > >(tee -a "$LOG_ROOT/launcher.log") 2>&1
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "[$(date --iso-8601=seconds)] waiting for completed pretrain run=$PRETRAIN_RUN"
while [[ ! -f "$PRETRAIN_CHECKPOINT" ]] || ! grep -q 'finished pretraining' "$PRETRAIN_LOG" 2>/dev/null; do
  if grep -Eq 'Traceback|ChildFailedError|OutOfMemoryError' "$PRETRAIN_LOG" 2>/dev/null; then
    echo "Pretraining failed; inspect $PRETRAIN_LOG" >&2
    exit 1
  fi
  sleep 60
done
echo "[$(date --iso-8601=seconds)] pretrain complete; checkpoint=$PRETRAIN_CHECKPOINT"

while ! timeout 30s "$PYTHON_BIN" -c \
  'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 4 else 1)' \
  >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] CUDA driver unavailable; retrying in 60 seconds"
  sleep 60
done

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_downstream() {
  local dataset="$1" head="$2" gpu="$3" config="$4"
  local run_name="seven_source_711_${dataset}_${head}_200_$RUN_TAG"
  local task_log="$LOG_ROOT/${dataset}_${head}.log"
  local finetune_head="mlp"

  if [[ "$head" == "diffloss" ]]; then
    finetune_head="diffusion"
  fi

  echo "[$(date --iso-8601=seconds)] starting dataset=$dataset head=$head gpu=$gpu"
  args=(
    --config "$config"
    --override "run.name=$run_name"
    --override "run.output_dir=$RUN_ROOT"
    --override "seed=42"
    --override "model.encoder_type=edge_update_attention"
    --override "model.num_heads=4"
    --override "model.finetune_head=$finetune_head"
    --override "model.graph_pooling=mean_max"
    --override "brep.edge_u_grid_size=10"
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
    --override "train.resume=null"
    --override "train.epochs=200"
    --override "train.batch_size=64"
    --override "train.gradient_accumulation_steps=4"
    --override "train.dataloader_seed=42"
    --override "wandb.enabled=false"
  )
  if [[ "$head" == "diffloss" ]]; then
    args+=(
      --override "label_diffusion.prediction_type=x_start_epsilon"
      --override "label_diffusion.x_start_loss_weight=1.0"
      --override "label_diffusion.epsilon_loss_weight=0.5"
      --override "label_diffusion.seed=42"
    )
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune "${args[@]}" \
    >"$task_log" 2>&1

  local run_dir checkpoint
  run_dir="$(latest_run "*_${run_name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint not found dataset=$dataset head=$head path=$checkpoint" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" \
    --split test \
    --output "$run_dir/test_metrics.json" \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed dataset=$dataset head=$head run=$run_dir"
}

# One queue per GPU avoids oversubscription.
pids=()
(
  run_downstream brepprediff_seg mlp "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_mlp.yaml"
  run_downstream brepprediff_seg diffloss "${GPUS[0]}" "$ROOT_DIR/configs/finetune_joint_brepprediff_diffloss_200.yaml"
) & pids+=("$!")
(
  run_downstream fusion360seg mlp "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_mlp.yaml"
  run_downstream fusion360seg diffloss "${GPUS[1]}" "$ROOT_DIR/configs/finetune_joint_fusion360seg_diffloss_200.yaml"
) & pids+=("$!")
(
  run_downstream mfcadpp_seg mlp "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml"
  run_downstream mfcadpp_seg diffloss "${GPUS[2]}" "$ROOT_DIR/configs/finetune_joint_mfcadpp_diffloss_200.yaml"
) & pids+=("$!")
(
  run_downstream tmcad_cls mlp "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml"
  run_downstream tmcad_cls diffloss "${GPUS[3]}" "$ROOT_DIR/configs/finetune_joint_tmcad_diffloss_200.yaml"
) & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one downstream queue failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

brepprediff_mlp="$(latest_run "*_seven_source_711_brepprediff_seg_mlp_200_$RUN_TAG")"
brepprediff_diff="$(latest_run "*_seven_source_711_brepprediff_seg_diffloss_200_$RUN_TAG")"
fusion_mlp="$(latest_run "*_seven_source_711_fusion360seg_mlp_200_$RUN_TAG")"
fusion_diff="$(latest_run "*_seven_source_711_fusion360seg_diffloss_200_$RUN_TAG")"
mfcad_mlp="$(latest_run "*_seven_source_711_mfcadpp_seg_mlp_200_$RUN_TAG")"
mfcad_diff="$(latest_run "*_seven_source_711_mfcadpp_seg_diffloss_200_$RUN_TAG")"
tmcad_mlp="$(latest_run "*_seven_source_711_tmcad_cls_mlp_200_$RUN_TAG")"
tmcad_diff="$(latest_run "*_seven_source_711_tmcad_cls_diffloss_200_$RUN_TAG")"
"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_new_occ_all_heads.py" \
  --pair BRepPreDiff "$brepprediff_mlp" "$brepprediff_diff" \
  --pair Fusion360Seg "$fusion_mlp" "$fusion_diff" \
  --pair MFCAD++ "$mfcad_mlp" "$mfcad_diff" \
  --pair TMCAD "$tmcad_mlp" "$tmcad_diff" \
  --output "$REPORT_PATH" >"$LOG_ROOT/comparison.log" 2>&1

echo "[$(date --iso-8601=seconds)] all downstream experiments completed report=$REPORT_PATH"
