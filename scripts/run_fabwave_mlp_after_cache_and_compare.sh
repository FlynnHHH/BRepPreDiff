#!/usr/bin/env bash
# Wait for the new FabWave cache/DiffLoss run, train a matched MLP baseline on
# GPU 1, then compare both best checkpoints on the test split.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
GPU_ID="${GPU_ID:-1}"
RUN_TAG="${RUN_TAG:-new_occ_fabwave_mlp_200_$(date +%Y%m%d-%H%M%S)}"
DIFF_RUN_TAG="${DIFF_RUN_TAG:-new_occ_fabwave_diffloss_200_20260819-103615}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/new_occ_features_downstreams}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260818-173143_new_occ_features_edge_update_resume_e095_20260818-1729/checkpoints/last.pt}"
LOG_DIR="$ROOT_DIR/runs/launch_logs/fabwave_mlp_compare_$RUN_TAG"
DIFF_LOG="$ROOT_DIR/runs/launch_logs/edge_update_xstart_epsilon_$DIFF_RUN_TAG/fabwave_cls.log"
REPORT_PATH="$ROOT_DIR/reports/fabwave_new_occ_mlp_vs_diffloss_2026-08-19.md"

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$LOG_DIR/launcher.pid"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "[$(date --iso-8601=seconds)] waiting for rebuilt cache and DiffLoss epoch 1"
until [[ -f "$DIFF_LOG" ]] && grep -q 'epoch=1/200 split=train start' "$DIFF_LOG"; do
  if [[ -f "$DIFF_LOG" ]] && grep -q 'Traceback' "$DIFF_LOG"; then
    echo "DiffLoss failed before MLP could start: $DIFF_LOG" >&2
    exit 1
  fi
  sleep 30
done

MLP_NAME="edge_update_fabwave_cls_mlp_$RUN_TAG"
MLP_LOG="$LOG_DIR/fabwave_mlp.log"
echo "[$(date --iso-8601=seconds)] starting matched MLP run on GPU $GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m brepprediff.training.finetune \
  --config "$ROOT_DIR/configs/finetune_joint_fabwave_min10_mlp_acc_200.yaml" \
  --override "run.name=$MLP_NAME" \
  --override "run.output_dir=$RUN_ROOT" \
  --override "model.encoder_type=edge_update_attention" \
  --override "model.num_heads=4" \
  --override "model.finetune_head=mlp" \
  --override "model.graph_pooling=mean_max" \
  --override "brep.edge_u_grid_size=10" \
  --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
  --override "train.resume=null" \
  --override "train.epochs=200" \
  --override "train.batch_size=64" \
  --override "train.gradient_accumulation_steps=4" \
  --override "wandb.enabled=false" \
  >"$MLP_LOG" 2>&1

MLP_RUN="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
  -name "*_${MLP_NAME}" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m brepprediff.training.evaluate \
  --checkpoint "$MLP_RUN/checkpoints/best.pt" \
  --split test --output "$MLP_RUN/test_metrics.json" \
  --batch-size 64 --num-workers 8 --device cuda \
  >>"$MLP_LOG" 2>&1

echo "[$(date --iso-8601=seconds)] MLP complete; waiting for DiffLoss test metrics"
while true; do
  DIFF_RUN="$(find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "*_edge_update_fabwave_cls_diffloss_xse_$DIFF_RUN_TAG" \
    -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
  [[ -n "$DIFF_RUN" && -f "$DIFF_RUN/test_metrics.json" ]] && break
  if [[ -f "$DIFF_LOG" ]] && grep -q 'Traceback' "$DIFF_LOG"; then
    echo "DiffLoss failed before comparison: $DIFF_LOG" >&2
    exit 1
  fi
  sleep 30
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_fabwave_new_occ_heads.py" \
  --mlp-run "$MLP_RUN" --diffloss-run "$DIFF_RUN" --output "$REPORT_PATH" \
  >"$LOG_DIR/comparison.log" 2>&1
echo "[$(date --iso-8601=seconds)] comparison complete: $REPORT_PATH"

