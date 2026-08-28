#!/usr/bin/env bash
# Run MLP controls for the four non-FabWave downstream tasks. GPU 0/1 queues
# wait for the currently running FabWave DiffLoss/MLP jobs, while GPU 2/3 start
# immediately. Generate a five-task comparison after every test evaluation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/python}"
RUN_TAG="${RUN_TAG:-new_occ_all_mlp_200_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/new_occ_features_downstreams}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260818-173143_new_occ_features_edge_update_resume_e095_20260818-1729/checkpoints/last.pt}"
LOG_DIR="$ROOT_DIR/runs/launch_logs/all_downstream_mlp_$RUN_TAG"
FAB_DIFF_TAG="${FAB_DIFF_TAG:-new_occ_fabwave_diffloss_200_20260819-103615}"
FAB_MLP_TAG="${FAB_MLP_TAG:-new_occ_fabwave_mlp_200_20260819-1040}"
FAB_DIFF_LOG="$ROOT_DIR/runs/launch_logs/edge_update_xstart_epsilon_$FAB_DIFF_TAG/fabwave_cls.log"
FAB_MLP_LOG="$ROOT_DIR/runs/launch_logs/fabwave_mlp_compare_$FAB_MLP_TAG/fabwave_mlp.log"
REPORT_PATH="$ROOT_DIR/reports/new_occ_mlp_vs_diffloss_all_downstreams_2026-08-19.md"

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$LOG_DIR/launcher.pid"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

latest_run() {
  local pattern="$1"
  find "$RUN_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

wait_for_test_metrics() {
  local pattern="$1" failure_log="$2" run_dir
  while true; do
    run_dir="$(latest_run "$pattern")"
    [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]] && return 0
    if [[ -f "$failure_log" ]] && grep -q 'Traceback' "$failure_log"; then
      echo "Upstream task failed: $failure_log" >&2
      return 1
    fi
    sleep 30
  done
}

run_mlp() {
  local task_name="$1" gpu="$2" config="$3"
  local run_name="edge_update_${task_name}_mlp_$RUN_TAG"
  local task_log="$LOG_DIR/${task_name}.log"

  echo "[$(date --iso-8601=seconds)] starting $task_name MLP on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.finetune \
    --config "$config" \
    --override "run.name=$run_name" \
    --override "run.output_dir=$RUN_ROOT" \
    --override "model.encoder_type=edge_update_attention" \
    --override "model.num_heads=4" \
    --override "model.finetune_head=mlp" \
    --override "brep.edge_u_grid_size=10" \
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT" \
    --override "train.resume=null" \
    --override "train.epochs=200" \
    --override "train.batch_size=64" \
    --override "train.gradient_accumulation_steps=4" \
    --override "wandb.enabled=false" \
    >"$task_log" 2>&1

  local run_dir
  run_dir="$(latest_run "*_${run_name}")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m blendit.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test --output "$run_dir/test_metrics.json" \
    --batch-size 64 --num-workers 8 --device cuda \
    >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed $task_name MLP"
}

pids=()
run_mlp mfcadpp_seg 2 "$ROOT_DIR/configs/finetune_joint_mfcadpp_mlp.yaml" & pids+=("$!")
run_mlp fusion360seg 3 "$ROOT_DIR/configs/finetune_joint_fusion360seg_mlp.yaml" & pids+=("$!")
(
  wait_for_test_metrics "*_edge_update_fabwave_cls_diffloss_xse_$FAB_DIFF_TAG" "$FAB_DIFF_LOG"
  run_mlp blendit_seg 0 "$ROOT_DIR/configs/finetune_joint_blendit_mlp.yaml"
) & pids+=("$!")
(
  wait_for_test_metrics "*_edge_update_fabwave_cls_mlp_$FAB_MLP_TAG" "$FAB_MLP_LOG"
  run_mlp tmcad_cls 1 "$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml"
) & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one MLP control failed; inspect $LOG_DIR" >&2
  exit "$status"
fi

FAB_MLP_RUN="$(latest_run "*_edge_update_fabwave_cls_mlp_$FAB_MLP_TAG")"
FAB_DIFF_RUN="$(latest_run "*_edge_update_fabwave_cls_diffloss_xse_$FAB_DIFF_TAG")"
BLENDIT_MLP_RUN="$(latest_run "*_edge_update_blendit_seg_mlp_$RUN_TAG")"
FUSION_MLP_RUN="$(latest_run "*_edge_update_fusion360seg_mlp_$RUN_TAG")"
MFCAD_MLP_RUN="$(latest_run "*_edge_update_mfcadpp_seg_mlp_$RUN_TAG")"
TMCAD_MLP_RUN="$(latest_run "*_edge_update_tmcad_cls_mlp_$RUN_TAG")"

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_new_occ_all_heads.py" \
  --pair BlendIt "$BLENDIT_MLP_RUN" "$RUN_ROOT/finetune/20260818-211021_edge_update_blendit_seg_diffloss_xse_new_occ_diffloss_200_gpu0_20260818-1953" \
  --pair Fusion360Seg "$FUSION_MLP_RUN" "$RUN_ROOT/finetune/20260818-194812_edge_update_fusion360seg_diffloss_xse_new_occ_diffloss_200_20260818-195000" \
  --pair MFCAD++ "$MFCAD_MLP_RUN" "$RUN_ROOT/finetune/20260818-194812_edge_update_mfcadpp_seg_diffloss_xse_new_occ_diffloss_200_20260818-195000" \
  --pair TMCAD "$TMCAD_MLP_RUN" "$RUN_ROOT/finetune/20260818-194812_edge_update_tmcad_cls_diffloss_xse_new_occ_diffloss_200_20260818-195000" \
  --pair FabWave "$FAB_MLP_RUN" "$FAB_DIFF_RUN" \
  --output "$REPORT_PATH" >"$LOG_DIR/comparison.log" 2>&1
echo "[$(date --iso-8601=seconds)] five-task comparison complete: $REPORT_PATH"

