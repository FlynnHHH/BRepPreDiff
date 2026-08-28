#!/usr/bin/env bash
# Run matched TMCAD MLP/DiffLoss experiments for seeds 43 and 44 on four GPUs,
# evaluate best checkpoints, then combine them with the existing seed-42 runs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
RUN_TAG="${RUN_TAG:-new_occ_tmcad_seeds43_44_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/new_occ_features_downstreams}"
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$ROOT_DIR/runs/pretrain/20260818-173143_new_occ_features_edge_update_resume_e095_20260818-1729/checkpoints/last.pt}"
LOG_DIR="$ROOT_DIR/runs/launch_logs/tmcad_three_seed_$RUN_TAG"
REPORT_PATH="$ROOT_DIR/reports/tmcad_new_occ_mlp_vs_diffloss_3seeds_2026-08-19.md"

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

run_experiment() {
  local head="$1" seed="$2" gpu="$3" config run_name log
  if [[ "$head" == "mlp" ]]; then
    config="$ROOT_DIR/configs/finetune_joint_tmcad_mlp.yaml"
  else
    config="$ROOT_DIR/configs/finetune_joint_tmcad_diffloss_200.yaml"
  fi
  run_name="edge_update_tmcad_${head}_seed${seed}_$RUN_TAG"
  log="$LOG_DIR/${head}_seed${seed}.log"

  echo "[$(date --iso-8601=seconds)] starting head=$head seed=$seed gpu=$gpu"
  args=(
    --config "$config"
    --override "run.name=$run_name"
    --override "run.output_dir=$RUN_ROOT"
    --override "seed=$seed"
    --override "model.encoder_type=edge_update_attention"
    --override "model.num_heads=4"
    --override "model.finetune_head=$([[ "$head" == "mlp" ]] && echo mlp || echo diffusion)"
    --override "model.graph_pooling=mean_max"
    --override "brep.edge_u_grid_size=10"
    --override "train.pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
    --override "train.resume=null"
    --override "train.epochs=200"
    --override "train.batch_size=64"
    --override "train.gradient_accumulation_steps=4"
    --override "train.dataloader_seed=$seed"
    --override "wandb.enabled=false"
  )
  if [[ "$head" == "diffloss" ]]; then
    args+=(
      --override "label_diffusion.prediction_type=x_start_epsilon"
      --override "label_diffusion.x_start_loss_weight=1.0"
      --override "label_diffusion.epsilon_loss_weight=0.5"
      --override "label_diffusion.seed=$seed"
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.finetune "${args[@]}" >"$log" 2>&1

  local run_dir
  run_dir="$(latest_run "*_${run_name}")"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" \
    --split test --output "$run_dir/test_metrics.json" \
    --batch-size 64 --num-workers 8 --device cuda >>"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed head=$head seed=$seed"
}

pids=()
run_experiment mlp 43 0 & pids+=("$!")
run_experiment diffloss 43 1 & pids+=("$!")
run_experiment mlp 44 2 & pids+=("$!")
run_experiment diffloss 44 3 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one TMCAD seed experiment failed; inspect $LOG_DIR" >&2
  exit "$status"
fi

MLP43="$(latest_run "*_edge_update_tmcad_mlp_seed43_$RUN_TAG")"
DIFF43="$(latest_run "*_edge_update_tmcad_diffloss_seed43_$RUN_TAG")"
MLP44="$(latest_run "*_edge_update_tmcad_mlp_seed44_$RUN_TAG")"
DIFF44="$(latest_run "*_edge_update_tmcad_diffloss_seed44_$RUN_TAG")"
MLP42="$RUN_ROOT/finetune/20260819-111619_edge_update_tmcad_cls_mlp_new_occ_all_mlp_200_20260819-1100"
DIFF42="$RUN_ROOT/finetune/20260818-194812_edge_update_tmcad_cls_diffloss_xse_new_occ_diffloss_200_20260818-195000"

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_tmcad_new_occ_three_seeds.py" \
  --pair 42 "$MLP42" "$DIFF42" \
  --pair 43 "$MLP43" "$DIFF43" \
  --pair 44 "$MLP44" "$DIFF44" \
  --output "$REPORT_PATH" >"$LOG_DIR/comparison.log" 2>&1
echo "[$(date --iso-8601=seconds)] three-seed comparison complete: $REPORT_PATH"

