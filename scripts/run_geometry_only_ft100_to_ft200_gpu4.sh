#!/usr/bin/env bash
# Continue all 14 geometry-only downstream runs from epoch 100 to epoch 200,
# evaluate the global validation-best checkpoints, and compare against the
# matched full-loss encoder report.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"
SOURCE_TAG="geometry_only_inductive9_pre100_ft100_seed42_20260901"
RUN_TAG="${RUN_TAG:-geometry_only_inductive9_pre100_ft200_seed42_20260902}"
SOURCE_ROOT="$ROOT_DIR/runs/geometry_only_inductive9"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/geometry_only_inductive9_ft200}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/$RUN_TAG}"
HEAD_REPORT="${HEAD_REPORT:-$ROOT_DIR/reports/${RUN_TAG}.md}"
BASELINE_REPORT="${BASELINE_REPORT:-$ROOT_DIR/reports/inductive9_lr1e4_pre100_full_mlp_diffloss200_seed42_20260901.md}"
COMPARISON_REPORT="${COMPARISON_REPORT:-$ROOT_DIR/reports/geometry_only_vs_inductive9_full_loss_pre100_ft200_seed42_20260902.md}"
PRETRAIN_CHECKPOINT="$ROOT_DIR/runs/geometry_only_inductive9/pretrain/20260901-201915_geometry_only_inductive9_no_fab_lr1e4_e100_seed42_geometry_only_inductive9_pre100_ft100_seed42_20260901/checkpoints/last.pt"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
printf '%s\n' "$RUN_TAG" >"$LOG_ROOT/run_tag"
printf '%s\n' "$HEAD_REPORT" >"$LOG_ROOT/head_report_path"
printf '%s\n' "$COMPARISON_REPORT" >"$LOG_ROOT/comparison_report_path"

if [[ ! -f "$PRETRAIN_CHECKPOINT" ]]; then
  echo "Pretrain checkpoint not found: $PRETRAIN_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$BASELINE_REPORT" ]]; then
  echo "Baseline report not found: $BASELINE_REPORT" >&2
  exit 2
fi
while ! timeout 30s nvidia-smi -i "$GPU_ID" >/dev/null 2>&1; do
  echo "[$(date --iso-8601=seconds)] GPU $GPU_ID unavailable; retrying in 60 seconds"
  sleep 60
done

tasks=(
  brepprediff_seg fusion360seg mfcadpp_seg tmcad_cls solidletters_cls
  cadsynth_seg mfinstseg_seg
)
mlp_configs=(
  configs/finetune_joint_brepprediff_mlp.yaml
  configs/finetune_joint_fusion360seg_mlp.yaml
  configs/finetune_joint_mfcadpp_mlp.yaml
  configs/finetune_joint_tmcad_mlp.yaml
  configs/finetune_solidletters_mlp.yaml
  configs/finetune_cadsynth.yaml
  configs/finetune_mfinstseg.yaml
)
diff_configs=(
  configs/finetune_joint_brepprediff_diffloss_200.yaml
  configs/finetune_joint_fusion360seg_diffloss_200.yaml
  configs/finetune_joint_mfcadpp_diffloss_200.yaml
  configs/finetune_joint_tmcad_diffloss_200.yaml
  configs/finetune_solidletters_diffloss.yaml
  configs/finetune_cadsynth_diffloss.yaml
  configs/finetune_mfinstseg_diffloss.yaml
)
batches=(256 256 256 256 256 512 512)
display_names=(BRepPreDiff Fusion360Seg MFCAD++ TMCAD SolidLetters CADSynth MFInstSeg)

latest_run() {
  local root="$1" pattern="$2"
  if [[ ! -d "$root/finetune" ]]; then
    return 0
  fi
  find "$root/finetune" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

source_run() {
  local task="$1" head="$2" candidate selected=""
  while IFS= read -r candidate; do
    if [[ -f "$candidate/test_metrics.json" && -f "$candidate/checkpoints/last.pt" ]]; then
      selected="$candidate"
    fi
  done < <(
    find "$SOURCE_ROOT/finetune" -mindepth 1 -maxdepth 1 -type d \
      -name "*_geometry_only_${task}_${head}_ft100_seed42_$SOURCE_TAG" \
      -printf '%T@ %p\n' | sort -n | cut -d' ' -f2-
  )
  printf '%s\n' "$selected"
}

resume_checkpoint_for() {
  local run_dir="$1"
  if [[ -f "$run_dir/checkpoints/last.pt" ]]; then
    printf '%s\n' "$run_dir/checkpoints/last.pt"
    return
  fi
  find "$run_dir/checkpoints" -maxdepth 1 -type f -name 'epoch_*.pt' -print 2>/dev/null \
    | sort -V | tail -n 1
}

run_downstream() {
  local index="$1" head="$2"
  local task="${tasks[$index]}" config finetune_head batch source_dir run_name run_dir
  local resume_checkpoint checkpoint task_log
  local split_args=() select_args=()
  task_log="$LOG_ROOT/${task}_${head}.log"
  if [[ "$head" == "mlp" ]]; then
    config="${mlp_configs[$index]}"
    finetune_head="mlp"
  else
    config="${diff_configs[$index]}"
    finetune_head="diffusion"
  fi
  batch="${batches[$index]}"
  source_dir="$(source_run "$task" "$head")"
  if [[ -z "$source_dir" ]]; then
    echo "Completed epoch-100 source run not found task=$task head=$head" >&2
    return 1
  fi

  run_name="geometry_only_${task}_${head}_ft200_seed42_$RUN_TAG"
  run_dir="$(latest_run "$RUN_ROOT" "*_${run_name}")"
  if [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]]; then
    echo "[$(date --iso-8601=seconds)] skipping completed task=$task head=$head run=$run_dir"
    return
  fi

  resume_checkpoint="$source_dir/checkpoints/last.pt"
  if [[ -n "$run_dir" ]]; then
    checkpoint="$(resume_checkpoint_for "$run_dir")"
    if [[ -n "$checkpoint" ]]; then
      resume_checkpoint="$checkpoint"
    fi
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
    --override "train.resume=$resume_checkpoint"
    --override "train.epochs=200"
    --override "train.batch_size=$batch"
    --override "train.gradient_accumulation_steps=1"
    --override "train.num_workers=16"
    --override "train.dataloader_seed=42"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=false"
    "${split_args[@]}"
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

  echo "[$(date --iso-8601=seconds)] continuing task=$task head=$head checkpoint=$resume_checkpoint target_epoch=200 gpu=$GPU_ID"
  GPU_ID="$GPU_ID" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" \
    >"$task_log" 2>&1
  run_dir="$(latest_run "$RUN_ROOT" "*_${run_name}")"
  if [[ -z "$run_dir" || ! -f "$run_dir/checkpoints/last.pt" ]]; then
    echo "Completed continuation run not found task=$task head=$head" >&2
    return 1
  fi

  select_args=(--candidate "$source_dir/checkpoints/best.pt")
  while IFS= read -r checkpoint; do
    select_args+=(--candidate "$checkpoint")
  done < <(
    find "$RUN_ROOT/finetune" -mindepth 3 -maxdepth 3 -type f \
      -path "*/checkpoints/best.pt" \
      -path "*_${run_name}/checkpoints/best.pt" -print | sort
  )
  conda run --no-capture-output -n blendit python scripts/select_best_validation_checkpoint.py \
    "${select_args[@]}" --metric acc \
    --output "$run_dir/checkpoints/best.pt" \
    --metadata "$run_dir/best_selection.json" >>"$task_log" 2>&1

  CUDA_VISIBLE_DEVICES="$GPU_ID" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$run_dir/checkpoints/best.pt" --split test \
    --output "$run_dir/test_metrics.json" --batch-size "$batch" \
    --num-workers 16 --device cuda >>"$task_log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed task=$task head=$head run=$run_dir"
}

# Three queues cap concurrency at three jobs on GPU 4.
queue_zero() {
  run_downstream 0 mlp
  run_downstream 1 diffloss
  run_downstream 3 mlp
  run_downstream 4 diffloss
  run_downstream 6 mlp
}
queue_one() {
  run_downstream 0 diffloss
  run_downstream 2 mlp
  run_downstream 3 diffloss
  run_downstream 5 mlp
  run_downstream 6 diffloss
}
queue_two() {
  run_downstream 1 mlp
  run_downstream 2 diffloss
  run_downstream 4 mlp
  run_downstream 5 diffloss
}

pids=()
queue_zero & pids+=("$!")
queue_one & pids+=("$!")
queue_two & pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one continuation queue failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

head_args=()
comparison_args=()
for index in "${!tasks[@]}"; do
  mlp_run="$(latest_run "$RUN_ROOT" "*_geometry_only_${tasks[$index]}_mlp_ft200_seed42_$RUN_TAG")"
  diff_run="$(latest_run "$RUN_ROOT" "*_geometry_only_${tasks[$index]}_diffloss_ft200_seed42_$RUN_TAG")"
  if [[ ! -f "$mlp_run/test_metrics.json" || ! -f "$diff_run/test_metrics.json" ]]; then
    echo "Missing test metrics for ${tasks[$index]}" >&2
    exit 1
  fi
  head_args+=(--pair "${display_names[$index]}" "$mlp_run" "$diff_run")
  comparison_args+=(--run "${display_names[$index]}" MLP "$mlp_run")
  comparison_args+=(--run "${display_names[$index]}" DiffLoss "$diff_run")
done

conda run --no-capture-output -n blendit python scripts/compare_new_occ_all_heads.py \
  "${head_args[@]}" \
  --title "Geometry-only inductive 9-source encoder：200-epoch MLP 与 DiffLoss 对照" \
  --epochs 200 \
  --batch-description "task batch size 256/512、梯度累积 1、GPU 4 三路并发" \
  --pretrain-description "同一 100-epoch geometry-only inductive 9-source（无 FabWave）encoder" \
  --output "$HEAD_REPORT" >"$LOG_ROOT/head_report.log" 2>&1

conda run --no-capture-output -n blendit python scripts/compare_encoder_reports.py \
  --baseline-report "$BASELINE_REPORT" \
  "${comparison_args[@]}" \
  --title "Geometry-only 与 full-loss inductive 9-source encoder：200-epoch 下游对照" \
  --candidate-name "Geometry-only encoder（categorical/relation loss weight = 0）" \
  --baseline-name "Full-loss encoder（lr=1e-4）" \
  --output "$COMPARISON_REPORT" >"$LOG_ROOT/comparison_report.log" 2>&1

echo "[$(date --iso-8601=seconds)] pipeline complete head_report=$HEAD_REPORT comparison_report=$COMPARISON_REPORT"
