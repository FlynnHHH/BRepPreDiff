#!/usr/bin/env bash
# For each downstream dataset, pretrain an independent encoder from that
# dataset's train split only, then fine-tune/evaluate MLP and DiffLoss heads on
# the same dataset's train/val/test splits.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS_TEXT="${GPU_IDS:-4 5 6 7}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
RUN_TAG="${RUN_TAG:-single_dataset_pretrain_pre${PRETRAIN_EPOCHS}_ft${FINETUNE_EPOCHS}_seed42_$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/single_dataset_pretrain}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/runs/launch_logs/$RUN_TAG}"
REPORT_PATH="${REPORT_PATH:-$ROOT_DIR/reports/${RUN_TAG}.md}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"

read -r -a GPU_IDS_ARRAY <<<"$GPU_IDS_TEXT"
if (( ${#GPU_IDS_ARRAY[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$WANDB_ENABLED" != "true" ]]; then
  export WANDB_MODE=disabled
fi

printf '%s\n' "$$" >"$LOG_ROOT/launcher.pid"
printf '%s\n' "$RUN_TAG" >"$LOG_ROOT/run_tag"
printf '%s\n' "$REPORT_PATH" >"$LOG_ROOT/report_path"
printf '%s\n' "${GPU_IDS_ARRAY[*]}" >"$LOG_ROOT/gpu_ids"

tasks=(
  brepprediff_seg fusion360seg mfcadpp_seg tmcad_cls solidletters_cls
  cadsynth_seg mfinstseg_seg
)
display_names=(BRepPreDiff Fusion360Seg MFCAD++ TMCAD SolidLetters CADSynth MFInstSeg)
data_configs=(
  data/finetune.yaml
  data/fusion360seg.yaml
  data/mfcad.yaml
  data/tmcad.yaml
  data/solidletters.yaml
  data/cadsynth.yaml
  data/mfinstseg.yaml
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

latest_run() {
  local stage="$1" pattern="$2"
  if [[ ! -d "$RUN_ROOT/$stage" ]]; then
    return 0
  fi
  find "$RUN_ROOT/$stage" -mindepth 1 -maxdepth 1 -type d \
    -name "$pattern" -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
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

wait_for_gpu() {
  local gpu="$1"
  while ! timeout 30s nvidia-smi -i "$gpu" >/dev/null 2>&1; do
    echo "[$(date --iso-8601=seconds)] GPU $gpu unavailable; retrying in 60 seconds"
    sleep 60
  done
}

run_pretrain() {
  local index="$1" gpu="$2"
  local task="${tasks[$index]}" data_config="${data_configs[$index]}"
  local name="single_${task}_train_only_pre${PRETRAIN_EPOCHS}_seed42_$RUN_TAG"
  local run_dir checkpoint log="$LOG_ROOT/${task}_pretrain.log"
  local data_args=()
  run_dir="$(latest_run pretrain "*_${name}")"
  if [[ -n "$run_dir" && -f "$run_dir/checkpoints/last.pt" ]] \
    && grep -q 'finished pretraining' "$run_dir/logs/pretrain.log" 2>/dev/null; then
    printf '%s\n' "$run_dir/checkpoints/last.pt"
    return
  fi

  # CADSynth uses OCC-cleaned splits consistently in pretraining and fine-tuning.
  if [[ "$task" == "cadsynth_seg" ]]; then
    data_args+=(--override "data.train_split=data/splits/cadsynth_train_clean.txt")
  fi
  args=(
    --config "$ROOT_DIR/configs/pretrain.yaml"
    --data-config "$ROOT_DIR/$data_config"
    --override "run.name=$name"
    --override "run.output_dir=$RUN_ROOT"
    --override "run.save_every_epochs=5"
    --override "run.show_progress=false"
    --override "seed=42"
    --override "data.strip_labels=true"
    --override "data.val_split=null"
    --override "data.test_split=null"
    --override "train.epochs=$PRETRAIN_EPOCHS"
    --override "train.batch_size=128"
    --override "train.num_workers=16"
    --override "train.lr=1.0e-4"
    --override "train.lr_scheduler=cosine"
    --override "train.min_lr=0.0"
    --override "train.validate_every_epochs=1"
    --override "train.dataloader_seed=42"
    --override "wandb.enabled=$WANDB_ENABLED"
    "${data_args[@]}"
  )
  checkpoint=""
  if [[ -n "$run_dir" ]]; then
    checkpoint="$(resume_checkpoint_for "$run_dir")"
  fi
  if [[ -n "$checkpoint" ]]; then
    args+=(--override "train.resume=$checkpoint")
    echo "[$(date --iso-8601=seconds)] resume pretrain task=$task gpu=$gpu checkpoint=$checkpoint" >>"$log"
  else
    args+=(--override "train.resume=null")
    echo "[$(date --iso-8601=seconds)] start pretrain task=$task gpu=$gpu train_only=true" >>"$log"
  fi
  GPU_ID="$gpu" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" pretrain "${args[@]}" >>"$log" 2>&1
  run_dir="$(latest_run pretrain "*_${name}")"
  if [[ -z "$run_dir" || ! -f "$run_dir/checkpoints/last.pt" ]]; then
    echo "Pretrain checkpoint missing task=$task" >&2
    return 1
  fi
  printf '%s\n' "$run_dir/checkpoints/last.pt"
}

run_finetune() {
  local index="$1" head="$2" gpu="$3" pretrain_checkpoint="$4"
  local task="${tasks[$index]}" batch="${batches[$index]}" config finetune_head
  local name run_dir checkpoint resume_checkpoint log split_args=()
  if [[ "$head" == "mlp" ]]; then
    config="${mlp_configs[$index]}"
    finetune_head=mlp
  else
    config="${diff_configs[$index]}"
    finetune_head=diffusion
  fi
  name="single_${task}_${head}_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG"
  log="$LOG_ROOT/${task}_${head}.log"
  run_dir="$(latest_run finetune "*_${name}")"
  if [[ -n "$run_dir" && -f "$run_dir/test_metrics.json" ]]; then
    return
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
    --override "run.name=$name"
    --override "run.output_dir=$RUN_ROOT"
    --override "run.save_every_epochs=10"
    --override "run.show_progress=false"
    --override "seed=42"
    --override "model.encoder_type=edge_update_attention"
    --override "model.num_heads=4"
    --override "model.finetune_head=$finetune_head"
    --override "model.graph_pooling=mean_max"
    --override "brep.edge_u_grid_size=10"
    --override "train.pretrain_checkpoint=$pretrain_checkpoint"
    --override "train.epochs=$FINETUNE_EPOCHS"
    --override "train.batch_size=$batch"
    --override "train.gradient_accumulation_steps=1"
    --override "train.num_workers=16"
    --override "train.dataloader_seed=42"
    --override "train.validate_every_epochs=1"
    --override "wandb.enabled=$WANDB_ENABLED"
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
  resume_checkpoint=""
  if [[ -n "$run_dir" ]]; then
    resume_checkpoint="$(resume_checkpoint_for "$run_dir")"
  fi
  if [[ -n "$resume_checkpoint" ]]; then
    args+=(--override "train.resume=$resume_checkpoint")
    echo "[$(date --iso-8601=seconds)] resume finetune task=$task head=$head gpu=$gpu checkpoint=$resume_checkpoint" >>"$log"
  else
    args+=(--override "train.resume=null")
    echo "[$(date --iso-8601=seconds)] start finetune task=$task head=$head gpu=$gpu" >>"$log"
  fi
  GPU_ID="$gpu" "$ROOT_DIR/scripts/run_blendit_gpu4.sh" finetune "${args[@]}" >>"$log" 2>&1
  run_dir="$(latest_run finetune "*_${name}")"
  checkpoint="$run_dir/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Best checkpoint missing task=$task head=$head" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n blendit \
    python -m brepprediff.training.evaluate \
    --checkpoint "$checkpoint" --split test --output "$run_dir/test_metrics.json" \
    --batch-size "$batch" --num-workers 16 --device cuda >>"$log" 2>&1
}

run_task() {
  local index="$1" gpu="$2" pretrain_checkpoint
  wait_for_gpu "$gpu"
  pretrain_checkpoint="$(run_pretrain "$index" "$gpu")"
  run_finetune "$index" mlp "$gpu" "$pretrain_checkpoint"
  run_finetune "$index" diffloss "$gpu" "$pretrain_checkpoint"
  echo "[$(date --iso-8601=seconds)] task complete task=${tasks[$index]} gpu=$gpu"
}

# Long datasets are distributed first to balance the four GPU queues.
queue_orders=("5" "4 3" "6 0" "2 1")
pids=()
for worker in "${!GPU_IDS_ARRAY[@]}"; do
  gpu="${GPU_IDS_ARRAY[$worker]}"
  order="${queue_orders[$((worker % ${#queue_orders[@]}))]}"
  (
    for index in $order; do
      run_task "$index" "$gpu"
    done
  ) >"$LOG_ROOT/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one dataset pipeline failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

report_args=()
for index in "${!tasks[@]}"; do
  mlp_run="$(latest_run finetune "*_single_${tasks[$index]}_mlp_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG")"
  diff_run="$(latest_run finetune "*_single_${tasks[$index]}_diffloss_ft${FINETUNE_EPOCHS}_seed42_$RUN_TAG")"
  report_args+=(--pair "${display_names[$index]}" "$mlp_run" "$diff_run")
done
conda run --no-capture-output -n blendit python scripts/compare_new_occ_all_heads.py \
  "${report_args[@]}" \
  --title "单数据集 train-only 预训练：7 个下游任务 MLP 与 DiffLoss 对照" \
  --epochs "$FINETUNE_EPOCHS" \
  --batch-description "task batch size 256/512、梯度累积 1、GPU ${GPU_IDS_ARRAY[*]} 队列并行" \
  --pretrain-description "每个下游任务各自数据集 train split 上独立训练的 ${PRETRAIN_EPOCHS}-epoch encoder" \
  --output "$REPORT_PATH" >"$LOG_ROOT/comparison.log" 2>&1

echo "[$(date --iso-8601=seconds)] pipeline complete report=$REPORT_PATH"
