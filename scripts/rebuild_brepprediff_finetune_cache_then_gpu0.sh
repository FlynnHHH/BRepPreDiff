#!/usr/bin/env bash
# Rebuild the BRepPreDiff downstream cache for the current OCC schema, then run the
# two GPU-0 downstream jobs. The other three jobs can continue independently.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/python}"
CACHE_BIN="${CACHE_BIN:-/home/hhfeng/miniconda3/envs/brepprediff/bin/brepprediff-cache}"
RUN_TAG="${RUN_TAG:-new_occ_diffloss_200_gpu0_$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT_DIR/runs/launch_logs/brepprediff_finetune_cache_$RUN_TAG"

choose_workers() {
  local cores load_floor reserve workers
  cores="$(nproc)"
  load_floor="$(awk '{print int($1)}' /proc/loadavg)"
  reserve=$((cores / 8))
  ((reserve < 4)) && reserve=4
  workers=$((cores - load_floor - reserve))
  ((workers < 8)) && workers=8
  ((workers > 24)) && workers=24
  printf '%d' "$workers"
}

mkdir -p "$LOG_DIR"
printf '%s\n' "$$" >"$LOG_DIR/launcher.pid"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

workers="$(choose_workers)"
echo "[$(date --iso-8601=seconds)] rebuilding BRepPreDiff downstream cache with workers=$workers"
for split in train val test; do
  "$CACHE_BIN" --config data/finetune.yaml --split "$split" \
    --workers "$workers" --overwrite-cache \
    >>"$LOG_DIR/cache.log" 2>&1
done
echo "[$(date --iso-8601=seconds)] cache rebuild complete; starting GPU-0 queue"

env PYTHON_BIN="$PYTHON_BIN" TASK_GROUP=gpu0 RUN_TAG="$RUN_TAG" \
  bash "$ROOT_DIR/scripts/run_edge_update_xstart_epsilon_five.sh"

