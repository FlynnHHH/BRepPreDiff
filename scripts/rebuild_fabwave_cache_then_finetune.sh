#!/usr/bin/env bash
# Rebuild FabWave caches for the current OCC schema, then start a fresh
# 200-epoch DiffLoss fine-tuning run and evaluate its best checkpoint.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_BIN="${CACHE_BIN:-/home/hhfeng/miniconda3/envs/blendit/bin/blendit-cache}"
RUN_TAG="${RUN_TAG:-new_occ_fabwave_diffloss_200_$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$ROOT_DIR/runs/launch_logs/fabwave_cache_$RUN_TAG"

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
echo "[$(date --iso-8601=seconds)] rebuilding FabWave cache with workers=$workers"
for split in train val test; do
  "$CACHE_BIN" \
    --config data/fabwave_no_rotary_shaft_no_washer_overlap_min10.yaml \
    --split "$split" --workers "$workers" --overwrite-cache \
    >>"$LOG_DIR/cache.log" 2>&1
done
echo "[$(date --iso-8601=seconds)] cache rebuild complete; starting fresh FabWave fine-tuning"

env TASK_GROUP=fabwave RUN_TAG="$RUN_TAG" \
  bash "$ROOT_DIR/scripts/run_edge_update_xstart_epsilon_five.sh"

