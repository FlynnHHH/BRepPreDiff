#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="/home/hhfeng/miniconda3/bin/conda"
CONDA_ENV="blendit"

# Prevent each OCC cache process from spawning its own BLAS/OpenMP pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

choose_workers() {
  local cores load_raw load_floor reserve workers
  local -a load_fields
  cores="$(nproc)"
  read -ra load_fields < /proc/loadavg
  # Five-minute load is less sensitive to short process-startup spikes.
  load_raw="${load_fields[1]}"
  load_floor="${load_raw%%.*}"
  reserve=$((cores / 8))
  if ((reserve < 4)); then
    reserve=4
  fi
  workers=$((cores - load_floor - reserve))
  if ((workers < 12)); then
    workers=12
  elif ((workers > 24)); then
    workers=24
  fi
  printf '%d' "$workers"
}

run_split() {
  local config="$1" split="$2" workers free_gb
  workers="$(choose_workers)"
  free_gb="$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')"
  if ((free_gb < 15)); then
    echo "[$(date --iso-8601=seconds)] abort: only ${free_gb} GB free" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] config=$config split=$split workers=$workers free_gb=$free_gb"
  nice -n 5 "$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV" \
    blendit-cache \
    --config "$config" \
    --split "$split" \
    --workers "$workers" \
    --overwrite-cache
}

cd "$PROJECT_ROOT"

run_split data/pretrain.yaml train

for split in train val test; do
  run_split data/tmcad.yaml "$split"
done

for split in train val test; do
  run_split data/fusion360seg.yaml "$split"
done

for split in train val test; do
  run_split data/mfcad.yaml "$split"
done

echo "[$(date --iso-8601=seconds)] all OCC grid caches rebuilt"
