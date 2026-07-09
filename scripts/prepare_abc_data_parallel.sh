#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pretrain STEP extraction is the expensive part. Tune this first.
export PRETRAIN_CACHE_WORKERS="${PRETRAIN_CACHE_WORKERS:-8}"

# Fine-tune has only a few thousand files; keep this modest to avoid excess IO.
export FINETUNE_CACHE_WORKERS="${FINETUNE_CACHE_WORKERS:-2}"

# The main script launches itself in screen by default. Use a distinct prefix so
# parallel runs are easy to identify with `screen -ls`.
export SCREEN_NAME_PREFIX="${SCREEN_NAME_PREFIX:-blendit_prepare_abc_parallel}"

exec "${SCRIPT_DIR}/prepare_abc_data.sh" "$@"
