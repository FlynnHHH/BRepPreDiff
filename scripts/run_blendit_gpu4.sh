#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-4}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {pretrain|finetune} [BRepPreDiff arguments...]" >&2
  exit 2
fi

stage="$1"
shift
case "$stage" in
  pretrain|finetune)
    module="brepprediff.training.${stage}"
    ;;
  *)
    echo "Unknown training stage: $stage (expected pretrain or finetune)" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
exec conda run --no-capture-output -n blendit \
  env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python -m "$module" "$@"
