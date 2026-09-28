#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${NPROC_PER_NODE:-1}" == "1" ]]; then
  exec "${BREPPREDIFF_PYTHON:-python}" -m brepprediff.training.finetune \
    --config configs/finetune.yaml "$@" --override model.finetune_head=mlp
fi

exec "${BREPPREDIFF_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m brepprediff.training.finetune --config configs/finetune.yaml \
  "$@" --override model.finetune_head=mlp
