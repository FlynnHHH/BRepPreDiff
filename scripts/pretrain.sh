#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ "${NPROC_PER_NODE:-4}" == "1" ]]; then
  exec "${BREPPREDIFF_PYTHON:-python}" -m brepprediff.training.pretrain \
    --config configs/pretrain.yaml "$@"
fi

exec "${BREPPREDIFF_PYTHON:-python}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  -m brepprediff.training.pretrain --config configs/pretrain.yaml "$@"
