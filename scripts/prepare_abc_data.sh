#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/nvme03/hhfeng/Blendit}"
ABC_ROOT="${ABC_ROOT:-/home/nvme03/hhfeng/ABCdataset}"

STEP_NAMES_JSON="${STEP_NAMES_JSON:-${PROJECT_DIR}/step_names.json}"
SAMPLED_NAMES_JSON="${SAMPLED_NAMES_JSON:-${PROJECT_DIR}/step_sampled_30000_names.json}"

PRETRAIN_STEPS_DIR="${PRETRAIN_STEPS_DIR:-step}"
PRETRAIN_SEGS_DIR="${PRETRAIN_SEGS_DIR:-seg}"
FINETUNE_FLAT_DIR="${FINETUNE_FLAT_DIR:-all_flat}"

SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/abc_splits}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/data/cache/abc_features}"
PRETRAIN_CACHE_DIR="${PRETRAIN_CACHE_DIR:-${CACHE_ROOT}/pretrain}"
FINETUNE_CACHE_DIR="${FINETUNE_CACHE_DIR:-${CACHE_ROOT}/finetune}"
INVALID_DIR="${INVALID_DIR:-${CACHE_ROOT}/invalid}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/default.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_CACHE="${RUN_CACHE:-1}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-false}"
FINETUNE_VAL_RATIO="${FINETUNE_VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
CACHE_WORKERS="${CACHE_WORKERS:-0}"
PRETRAIN_CACHE_WORKERS="${PRETRAIN_CACHE_WORKERS:-${CACHE_WORKERS}}"
FINETUNE_CACHE_WORKERS="${FINETUNE_CACHE_WORKERS:-${CACHE_WORKERS}}"
SCREEN_NAME_PREFIX="${SCREEN_NAME_PREFIX:-blendit_prepare_abc}"

PRETRAIN_TRAIN_SPLIT="${SPLIT_DIR}/pretrain_train.txt"
PRETRAIN_TRAIN_CLEAN_SPLIT="${SPLIT_DIR}/pretrain_train_clean.txt"
FINETUNE_TRAIN_SPLIT="${SPLIT_DIR}/finetune_train.txt"
FINETUNE_TRAIN_CLEAN_SPLIT="${SPLIT_DIR}/finetune_train_clean.txt"
FINETUNE_VAL_SPLIT="${SPLIT_DIR}/finetune_val.txt"
FINETUNE_VAL_CLEAN_SPLIT="${SPLIT_DIR}/finetune_val_clean.txt"

launch_in_screen_if_needed() {
  if [[ "${NO_SCREEN:-0}" == "1" || -n "${STY:-}" ]]; then
    return
  fi

  if ! command -v screen >/dev/null 2>&1; then
    echo "[screen] screen not found; running in foreground. Set NO_SCREEN=1 to keep this behavior explicit."
    return
  fi

  local started_at log_dir script_path quoted_project quoted_script quoted_log quoted_args inner_cmd
  started_at="$(date +%Y%m%d_%H%M%S)"
  SCREEN_NAME="${SCREEN_NAME:-${SCREEN_NAME_PREFIX}_${started_at}}"
  SCREEN_LOG="${SCREEN_LOG:-${PROJECT_DIR}/runs/logs/${SCREEN_NAME}.log}"
  log_dir="$(dirname "${SCREEN_LOG}")"
  mkdir -p "${log_dir}"

  script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  printf -v quoted_project "%q" "${PROJECT_DIR}"
  printf -v quoted_script "%q" "${script_path}"
  printf -v quoted_log "%q" "${SCREEN_LOG}"
  quoted_args=""
  if (($# > 0)); then
    printf -v quoted_args " %q" "$@"
  fi
  inner_cmd="set -euo pipefail; cd ${quoted_project}; NO_SCREEN=1 ${quoted_script}${quoted_args} 2>&1 | tee -a ${quoted_log}"

  echo "[screen] Starting detached session: ${SCREEN_NAME}"
  echo "[screen] Log: ${SCREEN_LOG}"
  echo "[screen] Attach: screen -r ${SCREEN_NAME}"
  if ! screen -dmS "${SCREEN_NAME}" bash -lc "${inner_cmd}"; then
    echo "[screen] Failed to start screen. Re-run with NO_SCREEN=1 for foreground debugging." >&2
    exit 1
  fi
  sleep "${SCREEN_STARTUP_WAIT:-1}"
  if ! screen -ls | grep -Fq ".${SCREEN_NAME}" && [[ ! -s "${SCREEN_LOG}" ]]; then
    echo "[screen] Started command returned, but no live session or non-empty log file was found." >&2
    echo "[screen] Re-run with NO_SCREEN=1 for foreground debugging." >&2
    exit 1
  fi
  exit 0
}

launch_in_screen_if_needed "$@"

cd "${PROJECT_DIR}"
mkdir -p "${SPLIT_DIR}" "${PRETRAIN_CACHE_DIR}" "${FINETUNE_CACHE_DIR}" "${INVALID_DIR}"
export STEP_NAMES_JSON SAMPLED_NAMES_JSON SPLIT_DIR ABC_ROOT FINETUNE_FLAT_DIR SEED FINETUNE_VAL_RATIO

echo "[1/3] Writing split files to ${SPLIT_DIR}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from pathlib import Path

step_names_path = Path(os.environ["STEP_NAMES_JSON"])
sampled_names_path = Path(os.environ["SAMPLED_NAMES_JSON"])
split_dir = Path(os.environ["SPLIT_DIR"])
abc_root = Path(os.environ["ABC_ROOT"])
finetune_flat_dir = os.environ["FINETUNE_FLAT_DIR"]
seed = int(os.environ["SEED"])
val_ratio = float(os.environ["FINETUNE_VAL_RATIO"])


def load_model_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "model_ids" in payload:
        values = payload["model_ids"]
    elif isinstance(payload, dict) and "files" in payload:
        values = [Path(item).stem for item in payload["files"]]
    elif isinstance(payload, list):
        values = [Path(item).stem for item in payload]
    else:
        raise ValueError(f"Unsupported names JSON format: {path}")
    return sorted({str(item).strip() for item in values if str(item).strip()})


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


all_ids = load_model_ids(step_names_path)
sampled_ids = load_model_ids(sampled_names_path)
sampled_set = set(sampled_ids)

pretrain_ids = [model_id for model_id in all_ids if model_id not in sampled_set]
pretrain_steps = [f"{model_id}.step" for model_id in pretrain_ids]

finetune_dir = abc_root / finetune_flat_dir
finetune_step_ids = {path.stem for path in finetune_dir.glob("*.step")}
finetune_seg_ids = {path.stem for path in finetune_dir.glob("*.seg")}
finetune_ids = sorted(finetune_step_ids & finetune_seg_ids)
all_flat_step_only = len(finetune_step_ids - finetune_seg_ids)
all_flat_seg_only = len(finetune_seg_ids - finetune_step_ids)
all_flat_not_in_sampled = len(set(finetune_ids) - sampled_set)
sampled_missing_in_all_flat = len(sampled_set - set(finetune_ids))

rng = random.Random(seed)
rng.shuffle(finetune_ids)
val_count = int(round(len(finetune_ids) * val_ratio))
val_ids = sorted(finetune_ids[:val_count])
train_ids = sorted(finetune_ids[val_count:])

split_dir.mkdir(parents=True, exist_ok=True)
write_lines(split_dir / "pretrain_train.txt", pretrain_steps)
write_lines(split_dir / "finetune_train.txt", [f"{model_id}.step" for model_id in train_ids])
write_lines(split_dir / "finetune_val.txt", [f"{model_id}.step" for model_id in val_ids])

print(f"all step ids:       {len(all_ids)}")
print(f"sampled ids:        {len(sampled_ids)}")
print(f"pretrain train ids: {len(pretrain_ids)}")
print(f"finetune train ids: {len(train_ids)}")
print(f"finetune val ids:   {len(val_ids)}")
print(f"all_flat step-only ids: {all_flat_step_only}")
print(f"all_flat seg-only ids:  {all_flat_seg_only}")
print(f"all_flat ids not in sampled JSON: {all_flat_not_in_sampled}")
print(f"sampled JSON ids missing in all_flat: {sampled_missing_in_all_flat}")
PY

if [[ "${RUN_CACHE}" == "0" ]]; then
  echo "[2/3] RUN_CACHE=0, split files generated only."
  echo "[3/3] Done."
  exit 0
fi

echo "Cache overwrite: ${OVERWRITE_CACHE}"

echo "[2/3] Building pretrain cache in ${PRETRAIN_CACHE_DIR}"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.dataset \
  --config "${CONFIG_PATH}" \
  --split train \
  --workers "${PRETRAIN_CACHE_WORKERS}" \
  --invalid-log "${INVALID_DIR}/pretrain_train_invalid.jsonl" \
  --override "data.root=${ABC_ROOT}" \
  --override "data.steps_dir=${PRETRAIN_STEPS_DIR}" \
  --override "data.segs_dir=${PRETRAIN_SEGS_DIR}" \
  --override "data.cache_dir=${PRETRAIN_CACHE_DIR}" \
  --override "data.train_split=${PRETRAIN_TRAIN_SPLIT}" \
  --override "data.val_split=null" \
  --override "data.test_split=null" \
  --override "data.labels_required=false" \
  --override "data.overwrite_cache=${OVERWRITE_CACHE}"

PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.filter_split \
  --split "${PRETRAIN_TRAIN_SPLIT}" \
  --invalid-log "${INVALID_DIR}/pretrain_train_invalid.jsonl" \
  --output "${PRETRAIN_TRAIN_CLEAN_SPLIT}" \
  --removed-output "${SPLIT_DIR}/pretrain_train_invalid_removed.txt"

echo "[3/3] Building finetune cache in ${FINETUNE_CACHE_DIR}"
PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.dataset \
  --config "${CONFIG_PATH}" \
  --split train \
  --workers "${FINETUNE_CACHE_WORKERS}" \
  --invalid-log "${INVALID_DIR}/finetune_train_invalid.jsonl" \
  --override "data.root=${ABC_ROOT}" \
  --override "data.steps_dir=${FINETUNE_FLAT_DIR}" \
  --override "data.segs_dir=${FINETUNE_FLAT_DIR}" \
  --override "data.cache_dir=${FINETUNE_CACHE_DIR}" \
  --override "data.train_split=${FINETUNE_TRAIN_SPLIT}" \
  --override "data.val_split=${FINETUNE_VAL_SPLIT}" \
  --override "data.test_split=null" \
  --override "data.labels_required=true" \
  --override "data.overwrite_cache=${OVERWRITE_CACHE}"

PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.filter_split \
  --split "${FINETUNE_TRAIN_SPLIT}" \
  --invalid-log "${INVALID_DIR}/finetune_train_invalid.jsonl" \
  --output "${FINETUNE_TRAIN_CLEAN_SPLIT}" \
  --removed-output "${SPLIT_DIR}/finetune_train_invalid_removed.txt"

PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.dataset \
  --config "${CONFIG_PATH}" \
  --split val \
  --workers "${FINETUNE_CACHE_WORKERS}" \
  --invalid-log "${INVALID_DIR}/finetune_val_invalid.jsonl" \
  --override "data.root=${ABC_ROOT}" \
  --override "data.steps_dir=${FINETUNE_FLAT_DIR}" \
  --override "data.segs_dir=${FINETUNE_FLAT_DIR}" \
  --override "data.cache_dir=${FINETUNE_CACHE_DIR}" \
  --override "data.train_split=${FINETUNE_TRAIN_SPLIT}" \
  --override "data.val_split=${FINETUNE_VAL_SPLIT}" \
  --override "data.test_split=null" \
  --override "data.labels_required=true" \
  --override "data.overwrite_cache=${OVERWRITE_CACHE}"

PYTHONPATH="${PROJECT_DIR}/src" "${PYTHON_BIN}" -m blendit.data.filter_split \
  --split "${FINETUNE_VAL_SPLIT}" \
  --invalid-log "${INVALID_DIR}/finetune_val_invalid.jsonl" \
  --output "${FINETUNE_VAL_CLEAN_SPLIT}" \
  --removed-output "${SPLIT_DIR}/finetune_val_invalid_removed.txt"

echo "Done."
