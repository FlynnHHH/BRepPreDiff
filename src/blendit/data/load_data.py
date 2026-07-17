from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from blendit.config import apply_overrides, load_config
from blendit.data.dataset import (
    CacheFailure,
    StepSegDataset,
    _configured_cache_dir,
    _iter_step_files,
    _match_seg,
    _read_split,
    _resolve_split_cache_for_item,
    _resolve_split_step,
)


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
_CACHE_HASH_SUFFIX = re.compile(r"^(?P<sample>.+)_[0-9a-f]{10}$")


@dataclass(frozen=True)
class InvalidCacheRemoval:
    cache_path: str
    error: str


@dataclass(frozen=True)
class DataPreparationResult:
    config: dict[str, Any]
    generated_splits: dict[str, str]
    built_cache_files: int
    removed_invalid_caches: tuple[InvalidCacheRemoval, ...]
    cache_failures: tuple[CacheFailure, ...]


class DataPreparationError(RuntimeError):
    def __init__(self, message: str, result: DataPreparationResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CacheScanFailure:
    sample_id: str
    cache_path: str
    error_type: str
    error: str


@dataclass(frozen=True)
class FilterResult:
    kept: int
    removed: int
    invalid_ids: int


def _cache_stem_to_sample_id(stem: str) -> str:
    match = _CACHE_HASH_SUFFIX.match(stem)
    return match.group("sample") if match else stem


def _scan_npz(path: Path, *, max_abs: float | None = None) -> list[str]:
    problems: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            array = data[key]
            if not np.issubdtype(array.dtype, np.floating):
                continue
            finite = np.isfinite(array)
            if not finite.all():
                bad_count = int(array.size - int(finite.sum()))
                problems.append(f"{key}: {bad_count} non-finite values")
            if max_abs is not None and array.size:
                finite_values = array[finite]
                if finite_values.size:
                    observed = float(np.max(np.abs(finite_values)))
                    if observed > max_abs:
                        problems.append(f"{key}: max_abs {observed:.6g} > {max_abs}")
    return problems


def scan_cache(
    cache_dir: str | Path,
    *,
    invalid_log: str | Path | None = None,
    max_abs: float | None = None,
    limit: int | None = None,
) -> tuple[int, list[CacheScanFailure]]:
    cache_dir = Path(cache_dir)
    cache_paths = sorted(path for path in cache_dir.rglob("*.npz") if path.is_file())
    if limit is not None:
        cache_paths = cache_paths[: max(0, int(limit))]

    failures: list[CacheScanFailure] = []
    for cache_path in tqdm(cache_paths, desc=f"scan {cache_dir}"):
        sample_id = _cache_stem_to_sample_id(cache_path.stem)
        try:
            problems = _scan_npz(cache_path, max_abs=max_abs)
        except Exception as exc:
            failures.append(
                CacheScanFailure(
                    sample_id=sample_id,
                    cache_path=str(cache_path),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            continue
        if problems:
            failures.append(
                CacheScanFailure(
                    sample_id=sample_id,
                    cache_path=str(cache_path),
                    error_type="NonFiniteCache",
                    error="; ".join(problems),
                )
            )

    if invalid_log is not None:
        path = Path(invalid_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure.__dict__, ensure_ascii=False) + "\n")
    return len(cache_paths), failures


def split_item_ids(item: str) -> set[str]:
    path = Path(item)
    ids = {item}
    if path.suffix:
        ids.update(
            {
                path.with_suffix("").as_posix(),
                path.stem,
                _cache_stem_to_sample_id(path.stem),
            }
        )
    else:
        ids.update({path.as_posix(), path.name})
    return {value for value in ids if value}


def _invalid_record_ids(record: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    if record.get("sample_id"):
        ids.add(str(record["sample_id"]))
    for key in ("step_path", "cache_path"):
        if not record.get(key):
            continue
        path = Path(str(record[key]))
        ids.update(
            {
                path.with_suffix("").as_posix(),
                path.stem,
                _cache_stem_to_sample_id(path.stem),
            }
        )
    return {value for value in ids if value}


def _load_invalid_ids(invalid_logs: Iterable[str | Path]) -> set[str]:
    invalid_ids: set[str] = set()
    for invalid_log in invalid_logs:
        path = Path(invalid_log)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not (line := line.strip()):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid record in {path}:{line_number}: expected object")
                invalid_ids.update(_invalid_record_ids(record))
    return invalid_ids


def filter_split(
    split_path: str | Path,
    invalid_logs: Iterable[str | Path],
    output_path: str | Path,
    *,
    removed_output_path: str | Path | None = None,
) -> FilterResult:
    split_path = Path(split_path)
    output_path = Path(output_path)
    invalid_ids = _load_invalid_ids(invalid_logs)
    lines = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    kept: list[str] = []
    removed: list[str] = []
    for item in lines:
        (removed if split_item_ids(item) & invalid_ids else kept).append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{item}\n" for item in kept), encoding="utf-8")
    if removed_output_path is not None:
        removed_path = Path(removed_output_path)
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.write_text("".join(f"{item}\n" for item in removed), encoding="utf-8")
    return FilterResult(kept=len(kept), removed=len(removed), invalid_ids=len(invalid_ids))


def _configured_split_paths(data_cfg: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        value = data_cfg.get(f"{split}_split")
        if value:
            paths[split] = Path(value)
    return paths


def split_has_items(config: dict[str, Any], split: str) -> bool:
    value = config.get("data", {}).get(f"{split}_split")
    if not value:
        return False
    path = Path(value)
    return bool(path.is_file() and _read_split(path))


def _discover_source_items(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    steps_dir = Path(data_cfg["steps_dir"])
    segs_dir = Path(data_cfg["segs_dir"])
    if not steps_dir.is_dir():
        raise FileNotFoundError(f"STEP directory does not exist: {steps_dir}")
    if bool(data_cfg.get("labels_required", True)) and not segs_dir.is_dir():
        raise FileNotFoundError(f"SEG/JSON directory does not exist: {segs_dir}")

    extensions = data_cfg.get("step_extensions", [".step", ".stp"])
    labels_required = bool(data_cfg.get("labels_required", True))
    items: list[str] = []
    missing_labels = 0
    for step_path in _iter_step_files(steps_dir, extensions):
        if labels_required and _match_seg(segs_dir, steps_dir, step_path) is None:
            missing_labels += 1
            continue
        items.append(step_path.relative_to(steps_dir).as_posix())

    if not items:
        detail = f"; skipped {missing_labels} STEP files without labels" if missing_labels else ""
        raise ValueError(f"No usable STEP samples found under {steps_dir}{detail}.")
    if missing_labels:
        print(f"split discovery: skipped {missing_labels} STEP files without SEG/JSON labels")
    return items


def _split_item_key(
    steps_dir: Path,
    item: str,
    extensions: Iterable[str],
) -> str:
    try:
        step_path = _resolve_split_step(steps_dir, item, extensions)
        return step_path.relative_to(steps_dir).with_suffix("").as_posix()
    except (FileNotFoundError, ValueError):
        path = Path(item)
        return path.with_suffix("").as_posix() if path.suffix else path.as_posix()


def _allocate_counts(total: int, splits: list[str], ratios: dict[str, float]) -> dict[str, int]:
    if not splits:
        return {}
    if len(splits) == 1:
        return {splits[0]: total}

    weights = {split: max(0.0, float(ratios.get(split, 0.0))) for split in splits}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {split: 1.0 for split in splits}
        weight_sum = float(len(splits))

    raw = {split: total * weights[split] / weight_sum for split in splits}
    counts = {split: int(math.floor(raw[split])) for split in splits}
    remainder = total - sum(counts.values())
    order = sorted(splits, key=lambda split: (-(raw[split] - counts[split]), splits.index(split)))
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def prepare_split_files(config: dict[str, Any]) -> dict[str, str]:
    data_cfg = config["data"]
    split_paths = _configured_split_paths(data_cfg)
    if "train" not in split_paths:
        raise ValueError("data.train_split must specify a split path; use null only for disabled val/test splits.")

    missing_splits = [split for split, path in split_paths.items() if not path.is_file()]
    if not missing_splits:
        return {}

    source_items = _discover_source_items(config)
    steps_dir = Path(data_cfg["steps_dir"])
    extensions = data_cfg.get("step_extensions", [".step", ".stp"])
    used_keys: set[str] = set()
    for split, path in split_paths.items():
        if split in missing_splits:
            continue
        for item in _read_split(path) or []:
            used_keys.add(_split_item_key(steps_dir, item, extensions))

    available = [
        item
        for item in source_items
        if _split_item_key(steps_dir, item, extensions) not in used_keys
    ]
    random.Random(int(data_cfg.get("split_seed", config.get("seed", 42)))).shuffle(available)
    configured_ratios = data_cfg.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    ratios = configured_ratios if isinstance(configured_ratios, dict) else DEFAULT_SPLIT_RATIOS
    counts = _allocate_counts(len(available), missing_splits, ratios)

    generated: dict[str, str] = {}
    offset = 0
    for split in missing_splits:
        path = split_paths[split]
        count = counts[split]
        items = sorted(available[offset : offset + count])
        offset += count
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")
        generated[split] = str(path)
        print(f"generated {split} split: path={path} samples={len(items)}")
    return generated


def _cache_dirs_for_splits(config: dict[str, Any], split_paths: dict[str, Path]) -> list[Path]:
    data_cfg = config["data"]
    unique: dict[Path, Path] = {}
    for split in split_paths:
        cache_dir = _configured_cache_dir(data_cfg, split)
        unique[cache_dir.resolve()] = cache_dir
    return list(unique.values())


def remove_invalid_caches(cache_dirs: Iterable[Path]) -> list[InvalidCacheRemoval]:
    removed: list[InvalidCacheRemoval] = []
    seen: set[Path] = set()
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        for cache_path in sorted(path for path in cache_dir.rglob("*.npz") if path.is_file()):
            resolved = cache_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                problems = _scan_npz(cache_path)
                error = "; ".join(problems)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            if not error:
                continue
            cache_path.unlink()
            removed.append(InvalidCacheRemoval(cache_path=str(cache_path), error=error))
            print(f"removed invalid cache: {cache_path} ({error})")
    return removed


def _missing_cache_items(config: dict[str, Any], split: str, split_path: Path) -> list[str]:
    data_cfg = config["data"]
    cache_dir = _configured_cache_dir(data_cfg, split)
    steps_dir = Path(data_cfg["steps_dir"])
    extensions = data_cfg.get("step_extensions", [".step", ".stp"])
    missing: list[str] = []
    for item in _read_split(split_path) or []:
        try:
            _resolve_split_cache_for_item(cache_dir, steps_dir, item, extensions)
        except (FileNotFoundError, ValueError):
            missing.append(item)
    return missing


def prepare_data(config: dict[str, Any], *, num_workers: int | None = None) -> DataPreparationResult:
    prepared_config = copy.deepcopy(config)
    generated_splits = prepare_split_files(prepared_config)
    split_paths = _configured_split_paths(prepared_config["data"])
    cache_dirs = _cache_dirs_for_splits(prepared_config, split_paths)

    removed_invalid = remove_invalid_caches(cache_dirs)
    built_cache_files = 0
    failures: list[CacheFailure] = []

    for split, split_path in split_paths.items():
        items = _read_split(split_path) or []
        if not items:
            print(f"cache {split}: split is empty, skipping")
            continue
        missing_before = _missing_cache_items(prepared_config, split, split_path)
        if not missing_before:
            print(f"cache {split}: complete ({len(items)} samples)")
            continue

        print(f"cache {split}: missing={len(missing_before)} total={len(items)}, building")
        builder = StepSegDataset(prepared_config, split=split, source_mode=True)
        builder.overwrite_cache = False
        pending_paths = {
            sample.cache_path
            for sample in builder.samples
            if not sample.cache_path.exists()
        }
        split_failures = builder.build_cache(num_workers=num_workers)
        failures.extend(split_failures)
        built_cache_files += sum(path.is_file() for path in pending_paths)

    removed_invalid.extend(remove_invalid_caches(cache_dirs))

    incomplete: dict[str, list[str]] = {}
    for split, split_path in split_paths.items():
        missing = _missing_cache_items(prepared_config, split, split_path)
        if missing:
            incomplete[split] = missing

    result = DataPreparationResult(
        config=prepared_config,
        generated_splits=generated_splits,
        built_cache_files=built_cache_files,
        removed_invalid_caches=tuple(removed_invalid),
        cache_failures=tuple(failures),
    )
    if incomplete:
        details = ", ".join(f"{split}={len(items)}" for split, items in incomplete.items())
        raise DataPreparationError(
            f"Cache preparation is incomplete after removing invalid files: {details}. "
            "See the invalid JSONL logs for extraction failures.",
            result,
        )
    return result


def prepare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing split files, complete B-Rep caches, and delete invalid caches."
    )
    parser.add_argument("--config", default="data/default.yaml")
    parser.add_argument("--workers", type=int, default=None, help="Parallel OCC cache workers.")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.override)
    result = prepare_data(config, num_workers=args.workers)
    print(
        "data ready: "
        f"generated_splits={len(result.generated_splits)} "
        f"built_cache_files={result.built_cache_files} "
        f"removed_invalid_caches={len(result.removed_invalid_caches)}"
    )


def cache_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build B-Rep feature cache for one configured split.")
    parser.add_argument("--config", default="data/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--invalid-log", default=None)
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite-cache", dest="overwrite_cache", action="store_true")
    overwrite_group.add_argument("--no-overwrite-cache", dest="overwrite_cache", action="store_false")
    parser.set_defaults(overwrite_cache=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.override)
    if args.overwrite_cache is not None:
        config.setdefault("data", {})["overwrite_cache"] = args.overwrite_cache
    dataset = StepSegDataset(config, split=args.split, source_mode=True)
    failures = dataset.build_cache(num_workers=args.workers, invalid_log=args.invalid_log)
    if failures:
        print(
            f"cache completed with {len(failures)} invalid samples; "
            f"see {dataset._invalid_log_path(args.invalid_log)}"
        )


def filter_split_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Remove invalid cache/build samples from a split file.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--invalid-log", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--removed-output", default=None)
    args = parser.parse_args(argv)

    result = filter_split(
        args.split,
        args.invalid_log,
        args.output,
        removed_output_path=args.removed_output,
    )
    print(
        f"filtered split: input={args.split} output={args.output} "
        f"kept={result.kept} removed={result.removed} invalid_ids={result.invalid_ids}"
    )


def scan_cache_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan cached B-Rep graph npz files for invalid values.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--invalid-log", default=None)
    parser.add_argument("--max-abs", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    total, failures = scan_cache(
        args.cache_dir,
        invalid_log=args.invalid_log,
        max_abs=args.max_abs,
        limit=args.limit,
    )
    print(f"cache scan complete: files={total} invalid={len(failures)}")
    for failure in failures[:20]:
        print(f"{failure.sample_id}: {failure.error}")
    if len(failures) > 20:
        print(f"... {len(failures) - 20} more invalid cache files")
    if failures:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "prepare": prepare_main,
        "cache": cache_main,
        "filter-split": filter_split_main,
        "scan-cache": scan_cache_main,
    }
    if args and args[0] in commands:
        commands[args[0]](args[1:])
    else:
        prepare_main(args)


if __name__ == "__main__":
    main()
