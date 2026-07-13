from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


_CACHE_HASH_SUFFIX = re.compile(r"^(?P<sample>.+)_[0-9a-f]{10}$")


@dataclass(frozen=True)
class CacheScanFailure:
    sample_id: str
    cache_path: str
    error_type: str
    error: str


def _sample_id_from_cache_path(path: Path) -> str:
    match = _CACHE_HASH_SUFFIX.match(path.stem)
    if match:
        return match.group("sample")
    return path.stem


def _scan_npz(path: Path, *, max_abs: float | None = None) -> list[str]:
    problems: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            array = data[key]
            if np.issubdtype(array.dtype, np.floating):
                finite = np.isfinite(array)
                if not finite.all():
                    bad_count = int(array.size - int(finite.sum()))
                    problems.append(f"{key}: {bad_count} non-finite values")
                if max_abs is not None and array.size:
                    finite_values = array[finite]
                    if finite_values.size and float(np.max(np.abs(finite_values))) > max_abs:
                        problems.append(f"{key}: max_abs {float(np.max(np.abs(finite_values))):.6g} > {max_abs}")
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
        sample_id = _sample_id_from_cache_path(cache_path)
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
        with path.open("w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure.__dict__, ensure_ascii=False) + "\n")

    return len(cache_paths), failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan cached B-Rep graph npz files for NaN/Inf values.")
    parser.add_argument("--cache-dir", required=True, help="Directory containing cached .npz files.")
    parser.add_argument("--invalid-log", default=None, help="Optional JSONL output for invalid cache files.")
    parser.add_argument("--max-abs", type=float, default=None, help="Optional threshold for unusually large finite values.")
    parser.add_argument("--limit", type=int, default=None, help="Scan only the first N cache files.")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
