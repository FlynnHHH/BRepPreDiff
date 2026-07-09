from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_CACHE_HASH_SUFFIX = re.compile(r"^(?P<sample>.+)_[0-9a-f]{10}$")


@dataclass(frozen=True)
class FilterResult:
    kept: int
    removed: int
    invalid_ids: int


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def _cache_stem_to_sample_id(stem: str) -> str:
    match = _CACHE_HASH_SUFFIX.match(stem)
    if match:
        return match.group("sample")
    return stem


def split_item_ids(item: str) -> set[str]:
    path = Path(item)
    ids = {item}
    if path.suffix:
        no_suffix = path.with_suffix("").as_posix()
        ids.add(no_suffix)
        ids.add(path.stem)
        ids.add(_cache_stem_to_sample_id(path.stem))
    else:
        ids.add(path.as_posix())
        ids.add(path.name)
    return {value for value in ids if value}


def invalid_record_ids(record: dict) -> set[str]:
    ids: set[str] = set()
    sample_id = record.get("sample_id")
    if sample_id:
        ids.add(str(sample_id))

    for key in ("step_path", "cache_path"):
        value = record.get(key)
        if not value:
            continue
        path = Path(str(value))
        ids.add(path.with_suffix("").as_posix())
        ids.add(path.stem)
        ids.add(_cache_stem_to_sample_id(path.stem))
    return {value for value in ids if value}


def load_invalid_ids(invalid_logs: Iterable[str | Path]) -> set[str]:
    invalid_ids: set[str] = set()
    for invalid_log in invalid_logs:
        path = Path(invalid_log)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid record in {path}:{line_number}: expected object")
                invalid_ids.update(invalid_record_ids(record))
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
    invalid_ids = load_invalid_ids(invalid_logs)
    lines = _read_lines(split_path)
    kept: list[str] = []
    removed: list[str] = []

    for item in lines:
        if split_item_ids(item) & invalid_ids:
            removed.append(item)
        else:
            kept.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{item}\n" for item in kept), encoding="utf-8")

    if removed_output_path is not None:
        removed_path = Path(removed_output_path)
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.write_text("".join(f"{item}\n" for item in removed), encoding="utf-8")

    return FilterResult(kept=len(kept), removed=len(removed), invalid_ids=len(invalid_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove invalid cache/build samples from a split file.")
    parser.add_argument("--split", required=True, help="Input split file.")
    parser.add_argument(
        "--invalid-log",
        action="append",
        required=True,
        help="JSONL invalid log from cache building. Can be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output filtered split file.")
    parser.add_argument("--removed-output", default=None, help="Optional file listing removed split entries.")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
