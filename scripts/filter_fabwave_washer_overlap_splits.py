#!/usr/bin/env python3
"""Regenerate stratified FabWave splits without Rotary_Shaft or overlapping Washers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile

from blendit.data.tmcad_cls import _allocate_counts


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path("/data/hhfeng/FabWave/CAD16-24/washers_overlapping_orings_302.csv")
SOURCE_PATTERN = "data/splits/fabwave_{split}_clean.txt"
OUTPUT_PATTERN = "data/splits/fabwave_no_rotary_shaft_no_washer_overlap_{split}.txt"
AUDIT_PATH = ROOT / "data/splits/fabwave_no_rotary_shaft_no_washer_overlap_audit.json"
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as stream:
        filenames = [row["washer_filename"].strip() for row in csv.DictReader(stream)]
    excluded = set(filenames)
    if len(filenames) != 302 or len(excluded) != 302:
        raise ValueError(
            f"Expected 302 unique CSV filenames, got rows={len(filenames)} unique={len(excluded)}"
        )

    all_items: set[str] = set()
    for split in SPLITS:
        source = ROOT / SOURCE_PATTERN.format(split=split)
        all_items.update(
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    overlap_items = {
        item
        for item in all_items
        if "Washers" in Path(item).parts and Path(item).name in excluded
    }
    rotary_items = {item for item in all_items if "Rotary_Shaft" in Path(item).parts}
    matched = {Path(item).name for item in overlap_items}

    if matched != excluded:
        raise ValueError(
            f"CSV/split mismatch: unmatched={sorted(excluded - matched)[:10]} "
            f"unexpected={sorted(matched - excluded)[:10]}"
        )
    kept_items = all_items - overlap_items - rotary_items
    by_class: dict[str, list[str]] = {}
    for item in kept_items:
        parts = Path(item).parts
        # FabWave paths are rooted as <collection>/<class>/..., but the source
        # directory below the class is not consistently named STEP.
        if len(parts) < 3:
            raise ValueError(f"Cannot identify FabWave class from split item: {item}")
        by_class.setdefault(parts[1], []).append(item)

    regenerated: dict[str, list[str]] = {split: [] for split in SPLITS}
    for class_name, class_items in sorted(by_class.items()):
        shuffled = sorted(class_items)
        random.Random(f"{SEED}:{class_name}").shuffle(shuffled)
        counts = _allocate_counts(len(shuffled), RATIOS)
        offset = 0
        for split, count in zip(SPLITS, counts):
            regenerated[split].extend(shuffled[offset : offset + count])
            offset += count

    split_audit: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        output = ROOT / OUTPUT_PATTERN.format(split=split)
        items = sorted(regenerated[split])
        atomic_write(output, "".join(f"{item}\n" for item in items))
        split_audit[split] = {"after": len(items)}

    audit = {
        "source_csv": str(CSV_PATH),
        "protocol": "stratified_80_10_10",
        "seed": SEED,
        "source_valid_samples": len(all_items),
        "excluded_rotary_shaft": len(rotary_items),
        "excluded_overlapping_washers": len(overlap_items),
        "remaining_samples": len(kept_items),
        "source_split_pattern": SOURCE_PATTERN,
        "output_split_pattern": OUTPUT_PATTERN,
        "splits": split_audit,
    }
    atomic_write(AUDIT_PATH, json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
