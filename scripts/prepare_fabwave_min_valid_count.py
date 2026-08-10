#!/usr/bin/env python3
"""Build a remapped FabWave dataset after dropping classes below a valid-count threshold."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np

from blendit.data.dataset import _cache_path
from blendit.data.graph import save_graph_npz


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--dataset-root", default="/data/hhfeng/FabWave")
    parser.add_argument("--source-cache", default="data/cache/features/fabwave_no_rotary_shaft")
    parser.add_argument("--source-class-map", default="data/splits/fabwave_no_rotary_shaft_class_map.json")
    parser.add_argument(
        "--source-split-pattern",
        default="data/splits/fabwave_no_rotary_shaft_no_washer_overlap_{split}.txt",
    )
    parser.add_argument(
        "--output-cache",
        default="data/cache/features/fabwave_no_rotary_shaft_no_washer_overlap_min10",
    )
    parser.add_argument(
        "--output-labels",
        default="data/labels/fabwave_no_rotary_shaft_no_washer_overlap_min10",
    )
    parser.add_argument(
        "--output-class-map",
        default="data/splits/fabwave_no_rotary_shaft_no_washer_overlap_min10_class_map.json",
    )
    parser.add_argument(
        "--output-split-pattern",
        default="data/splits/fabwave_no_rotary_shaft_no_washer_overlap_min10_{split}.txt",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.min_count < 1:
        raise ValueError("--min-count must be positive")

    dataset_root = resolve(args.dataset_root)
    source_cache = resolve(args.source_cache)
    output_cache = resolve(args.output_cache)
    output_labels = resolve(args.output_labels)
    source_map_path = resolve(args.source_class_map)
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    source_classes = source_map["classes"]
    source_ids = {row["class_name"]: int(row["class_id"]) for row in source_classes}

    split_items: dict[str, list[str]] = {}
    valid_counts: Counter[str] = Counter()
    for split in SPLITS:
        source = resolve(args.source_split_pattern.format(split=split))
        items = [line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(items) != len(set(items)):
            raise ValueError(f"Duplicate entries in {source}")
        split_items[split] = items
        for item in items:
            parts = Path(item).parts
            if len(parts) < 3 or parts[1] not in source_ids:
                raise ValueError(f"Cannot map FabWave class for {item!r}")
            valid_counts[parts[1]] += 1

    removed = [row for row in source_classes if valid_counts[row["class_name"]] < args.min_count]
    kept_classes = [row for row in source_classes if valid_counts[row["class_name"]] >= args.min_count]
    removed_names = {row["class_name"] for row in removed}
    id_remap = {int(row["class_id"]): index for index, row in enumerate(kept_classes)}

    output_cache.mkdir(parents=True, exist_ok=True)
    output_counts: dict[str, int] = {}
    removed_split_counts: dict[str, int] = {}
    all_kept: list[str] = []
    for split, items in split_items.items():
        kept = [item for item in items if Path(item).parts[1] not in removed_names]
        output_counts[split] = len(kept)
        removed_split_counts[split] = len(items) - len(kept)
        all_kept.extend(kept)
        output = resolve(args.output_split_pattern.format(split=split))
        atomic_text(output, "".join(f"{item}\n" for item in kept))

    if len(all_kept) != len(set(all_kept)):
        raise ValueError("Output splits are not mutually disjoint")

    for item in all_kept:
        relative = Path(item)
        step_path = dataset_root / relative
        if not step_path.is_file():
            raise FileNotFoundError(step_path)
        source_path = _cache_path(source_cache, dataset_root, step_path)
        target_path = _cache_path(output_cache, dataset_root, step_path)
        with np.load(source_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        labels = np.asarray(arrays.get("labels"), dtype=np.int64)
        if labels.size != 1:
            raise ValueError(f"Expected one class label in {source_path}, got {labels}")
        source_id = int(labels.item())
        expected_source_id = source_ids[relative.parts[1]]
        if source_id != expected_source_id or source_id not in id_remap:
            raise ValueError(f"Inconsistent source class for {item}: cache={source_id}, expected={expected_source_id}")
        target_id = id_remap[source_id]
        arrays["labels"] = np.asarray([target_id], dtype=np.int64)
        if args.overwrite or not target_path.exists():
            save_graph_npz(target_path, arrays)

        label_path = output_labels / relative.with_suffix(".cls")
        one_hot = [0] * len(kept_classes)
        one_hot[target_id] = 1
        expected_label = " ".join(map(str, one_hot)) + "\n"
        if args.overwrite or not label_path.exists():
            atomic_text(label_path, expected_label)
        elif label_path.read_text(encoding="utf-8") != expected_label:
            raise ValueError(f"Existing output label is inconsistent: {label_path}")

    result = {
        "dataset": "FabWave",
        "derived_from": str(source_map_path),
        "minimum_valid_count": args.min_count,
        "source_valid_samples": sum(valid_counts.values()),
        "output_valid_samples": len(all_kept),
        "removed_classes": [
            {
                "class_name": row["class_name"],
                "source_class_id": row["class_id"],
                "valid_count": valid_counts[row["class_name"]],
            }
            for row in removed
        ],
        "encoding": "one_hot",
        "num_classes": len(kept_classes),
        "classes": [
            {
                "class_id": index,
                "class_name": row["class_name"],
                "valid_count": valid_counts[row["class_name"]],
                "source_class_id": row["class_id"],
            }
            for index, row in enumerate(kept_classes)
        ],
        "split_counts": output_counts,
        "removed_split_counts": removed_split_counts,
    }
    atomic_text(resolve(args.output_class_map), json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
