#!/usr/bin/env python3
"""Build a remapped FabWave classification dataset with one class removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blendit.data.dataset import _cache_path
from blendit.data.graph import save_graph_npz


ROOT = Path(__file__).resolve().parents[1]
STEP_EXTENSIONS = (".step", ".stp", ".STEP", ".STP")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-class", default="Rotary_Shaft")
    parser.add_argument("--dataset-root", default="/data/hhfeng/FabWave")
    parser.add_argument("--source-labels", default="data/labels/fabwave")
    parser.add_argument("--source-cache", default="/data/hhfeng/blendit/cache/features/fabwave")
    parser.add_argument("--source-class-map", default="data/splits/fabwave_class_map.json")
    parser.add_argument("--source-split-pattern", default="data/splits/fabwave_{split}_clean.txt")
    parser.add_argument("--output-labels", default="data/labels/fabwave_no_rotary_shaft")
    parser.add_argument("--output-cache", default="/data/hhfeng/blendit/cache/features/fabwave_no_rotary_shaft")
    parser.add_argument("--output-class-map", default="data/splits/fabwave_no_rotary_shaft_class_map.json")
    parser.add_argument("--output-split-pattern", default="data/splits/fabwave_no_rotary_shaft_{split}.txt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def resolve_step(dataset_root: Path, item: str) -> Path:
    direct = dataset_root / item
    if direct.is_file():
        return direct
    if Path(item).suffix:
        raise FileNotFoundError(direct)
    for extension in STEP_EXTENSIONS:
        candidate = dataset_root / f"{item}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No STEP file found for split item {item!r}")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    dataset_root = resolve(args.dataset_root)
    source_cache = resolve(args.source_cache)
    output_cache = resolve(args.output_cache)
    output_labels = resolve(args.output_labels)
    class_map_path = resolve(args.source_class_map)
    source_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    source_classes = source_map["classes"]
    source_names = [row["class_name"] for row in source_classes]
    if args.remove_class not in source_names:
        raise ValueError(f"Unknown class {args.remove_class!r}; choices={source_names}")
    removed_id = source_names.index(args.remove_class)
    target_classes = [row for row in source_classes if row["class_name"] != args.remove_class]
    target_names = [row["class_name"] for row in target_classes]

    split_counts: dict[str, int] = {}
    removed_counts: dict[str, int] = {}
    all_items: list[str] = []
    for split in SPLITS:
        source_split = resolve(args.source_split_pattern.format(split=split))
        items = [
            line.strip()
            for line in source_split.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        kept = [item for item in items if args.remove_class not in Path(item).parts]
        removed_counts[split] = len(items) - len(kept)
        split_counts[split] = len(kept)
        all_items.extend(kept)
        output_split = resolve(args.output_split_pattern.format(split=split))
        atomic_text(output_split, "".join(f"{item}\n" for item in kept))

    if len(all_items) != len(set(all_items)):
        raise ValueError("Derived splits contain duplicate samples")

    output_cache.mkdir(parents=True, exist_ok=True)
    for item in all_items:
        step_path = resolve_step(dataset_root, item)
        source_path = _cache_path(source_cache, dataset_root, step_path)
        target_path = _cache_path(output_cache, dataset_root, step_path)
        relative = step_path.relative_to(dataset_root)
        label_path = output_labels / relative.with_suffix(".cls")
        with np.load(source_path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        labels = np.asarray(arrays.get("labels"), dtype=np.int64)
        if labels.size != 1:
            raise ValueError(f"Expected one cached class label in {source_path}, got {labels}")
        source_id = int(labels.item())
        if source_id == removed_id:
            raise ValueError(f"Removed class leaked into derived split: {item}")
        target_id = source_id - 1 if source_id > removed_id else source_id
        arrays["labels"] = np.asarray([target_id], dtype=np.int64)
        if args.overwrite or not target_path.exists():
            save_graph_npz(target_path, arrays)
        one_hot = [0] * len(target_names)
        one_hot[target_id] = 1
        expected_label = " ".join(map(str, one_hot)) + "\n"
        if args.overwrite or not label_path.exists():
            atomic_text(label_path, expected_label)
        elif label_path.read_text(encoding="utf-8") != expected_label:
            raise ValueError(f"Existing derived label is inconsistent: {label_path}")

    derived_map = {
        "dataset": "FabWave",
        "derived_from": str(class_map_path),
        "removed_class": {"class_id": removed_id, "class_name": args.remove_class},
        "encoding": "one_hot",
        "num_classes": len(target_names),
        "classes": [
            {
                "class_id": index,
                "class_name": row["class_name"],
                "model_count": row["model_count"],
                "source_class_id": row["class_id"],
            }
            for index, row in enumerate(target_classes)
        ],
        "split_counts": split_counts,
        "removed_split_counts": removed_counts,
    }
    atomic_text(resolve(args.output_class_map), json.dumps(derived_map, indent=2) + "\n")
    print(json.dumps(derived_map, indent=2))


if __name__ == "__main__":
    main()
