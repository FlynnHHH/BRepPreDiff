#!/usr/bin/env python3
"""Create deterministic, nested, class-covering supervision-ratio splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from brepprediff.config import load_experiment_config
from brepprediff.data.dataset import StepSegDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--ratios", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-split", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_rank(seed: int, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()


def ratio_tag(ratio: float) -> str:
    return f"{ratio:g}".replace(".", "p")


def class_covering_order(
    sample_ids: list[str], sample_classes: list[set[int]], classes: set[int], seed: int
) -> list[int]:
    """Return one nested order whose shortest prefix greedily covers all classes."""
    ranked = sorted(range(len(sample_ids)), key=lambda i: stable_rank(seed, sample_ids[i]))
    stable_position = {index: position for position, index in enumerate(ranked)}
    remaining = set(ranked)
    missing = set(classes)
    prefix: list[int] = []
    while missing:
        best = max(
            ranked,
            key=lambda i: (len(sample_classes[i] & missing), -stable_position[i]),
        )
        gain = sample_classes[best] & missing
        if best not in remaining or not gain:
            available = set().union(*(sample_classes[i] for i in remaining)) if remaining else set()
            raise ValueError(f"Cannot cover classes {sorted(missing - available)}")
        prefix.append(best)
        remaining.remove(best)
        missing -= gain
        ranked.remove(best)
    return prefix + ranked


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.source_split is not None:
        config["data"]["train_split"] = str(args.source_split)
    dataset = StepSegDataset(config, split="train")
    split_path = Path(config["data"]["train_split"])
    split_items = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(split_items) != len(dataset.samples):
        raise ValueError(f"Split/cache mismatch: {len(split_items)} vs {len(dataset.samples)}")

    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    sample_classes: list[set[int]] = []
    sample_face_counts: list[dict[int, int]] = []
    for sample in dataset.samples:
        with np.load(sample.cache_path, allow_pickle=False) as arrays:
            labels = np.asarray(arrays["labels"], dtype=np.int64).reshape(-1)
        labels = labels[labels != ignore_index]
        values, counts = np.unique(labels, return_counts=True)
        histogram = {int(value): int(count) for value, count in zip(values, counts)}
        sample_face_counts.append(histogram)
        sample_classes.append(set(histogram))

    expected = set(range(int(config["model"]["num_classes"])))
    available = set().union(*sample_classes)
    if expected - available:
        raise ValueError(f"Training split has no samples for classes {sorted(expected - available)}")
    sample_ids = [sample.sample_id for sample in dataset.samples]
    order = class_covering_order(sample_ids, sample_classes, expected, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous: set[int] = set()
    for ratio in sorted(set(args.ratios)):
        if not 0 < ratio <= 100:
            raise ValueError(f"Ratio must be in (0, 100], got {ratio}")
        count = len(split_items) if ratio == 100 else math.ceil(len(split_items) * ratio / 100)
        selected = set(order[:count])
        if not previous <= selected:
            raise AssertionError("Generated supervision splits are not nested")
        previous = selected
        selected_in_source_order = sorted(selected)
        output = args.output_dir / f"{args.task_name}_r{ratio_tag(ratio)}_train.txt"
        output.write_text("".join(f"{split_items[i]}\n" for i in selected_in_source_order))
        model_support = {
            str(class_id): sum(class_id in sample_classes[i] for i in selected)
            for class_id in sorted(expected)
        }
        face_support = {
            str(class_id): sum(sample_face_counts[i].get(class_id, 0) for i in selected)
            for class_id in sorted(expected)
        }
        audit = {
            "task": args.task_name,
            "config": str(args.config),
            "source_split": str(split_path),
            "seed": args.seed,
            "requested_ratio_percent": ratio,
            "actual_ratio_percent": 100 * count / len(split_items),
            "source_samples": len(split_items),
            "selected_samples": count,
            "definition": "Labeled CAD-model ratio; nested class-covering prefix, then stable SHA-256 order",
            "model_support_per_class": model_support,
            "face_support_per_class": face_support,
            "split": str(output),
        }
        output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"task={args.task_name} ratio={ratio:g}% selected={count}/{len(split_items)} split={output}")


if __name__ == "__main__":
    main()
