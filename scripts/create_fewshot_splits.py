#!/usr/bin/env python3
"""Create deterministic, nested K-shot training splits from cached labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from brepprediff.config import load_experiment_config
from brepprediff.data.dataset import StepSegDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_rank(seed: int, class_id: int, sample_id: str) -> bytes:
    payload = f"{seed}:{class_id}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dataset = StepSegDataset(config, split="train")
    split_path = Path(config["data"]["train_split"])
    split_items = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(split_items) != len(dataset.samples):
        raise ValueError(
            f"Split/cache mismatch: {len(split_items)} split items vs {len(dataset.samples)} samples"
        )

    sample_classes: list[set[int]] = []
    candidates: dict[int, list[int]] = {}
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    for index, sample in enumerate(dataset.samples):
        with np.load(sample.cache_path, allow_pickle=False) as arrays:
            labels = np.asarray(arrays["labels"], dtype=np.int64).reshape(-1)
        classes = {int(value) for value in labels.tolist() if int(value) != ignore_index}
        if not classes:
            raise ValueError(f"Sample has no usable labels: {sample.sample_id}")
        sample_classes.append(classes)
        for class_id in classes:
            candidates.setdefault(class_id, []).append(index)

    expected_classes = set(range(int(config["model"]["num_classes"])))
    missing = sorted(expected_classes - candidates.keys())
    if missing:
        raise ValueError(f"Training split has no samples for classes: {missing}")

    ranked: dict[int, list[int]] = {}
    for class_id in sorted(expected_classes):
        ranked[class_id] = sorted(
            candidates[class_id],
            key=lambda index: stable_rank(args.seed, class_id, dataset.samples[index].sample_id),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shot in sorted(set(args.shots)):
        shortages = {
            class_id: len(indices)
            for class_id, indices in ranked.items()
            if len(indices) < shot
        }
        if shortages:
            raise ValueError(f"Cannot create {shot}-shot split; class availability: {shortages}")

        selected_by_class = {
            class_id: indices[:shot] for class_id, indices in ranked.items()
        }
        selected = sorted({index for indices in selected_by_class.values() for index in indices})
        output_path = args.output_dir / f"{args.task_name}_{shot}shot_train.txt"
        output_path.write_text(
            "".join(f"{split_items[index]}\n" for index in selected),
            encoding="utf-8",
        )
        actual_support = {
            class_id: sum(class_id in sample_classes[index] for index in selected)
            for class_id in sorted(expected_classes)
        }
        audit = {
            "task": args.task_name,
            "config": str(args.config),
            "source_split": str(split_path),
            "seed": args.seed,
            "shot": shot,
            "definition": "K cached CAD models selected per semantic class; union used for segmentation",
            "source_samples": len(dataset.samples),
            "selected_unique_samples": len(selected),
            "available_per_class": {
                str(class_id): len(ranked[class_id]) for class_id in sorted(expected_classes)
            },
            "selected_per_class": {
                str(class_id): shot for class_id in sorted(expected_classes)
            },
            "actual_support_per_class_after_union": {
                str(class_id): actual_support[class_id] for class_id in sorted(expected_classes)
            },
            "split": str(output_path),
        }
        audit_path = output_path.with_suffix(".audit.json")
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print(
            f"task={args.task_name} shot={shot} selected={len(selected)} "
            f"split={output_path}"
        )


if __name__ == "__main__":
    main()
