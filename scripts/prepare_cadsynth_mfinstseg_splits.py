#!/usr/bin/env python3
"""Create reproducible BRepPreDiff splits for CADSynth and MFInstSeg."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    temporary.replace(path)


def prepare_cadsynth(root: Path, output: Path) -> dict[str, int]:
    all_ids = {path.stem for path in (root / "step").glob("*.stp")}
    train = _read_ids(root / "train.txt")
    test = _read_ids(root / "test.txt")
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("CADSynth official split files contain duplicate IDs.")
    train_set, test_set = set(train), set(test)
    if train_set & test_set:
        raise ValueError("CADSynth official train and test splits overlap.")
    if not train_set | test_set <= all_ids:
        raise ValueError("CADSynth official splits reference missing STEP files.")
    val = sorted(all_ids - train_set - test_set)
    if not val:
        raise ValueError("CADSynth has no samples left for validation.")

    splits = {"train": train, "val": val, "test": test}
    for name, values in splits.items():
        _write(output / f"cadsynth_{name}.txt", values)
    return {name: len(values) for name, values in splits.items()}


def prepare_mfinstseg(root: Path, output: Path, seed: int) -> dict[str, int]:
    step_ids = {path.stem for path in (root / "steps").glob("*.step")}
    label_ids = {path.stem for path in (root / "labels").glob("*.json")}
    if step_ids != label_ids:
        raise ValueError(
            "MFInstSeg STEP/label IDs differ: "
            f"missing_labels={len(step_ids - label_ids)}, missing_steps={len(label_ids - step_ids)}"
        )
    values = sorted(step_ids)
    random.Random(seed).shuffle(values)
    train_end = int(len(values) * 0.8)
    val_end = train_end + int(len(values) * 0.1)
    splits = {
        "train": sorted(values[:train_end]),
        "val": sorted(values[train_end:val_end]),
        "test": sorted(values[val_end:]),
    }
    for name, split_values in splits.items():
        _write(output / f"mfinstseg_{name}.txt", split_values)
    return {name: len(split_values) for name, split_values in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cadsynth-root", type=Path, default=Path("/home/nvme03/hhfeng/CADSynth"))
    parser.add_argument("--mfinstseg-root", type=Path, default=Path("/home/nvme03/hhfeng/MFInstSeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cad_counts = prepare_cadsynth(args.cadsynth_root, args.output_dir)
    mf_counts = prepare_mfinstseg(args.mfinstseg_root, args.output_dir, args.seed)
    print(f"CADSynth splits: {cad_counts}")
    print(f"MFInstSeg splits: {mf_counts}")


if __name__ == "__main__":
    main()
