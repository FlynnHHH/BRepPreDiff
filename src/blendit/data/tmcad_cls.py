from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


STEP_EXTENSIONS = {".step", ".stp"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class TMCADPreparationResult:
    class_names: tuple[str, ...]
    model_count: int
    written_labels: int
    split_counts: dict[str, int]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _step_files(category_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in category_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in STEP_EXTENSIONS
    )


def discover_tmcad(dataset_root: str | Path) -> dict[str, list[Path]]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TMCAD dataset root does not exist: {root}")
    categories: dict[str, list[Path]] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        files = _step_files(directory)
        if files:
            categories[directory.name] = files
    if not categories:
        raise ValueError(f"No category directories containing STEP/STP models found under {root}.")
    return categories


def _allocate_counts(total: int, ratios: Sequence[float]) -> list[int]:
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError(f"Expected {len(SPLIT_NAMES)} split ratios, got {len(ratios)}.")
    weights = [max(0.0, float(value)) for value in ratios]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    raw = [total * value / weight_sum for value in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def prepare_tmcad_classification(
    dataset_root: str | Path,
    *,
    labels_root: str | Path | None = None,
    splits_dir: str | Path | None = None,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    overwrite: bool = False,
) -> TMCADPreparationResult:
    root = Path(dataset_root).expanduser().resolve()
    labels = Path(labels_root).expanduser().resolve() if labels_root else root
    split_root = (
        Path(splits_dir).expanduser().resolve()
        if splits_dir
        else root / "blendit_splits"
    )
    categories = discover_tmcad(root)
    class_names = tuple(categories)
    split_items: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    written_labels = 0

    for class_id, class_name in enumerate(class_names):
        one_hot = [0] * len(class_names)
        one_hot[class_id] = 1
        expected = " ".join(str(value) for value in one_hot) + "\n"
        model_paths = categories[class_name]
        for step_path in model_paths:
            relative = step_path.relative_to(root)
            cls_path = labels / relative.with_suffix(".cls")
            if cls_path.exists() and not overwrite:
                actual = cls_path.read_text(encoding="utf-8")
                if actual != expected:
                    raise ValueError(
                        f"Existing CLS label differs from the category mapping: {cls_path}; "
                        "pass --overwrite to replace it."
                    )
            else:
                _atomic_write(cls_path, expected)
                written_labels += 1

        shuffled = list(model_paths)
        random.Random(f"{seed}:{class_name}").shuffle(shuffled)
        counts = _allocate_counts(len(shuffled), ratios)
        offset = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            selected = shuffled[offset : offset + count]
            offset += count
            split_items[split_name].extend(
                path.relative_to(root).as_posix() for path in selected
            )

    split_counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        items = sorted(split_items[split_name])
        _atomic_write(split_root / f"{split_name}.txt", "".join(f"{item}\n" for item in items))
        split_counts[split_name] = len(items)

    class_map = {
        "dataset": "TMCAD",
        "encoding": "one_hot",
        "num_classes": len(class_names),
        "classes": [
            {"class_id": index, "class_name": name}
            for index, name in enumerate(class_names)
        ],
    }
    _atomic_write(split_root / "class_map.json", json.dumps(class_map, indent=2) + "\n")
    return TMCADPreparationResult(
        class_names=class_names,
        model_count=sum(len(paths) for paths in categories.values()),
        written_labels=written_labels,
        split_counts=split_counts,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one-hot CLS labels and stratified Blendit splits for TMCAD."
    )
    parser.add_argument("--dataset-root", default="/data/hhfeng/TMCAD")
    parser.add_argument(
        "--labels-root",
        default=None,
        help="Optional mirror root for CLS files; defaults to writing beside each STEP model.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Defaults to <dataset-root>/blendit_splits.",
    )
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = prepare_tmcad_classification(
        args.dataset_root,
        labels_root=args.labels_root,
        splits_dir=args.splits_dir,
        ratios=args.ratios,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "classes": list(result.class_names),
                "model_count": result.model_count,
                "written_labels": result.written_labels,
                "split_counts": result.split_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
