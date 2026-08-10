from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from blendit.data.tmcad_cls import _allocate_counts


STEP_EXTENSIONS = {".step", ".stp"}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class FabWavePreparationResult:
    class_names: tuple[str, ...]
    model_count: int
    written_labels: int
    split_counts: dict[str, int]
    empty_categories: tuple[str, ...]


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


def discover_fabwave(
    dataset_root: str | Path,
) -> tuple[dict[str, list[Path]], tuple[str, ...]]:
    """Discover classes stored one level below FabWave's three numbered groups."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"FabWave dataset root does not exist: {root}")

    categories: dict[str, list[Path]] = {}
    empty: list[str] = []
    def group_order(path: Path) -> tuple[int, str]:
        numbers = re.findall(r"\d+", path.name)
        return (int(numbers[0]) if numbers else 10**9, path.name.casefold())

    group_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=group_order)
    for group_dir in group_dirs:
        for category_dir in sorted(
            (path for path in group_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        ):
            class_name = category_dir.name
            files = _step_files(category_dir)
            if not files:
                empty.append(f"{group_dir.name}/{class_name}")
                continue
            if class_name in categories:
                raise ValueError(
                    f"Duplicate FabWave class name {class_name!r}; category names must be unique."
                )
            categories[class_name] = files
    if not categories:
        raise ValueError(f"No category directories containing STEP/STP models found under {root}.")
    return categories, tuple(empty)


def prepare_fabwave_classification(
    dataset_root: str | Path,
    *,
    labels_root: str | Path,
    splits_dir: str | Path,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    overwrite: bool = False,
) -> FabWavePreparationResult:
    root = Path(dataset_root).expanduser().resolve()
    labels = Path(labels_root).expanduser().resolve()
    split_root = Path(splits_dir).expanduser().resolve()
    categories, empty_categories = discover_fabwave(root)
    class_names = tuple(categories)
    split_items: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    written_labels = 0

    for class_id, (class_name, model_paths) in enumerate(categories.items()):
        one_hot = [0] * len(class_names)
        one_hot[class_id] = 1
        expected = " ".join(str(value) for value in one_hot) + "\n"
        for step_path in model_paths:
            relative = step_path.relative_to(root)
            cls_path = labels / relative.with_suffix(".cls")
            if cls_path.exists() and not overwrite:
                if cls_path.read_text(encoding="utf-8") != expected:
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
        _atomic_write(split_root / f"fabwave_{split_name}.txt", "".join(f"{item}\n" for item in items))
        split_counts[split_name] = len(items)

    class_map = {
        "dataset": "FabWave",
        "encoding": "one_hot",
        "num_classes": len(class_names),
        "empty_categories_excluded": list(empty_categories),
        "classes": [
            {
                "class_id": index,
                "class_name": name,
                "model_count": len(categories[name]),
            }
            for index, name in enumerate(class_names)
        ],
    }
    _atomic_write(split_root / "fabwave_class_map.json", json.dumps(class_map, indent=2) + "\n")
    return FabWavePreparationResult(
        class_names=class_names,
        model_count=sum(len(paths) for paths in categories.values()),
        written_labels=written_labels,
        split_counts=split_counts,
        empty_categories=empty_categories,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one-hot CLS labels and stratified splits for FabWave."
    )
    parser.add_argument("--dataset-root", default="/data/hhfeng/FabWave")
    parser.add_argument("--labels-root", default="data/labels/fabwave")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = prepare_fabwave_classification(
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
                "empty_categories": list(result.empty_categories),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
