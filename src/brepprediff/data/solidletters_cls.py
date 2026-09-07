from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from brepprediff.data.tmcad_cls import _allocate_counts


CLASS_NAMES = tuple(chr(code) for code in range(ord("a"), ord("z") + 1))
STEP_EXTENSIONS = (".step", ".stp")
_SAMPLE_NAME = re.compile(r"^(?P<label>[A-Za-z])_(?P<variant>.+)$")


@dataclass(frozen=True)
class SolidLettersPreparationResult:
    class_names: tuple[str, ...]
    model_count: int
    written_labels: int
    split_counts: dict[str, int]
    unlisted_models: tuple[str, ...]


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


def _discover_steps(steps_dir: Path) -> dict[str, Path]:
    if not steps_dir.is_dir():
        raise FileNotFoundError(f"SolidLetters STEP directory does not exist: {steps_dir}")
    models: dict[str, Path] = {}
    for path in sorted(steps_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in STEP_EXTENSIONS:
            continue
        match = _SAMPLE_NAME.fullmatch(path.stem)
        if match is None:
            raise ValueError(
                f"SolidLetters model name must start with a single letter and underscore: {path.name}"
            )
        label = match.group("label").lower()
        if label not in CLASS_NAMES:
            raise ValueError(f"Unsupported SolidLetters label prefix {label!r}: {path.name}")
        if path.stem in models:
            raise ValueError(f"Duplicate SolidLetters model stem: {path.stem}")
        models[path.stem] = path
    if not models:
        raise ValueError(f"No STEP/STP models found under {steps_dir}.")
    return models


def _read_official_split(path: Path, models: dict[str, Path]) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SolidLetters split file does not exist: {path}")
    items: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        stem = Path(item).stem
        if stem not in models:
            raise FileNotFoundError(f"Unknown model in {path}:{line_number}: {item!r}")
        if stem in seen:
            raise ValueError(f"Duplicate model in {path}:{line_number}: {item!r}")
        seen.add(stem)
        items.append(stem)
    return items


def _class_name(stem: str) -> str:
    match = _SAMPLE_NAME.fullmatch(stem)
    assert match is not None
    return match.group("label").lower()


def prepare_solidletters_classification(
    dataset_root: str | Path,
    *,
    labels_root: str | Path,
    splits_dir: str | Path,
    validation_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> SolidLettersPreparationResult:
    """Create one-hot labels and train/val/test splits from SolidLetters metadata.

    The dataset's official test split is preserved. Validation is sampled per
    letter from the official training split, leaving the remainder for training.
    """
    root = Path(dataset_root).expanduser().resolve()
    steps_dir = root / "step"
    labels = Path(labels_root).expanduser().resolve()
    split_root = Path(splits_dir).expanduser().resolve()
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1).")

    models = _discover_steps(steps_dir)
    official_train = _read_official_split(root / "train.txt", models)
    official_test = _read_official_split(root / "test.txt", models)
    overlap = set(official_train) & set(official_test)
    if overlap:
        raise ValueError(f"Official SolidLetters train/test splits overlap ({len(overlap)} models).")

    by_class: dict[str, list[str]] = {name: [] for name in CLASS_NAMES}
    for stem in official_train:
        by_class[_class_name(stem)].append(stem)

    split_items: dict[str, list[str]] = {"train": [], "val": [], "test": official_test}
    for class_name, items in by_class.items():
        shuffled = list(items)
        random.Random(f"{seed}:{class_name}").shuffle(shuffled)
        train_count, val_count, _ = _allocate_counts(
            len(shuffled), (1.0 - validation_ratio, validation_ratio, 0.0)
        )
        split_items["train"].extend(shuffled[:train_count])
        split_items["val"].extend(shuffled[train_count : train_count + val_count])

    written_labels = 0
    for stem, step_path in models.items():
        class_id = ord(_class_name(stem)) - ord("a")
        one_hot = [0] * len(CLASS_NAMES)
        one_hot[class_id] = 1
        expected = " ".join(str(value) for value in one_hot) + "\n"
        cls_path = labels / step_path.relative_to(steps_dir).with_suffix(".cls")
        if cls_path.exists() and not overwrite:
            if cls_path.read_text(encoding="utf-8") != expected:
                raise ValueError(
                    f"Existing CLS label differs from the filename prefix: {cls_path}; "
                    "pass --overwrite to replace it."
                )
        else:
            _atomic_write(cls_path, expected)
            written_labels += 1

    split_counts: dict[str, int] = {}
    for split_name, stems in split_items.items():
        paths = sorted(models[stem].name for stem in stems)
        _atomic_write(
            split_root / f"solidletters_{split_name}.txt",
            "".join(f"{item}\n" for item in paths),
        )
        split_counts[split_name] = len(paths)

    listed = set(official_train) | set(official_test)
    unlisted = tuple(sorted(models.keys() - listed))
    class_map = {
        "dataset": "SolidLetters",
        "encoding": "one_hot",
        "num_classes": len(CLASS_NAMES),
        "source_train_split": str(root / "train.txt"),
        "source_test_split": str(root / "test.txt"),
        "validation_ratio_from_official_train": validation_ratio,
        "seed": seed,
        "unlisted_models_excluded": list(unlisted),
        "classes": [
            {
                "class_id": class_id,
                "class_name": class_name,
                "model_count": sum(_class_name(stem) == class_name for stem in models),
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
    }
    _atomic_write(
        split_root / "solidletters_class_map.json",
        json.dumps(class_map, indent=2) + "\n",
    )
    return SolidLettersPreparationResult(
        class_names=CLASS_NAMES,
        model_count=len(models),
        written_labels=written_labels,
        split_counts=split_counts,
        unlisted_models=unlisted,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one-hot CLS labels and BRepPreDiff splits for SolidLetters."
    )
    parser.add_argument("--dataset-root", default="/home/nvme03/hhfeng/SolidLetters")
    parser.add_argument("--labels-root", default="data/labels/solidletters")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = prepare_solidletters_classification(
        args.dataset_root,
        labels_root=args.labels_root,
        splits_dir=args.splits_dir,
        validation_ratio=args.validation_ratio,
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
                "unlisted_models": list(result.unlisted_models),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
