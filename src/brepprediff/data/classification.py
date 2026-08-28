from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CLASS_LABEL_EXTENSIONS = (".cls",)


def match_classification_label(
    labels_dir: Path,
    steps_dir: Path,
    step_path: Path,
) -> Path | None:
    relative_path = step_path.relative_to(steps_dir)
    candidates = (
        labels_dir / relative_path.with_suffix(".cls"),
        labels_dir / f"{step_path.stem}.cls",
    )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def read_class_label(path: str | Path, config: dict[str, Any]) -> np.ndarray:
    """Read and validate a one-hot CLS file, returning one model-level class id."""
    label_path = Path(path)
    values: list[float] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.extend(float(value) for value in line.replace(",", " ").split())

    expected = config.get("labels", {}).get("num_classes")
    if expected is None:
        expected = config.get("model", {}).get("num_classes")
    if expected is not None and len(values) != int(expected):
        raise ValueError(
            f"Classification label length mismatch for {label_path}: got {len(values)}, "
            f"expected {int(expected)}."
        )
    if not values:
        raise ValueError(f"Classification label is empty: {label_path}")
    vector = np.asarray(values, dtype=np.float32)
    if not np.isin(vector, np.asarray([0.0, 1.0], dtype=np.float32)).all():
        raise ValueError(f"Classification label must be a binary one-hot vector: {label_path}")
    active = np.flatnonzero(vector == 1.0)
    if active.size != 1:
        raise ValueError(
            f"Classification label must contain exactly one active class: {label_path}"
        )
    return np.asarray([int(active[0])], dtype=np.int64)


def extract_classification_arrays(
    extractor: Any,
    step_path: Path,
    label_path: Path | None,
    config: dict[str, Any],
    *,
    labels_required: bool,
) -> dict[str, np.ndarray]:
    arrays = extractor.extract(
        step_path,
        None,
        labels_required=False,
        strict_label_count=False,
    )
    if label_path is None:
        if labels_required:
            raise FileNotFoundError(
                f"Classification labels are required but no CLS path was found for {step_path}."
            )
    else:
        arrays["labels"] = read_class_label(label_path, config)
    return arrays


__all__ = [
    "CLASS_LABEL_EXTENSIONS",
    "extract_classification_arrays",
    "match_classification_label",
    "read_class_label",
]
