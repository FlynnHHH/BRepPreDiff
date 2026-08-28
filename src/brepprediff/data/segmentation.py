from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


SEGMENTATION_LABEL_EXTENSIONS = (".seg", ".json")


def match_segmentation_label(
    labels_dir: Path,
    steps_dir: Path,
    step_path: Path,
) -> Path | None:
    relative_path = step_path.relative_to(steps_dir)
    candidates = [
        candidate
        for extension in SEGMENTATION_LABEL_EXTENSIONS
        for candidate in (
            labels_dir / relative_path.with_suffix(extension),
            labels_dir / f"{step_path.stem}{extension}",
        )
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def extract_segmentation_arrays(
    extractor: Any,
    step_path: Path,
    label_path: Path | None,
    *,
    labels_required: bool,
    strict_label_count: bool,
) -> dict[str, np.ndarray]:
    return extractor.extract(
        step_path,
        label_path,
        labels_required=labels_required,
        strict_label_count=strict_label_count,
    )


__all__ = [
    "SEGMENTATION_LABEL_EXTENSIONS",
    "extract_segmentation_arrays",
    "match_segmentation_label",
]
