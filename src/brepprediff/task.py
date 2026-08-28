from __future__ import annotations

from typing import Any


SEGMENTATION = "seg"
CLASSIFICATION = "cls"


def task_type(config: dict[str, Any]) -> str:
    """Return the configured downstream task, defaulting to face segmentation."""
    configured = config.get("task", SEGMENTATION)
    if isinstance(configured, dict):
        configured = configured.get("type", SEGMENTATION)
    value = str(configured).strip().lower()
    aliases = {
        "seg": SEGMENTATION,
        "segmentation": SEGMENTATION,
        "cls": CLASSIFICATION,
        "classification": CLASSIFICATION,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported downstream task {configured!r}; expected one of {sorted(aliases)}."
        ) from exc


def task_label(config: dict[str, Any]) -> str:
    return "face segmentation" if task_type(config) == SEGMENTATION else "model classification"
