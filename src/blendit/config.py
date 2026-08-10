from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating the input."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def load_experiment_config(
    training_path: str | Path,
    data_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Merge a training-only config with its prepare-data config."""
    training_config_path = Path(training_path)
    training_config = load_config(training_config_path)
    configured_data_path = data_path or training_config.get("data_config")
    if not configured_data_path:
        raise ValueError(
            f"Training config {training_path} must set data_config or be used with --data-config."
        )
    data_config_path = Path(configured_data_path)
    if data_path is None and not data_config_path.is_absolute():
        data_config_path = training_config_path.parent / data_config_path
    data_config = load_config(data_config_path)
    config = deep_update(data_config, training_config)
    config["data_config"] = str(data_config_path)
    return apply_overrides(config, overrides)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, (list, dict)):
            return parsed
    return raw


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    cfg = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value.")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        cursor = cfg
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_scalar(raw_value)
    return cfg


def feature_dims(config: dict[str, Any]) -> tuple[int, int]:
    """Return face and edge continuous feature dimensions."""
    grid = int(config["brep"]["uv_grid_size"])
    face_dim = 11 + grid * grid * 6
    edge_dim = 3
    return face_dim, edge_dim
