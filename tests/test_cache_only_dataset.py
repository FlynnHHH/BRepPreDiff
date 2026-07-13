from __future__ import annotations

from pathlib import Path

import numpy as np

from blendit.config import load_config
from blendit.data.dataset import StepSegDataset
from blendit.data.graph import save_graph_npz


def _write_cache(path: Path) -> None:
    save_graph_npz(
        path,
        {
            "face_cont": np.zeros((2, 3), dtype=np.float32),
            "face_surface_type": np.zeros((2,), dtype=np.int64),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_cont": np.empty((0, 1), dtype=np.float32),
            "edge_type": np.empty((0,), dtype=np.int64),
            "edge_relation": np.empty((0,), dtype=np.int64),
            "labels": np.zeros((2,), dtype=np.int64),
        },
    )


def _cache_only_config(tmp_path: Path, split_path: Path) -> dict:
    config = load_config("configs/default.yaml")
    config["data"].update(
        {
            "root": str(tmp_path / "missing_root"),
            "steps_dir": "missing_steps",
            "segs_dir": "missing_segs",
            "cache_dir": str(tmp_path / "cache"),
            "cache_only": True,
            "train_split": str(split_path),
        }
    )
    return config


def test_cache_only_train_uses_all_cache_files_and_ignores_split(tmp_path: Path):
    cache_path = tmp_path / "cache" / "part_abc123.npz"
    _write_cache(cache_path)
    other_cache_path = tmp_path / "cache" / "other_def456.npz"
    _write_cache(other_cache_path)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part_abc123.npz\n", encoding="utf-8")

    dataset = StepSegDataset(_cache_only_config(tmp_path, split_path), split="train")

    assert len(dataset) == 2
    assert {dataset[index].sample_id for index in range(len(dataset))} == {
        "part_abc123",
        "other_def456",
    }


def test_cache_only_val_split_can_reference_original_step_stem(tmp_path: Path):
    cache_path = tmp_path / "cache" / "part_abc123.npz"
    _write_cache(cache_path)
    other_cache_path = tmp_path / "cache" / "other_def456.npz"
    _write_cache(other_cache_path)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part.step\n", encoding="utf-8")
    config = _cache_only_config(tmp_path, split_path)
    config["data"]["val_split"] = str(split_path)

    dataset = StepSegDataset(config, split="val")
    graph = dataset[0]

    assert len(dataset) == 1
    assert graph.sample_id == "part"
    assert graph.num_faces == 2


def test_cache_only_uses_split_specific_cache_dirs(tmp_path: Path):
    train_cache = tmp_path / "cache_split" / "train"
    val_cache = tmp_path / "cache_split" / "val"
    _write_cache(train_cache / "train_abc123.npz")
    _write_cache(val_cache / "val_def456.npz")
    _write_cache(tmp_path / "cache" / "ignored_xyz789.npz")
    split_path = tmp_path / "val_split.txt"
    split_path.write_text("val.step\n", encoding="utf-8")

    config = _cache_only_config(tmp_path, split_path)
    config["data"]["cache_dirs"] = {
        "train": str(train_cache),
        "val": str(val_cache),
    }
    config["data"]["val_split"] = str(split_path)

    train_dataset = StepSegDataset(config, split="train")
    val_dataset = StepSegDataset(config, split="val")

    assert len(train_dataset) == 1
    assert train_dataset[0].sample_id == "train_abc123"
    assert len(val_dataset) == 1
    assert val_dataset[0].sample_id == "val"
