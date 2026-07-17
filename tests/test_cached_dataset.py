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


def _cached_config(tmp_path: Path, split_path: Path) -> dict:
    config = load_config("data/default.yaml")
    config["data"].update(
        {
            "steps_dir": str(tmp_path / "missing_steps"),
            "segs_dir": str(tmp_path / "missing_segs"),
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(split_path),
        }
    )
    return config


def test_step_and_seg_directories_are_configured_independently(tmp_path: Path):
    steps_dir = tmp_path / "cad_storage" / "models"
    segs_dir = tmp_path / "label_storage" / "annotations"
    steps_dir.mkdir(parents=True)
    segs_dir.mkdir(parents=True)
    step_path = steps_dir / "part.step"
    seg_path = segs_dir / "part.seg"
    step_path.touch()
    seg_path.write_text("0\n", encoding="utf-8")

    config = load_config("data/default.yaml")
    config["data"].update(
        {
            "steps_dir": str(steps_dir),
            "segs_dir": str(segs_dir),
            "cache_dir": str(tmp_path / "cache"),
            "train_split": None,
        }
    )

    dataset = StepSegDataset(config, split="train", source_mode=True)

    assert len(dataset) == 1
    assert dataset.samples[0].step_path == step_path
    assert dataset.samples[0].seg_path == seg_path


def test_cached_train_uses_configured_split(tmp_path: Path):
    cache_path = tmp_path / "cache" / "part_abc123.npz"
    _write_cache(cache_path)
    other_cache_path = tmp_path / "cache" / "other_def456.npz"
    _write_cache(other_cache_path)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part_abc123.npz\n", encoding="utf-8")

    dataset = StepSegDataset(_cached_config(tmp_path, split_path), split="train")

    assert len(dataset) == 1
    assert dataset[0].sample_id == "part_abc123"


def test_cached_val_split_can_reference_original_step_stem(tmp_path: Path):
    cache_path = tmp_path / "cache" / "part_abc123.npz"
    _write_cache(cache_path)
    other_cache_path = tmp_path / "cache" / "other_def456.npz"
    _write_cache(other_cache_path)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part.step\n", encoding="utf-8")
    config = _cached_config(tmp_path, split_path)
    config["data"]["val_split"] = str(split_path)

    dataset = StepSegDataset(config, split="val")
    graph = dataset[0]

    assert len(dataset) == 1
    assert graph.sample_id == "part"
    assert graph.num_faces == 2


def test_cached_dataset_uses_split_specific_cache_dirs(tmp_path: Path):
    train_cache = tmp_path / "cache_split" / "train"
    val_cache = tmp_path / "cache_split" / "val"
    _write_cache(train_cache / "train_abc123.npz")
    _write_cache(val_cache / "val_def456.npz")
    _write_cache(tmp_path / "cache" / "ignored_xyz789.npz")
    train_split_path = tmp_path / "train_split.txt"
    train_split_path.write_text("train.step\n", encoding="utf-8")
    split_path = tmp_path / "val_split.txt"
    split_path.write_text("val.step\n", encoding="utf-8")

    config = _cached_config(tmp_path, train_split_path)
    config["data"]["cache_dirs"] = {
        "train": str(train_cache),
        "val": str(val_cache),
    }
    config["data"]["val_split"] = str(split_path)

    train_dataset = StepSegDataset(config, split="train")
    val_dataset = StepSegDataset(config, split="val")

    assert len(train_dataset) == 1
    assert train_dataset[0].sample_id == "train"
    assert len(val_dataset) == 1
    assert val_dataset[0].sample_id == "val"
