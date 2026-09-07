from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from brepprediff.config import load_config
from brepprediff.data.dataset import StepSegDataset, build_dataloader
from brepprediff.data.graph import save_graph_npz


def _write_cache(path: Path) -> None:
    save_graph_npz(
        path,
        {
            "face_cont": np.zeros((2, 123), dtype=np.float32),
            "face_surface_type": np.zeros((2,), dtype=np.int64),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_cont": np.empty((0, 27), dtype=np.float32),
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


def test_explicitly_unlabeled_source_ignores_co_located_json(tmp_path: Path):
    steps_dir = tmp_path / "fusion_gallery"
    steps_dir.mkdir()
    step_path = steps_dir / "part.step"
    step_path.touch()
    # Fusion360Rec/Fusion360Ass JSON is task metadata, not a face-label list.
    (steps_dir / "part.json").write_text('{"metadata": {}}', encoding="utf-8")

    config = load_config("data/default.yaml")
    config["data"].update(
        {
            "steps_dir": str(steps_dir),
            "segs_dir": str(steps_dir),
            "cache_dir": str(tmp_path / "cache"),
            "train_split": None,
            "labels_required": False,
        }
    )

    dataset = StepSegDataset(config, split="train", source_mode=True)

    assert len(dataset) == 1
    assert dataset.samples[0].step_path == step_path
    assert dataset.samples[0].seg_path is None


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


def test_cache_only_split_indexes_hashed_names_once(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "cache" / "part_with_underscores_abc123.npz"
    _write_cache(cache_path)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part_with_underscores.step\n", encoding="utf-8")
    config = _cached_config(tmp_path, split_path)

    original_rglob = Path.rglob
    calls = 0

    def count_rglob(path, pattern):
        nonlocal calls
        calls += 1
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", count_rglob)
    dataset = StepSegDataset(config, split="train")

    assert calls == 1
    assert dataset[0].sample_id == "part_with_underscores"


def test_cache_only_split_uses_relative_path_hash_to_disambiguate_stems(tmp_path: Path):
    import hashlib

    cache_dir = tmp_path / "cache"
    first_item = "bearing/1.stp"
    second_item = "bolt/1.stp"
    first_digest = hashlib.sha1(first_item.encode("utf-8")).hexdigest()[:10]
    second_digest = hashlib.sha1(second_item.encode("utf-8")).hexdigest()[:10]
    first_cache = cache_dir / f"1_{first_digest}.npz"
    _write_cache(first_cache)
    _write_cache(cache_dir / f"1_{second_digest}.npz")
    split_path = tmp_path / "split.txt"
    split_path.write_text(f"{first_item}\n", encoding="utf-8")

    dataset = StepSegDataset(_cached_config(tmp_path, split_path), split="train")

    assert len(dataset) == 1
    assert dataset.samples[0].cache_path == first_cache


def test_cached_dataset_rejects_legacy_feature_dimensions(tmp_path: Path):
    cache_path = tmp_path / "cache" / "part_abc123.npz"
    _write_cache(cache_path)
    with np.load(cache_path) as current:
        arrays = {key: current[key] for key in current.files}
    arrays["face_cont"] = np.zeros((2, 107), dtype=np.float32)
    arrays["edge_cont"] = np.empty((0, 3), dtype=np.float32)
    save_graph_npz(cache_path, arrays)
    split_path = tmp_path / "split.txt"
    split_path.write_text("part_abc123.npz\n", encoding="utf-8")

    dataset = StepSegDataset(_cached_config(tmp_path, split_path), split="train")

    with pytest.raises(ValueError, match="Rebuild the feature cache"):
        dataset[0]


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


def test_dataloader_seed_is_independent_from_model_rng(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_names = [f"part_{index}.npz" for index in range(8)]
    for cache_name in cache_names:
        _write_cache(cache_dir / cache_name)
    split_path = tmp_path / "split.txt"
    split_path.write_text("\n".join(cache_names) + "\n", encoding="utf-8")
    config = _cached_config(tmp_path, split_path)
    config["train"] = {
        "batch_size": 2,
        "num_workers": 0,
        "dataloader_seed": 42,
    }

    torch.manual_seed(7)
    first_order = [
        sample_id
        for batch in build_dataloader(config, split="train", shuffle=True)
        for sample_id in batch.sample_ids
    ]
    torch.manual_seed(999)
    torch.rand(1000)
    second_order = [
        sample_id
        for batch in build_dataloader(config, split="train", shuffle=True)
        for sample_id in batch.sample_ids
    ]

    assert first_order == second_order
