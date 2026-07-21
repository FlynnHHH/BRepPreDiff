from __future__ import annotations

from pathlib import Path

import numpy as np

from blendit.config import load_config
from blendit.data.dataset import StepSegDataset
from blendit.data.graph import save_graph_npz
from blendit.data.load_data import prepare_data, prepare_split_files
from blendit.training.common import DistributedContext, prepare_training_data


def _write_valid_cache(path: Path) -> None:
    save_graph_npz(
        path,
        {
            "face_cont": np.zeros((1, 3), dtype=np.float32),
            "face_surface_type": np.zeros((1,), dtype=np.int64),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_cont": np.empty((0, 1), dtype=np.float32),
            "edge_type": np.empty((0,), dtype=np.int64),
            "edge_relation": np.empty((0,), dtype=np.int64),
            "labels": np.zeros((1,), dtype=np.int64),
        },
    )


def _source_config(tmp_path: Path) -> dict:
    steps_dir = tmp_path / "step_files"
    segs_dir = tmp_path / "seg_files"
    steps_dir.mkdir()
    segs_dir.mkdir()
    config = load_config("data/default.yaml")
    config["data"].update(
        {
            "steps_dir": str(steps_dir),
            "segs_dir": str(segs_dir),
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(tmp_path / "data" / "train.txt"),
            "val_split": str(tmp_path / "data" / "val.txt"),
            "test_split": str(tmp_path / "data" / "test.txt"),
            "cache_num_workers": 0,
        }
    )
    return config


def test_missing_split_files_are_generated_from_step_and_seg_directories(tmp_path: Path):
    config = _source_config(tmp_path)
    steps_dir = Path(config["data"]["steps_dir"])
    segs_dir = Path(config["data"]["segs_dir"])
    for index in range(10):
        (steps_dir / f"part_{index}.step").touch()
        (segs_dir / f"part_{index}.seg").write_text("0\n", encoding="utf-8")

    generated = prepare_split_files(config)

    assert set(generated) == {"train", "val", "test"}
    split_items = {
        split: set(Path(path).read_text(encoding="utf-8").splitlines())
        for split, path in generated.items()
    }
    assert {split: len(items) for split, items in split_items.items()} == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert not (split_items["train"] & split_items["val"])
    assert not (split_items["train"] & split_items["test"])
    assert not (split_items["val"] & split_items["test"])


def test_prepare_data_rebuilds_invalid_and_missing_caches(tmp_path: Path, monkeypatch):
    config = _source_config(tmp_path)
    config["data"]["val_split"] = None
    config["data"]["test_split"] = None
    steps_dir = Path(config["data"]["steps_dir"])
    segs_dir = Path(config["data"]["segs_dir"])
    for name in ("a", "b"):
        (steps_dir / f"{name}.step").touch()
        (segs_dir / f"{name}.seg").write_text("0\n", encoding="utf-8")
    split_path = Path(config["data"]["train_split"])
    split_path.parent.mkdir(parents=True)
    split_path.write_text("a.step\nb.step\n", encoding="utf-8")

    source_dataset = StepSegDataset(config, split="train", source_mode=True)
    invalid_path = source_dataset.samples[0].cache_path
    invalid_path.parent.mkdir(parents=True)
    np.savez(invalid_path, face_cont=np.asarray([[np.nan]], dtype=np.float32))

    def write_cache(self, sample):
        _write_valid_cache(sample.cache_path)

    monkeypatch.setattr(StepSegDataset, "_extract_to_cache", write_cache)

    result = prepare_data(config, num_workers=0)

    assert result.built_cache_files == 2
    assert any(removal.cache_path == str(invalid_path) for removal in result.removed_invalid_caches)
    cached_dataset = StepSegDataset(result.config, split="train")
    assert len(cached_dataset) == 2
    assert {cached_dataset[index].sample_id for index in range(2)} == {"a", "b"}


def test_non_main_rank_waits_without_a_long_collective(tmp_path: Path, monkeypatch):
    config = {
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "prepare_on_start": True,
        }
    }
    prepared_config = {"data": {"cache_dir": str(tmp_path / "prepared_cache")}}
    marker = tmp_path / ".blendit_coord" / "prepare_test-token.ready"
    broadcast_calls = []

    def fake_broadcast(values, src):
        broadcast_calls.append(src)
        if len(broadcast_calls) == 1:
            values[0] = "test-token"
            marker.parent.mkdir(parents=True)
            marker.write_text("ready\n", encoding="utf-8")
        else:
            values[0] = prepared_config

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("non-main rank must not prepare data")

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)
    monkeypatch.setattr("blendit.data.load_data.prepare_data", unexpected_prepare)

    result = prepare_training_data(
        config,
        DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2, backend="gloo"),
    )

    assert result == prepared_config
    assert broadcast_calls == [0, 0]


def test_training_skips_data_preparation_by_default(tmp_path: Path, monkeypatch, capsys):
    config = {"data": {"cache_dir": str(tmp_path / "cache")}}

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("prepare_data must be opt-in during training startup")

    monkeypatch.setattr("blendit.data.load_data.prepare_data", unexpected_prepare)

    result = prepare_training_data(config)

    assert result is config
    assert "data.prepare_on_start=false" in capsys.readouterr().out


def test_non_main_rank_skips_without_distributed_coordination(tmp_path: Path, monkeypatch):
    config = {
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "prepare_on_start": False,
        }
    }

    def unexpected_broadcast(*args, **kwargs):
        raise AssertionError("disabled preparation must not enter distributed coordination")

    monkeypatch.setattr("torch.distributed.broadcast_object_list", unexpected_broadcast)

    result = prepare_training_data(
        config,
        DistributedContext(enabled=True, rank=1, local_rank=1, world_size=2, backend="gloo"),
    )

    assert result is config
