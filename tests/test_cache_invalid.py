from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blendit.config import load_config
from blendit.data.dataset import NonFiniteCacheError, StepSegDataset, _save_occ_cache


def test_nonfinite_occ_cache_is_moved_to_invalid_dir(tmp_path: Path):
    cache_path = tmp_path / "cache" / "bad.npz"
    invalid_dir = tmp_path / "invalid"
    arrays = {
        "face_cont": np.array([[0.0, np.nan], [np.inf, 1.0]], dtype=np.float32),
        "face_surface_type": np.array([1, 2], dtype=np.int64),
    }

    with pytest.raises(NonFiniteCacheError, match=r"face_cont: 2 non-finite values"):
        _save_occ_cache(cache_path, arrays, invalid_dir)

    invalid_path = invalid_dir / cache_path.name
    assert not cache_path.exists()
    assert invalid_path.exists()
    with np.load(invalid_path, allow_pickle=False) as data:
        assert np.isnan(data["face_cont"][0, 1])
        assert np.isposinf(data["face_cont"][1, 0])


def test_finite_occ_cache_remains_in_cache_dir(tmp_path: Path):
    cache_path = tmp_path / "cache" / "good.npz"
    invalid_dir = tmp_path / "invalid"

    _save_occ_cache(
        cache_path,
        {"face_cont": np.array([[0.0, 1.0]], dtype=np.float32)},
        invalid_dir,
    )

    assert cache_path.exists()
    assert not invalid_dir.exists()


def test_cache_build_moves_nonfinite_occ_output_and_logs_failure(tmp_path: Path, monkeypatch):
    step_path = tmp_path / "nonfinite.step"
    step_path.write_text("fake step file\n", encoding="utf-8")
    split_path = tmp_path / "split.txt"
    split_path.write_text("nonfinite.step\n", encoding="utf-8")
    invalid_dir = tmp_path / "invalid"
    invalid_log = invalid_dir / "failures.jsonl"

    config = load_config("configs/default.yaml")
    config["data"].update(
        {
            "root": str(tmp_path),
            "steps_dir": ".",
            "segs_dir": ".",
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(split_path),
            "labels_required": False,
            "overwrite_cache": True,
        }
    )

    class FakeExtractor:
        def __init__(self, config):
            pass

        def extract(self, *args, **kwargs):
            return {"face_cont": np.array([[np.nan]], dtype=np.float32)}

    monkeypatch.setattr("blendit.brep.occ_extractor.OccBRepExtractor", FakeExtractor)
    dataset = StepSegDataset(config, split="train")
    cache_path = dataset.samples[0].cache_path

    failures = dataset.build_cache(num_workers=0, invalid_log=invalid_log)

    assert len(failures) == 1
    assert failures[0].error_type == "NonFiniteCacheError"
    assert not cache_path.exists()
    assert (invalid_dir / cache_path.name).exists()
    record = json.loads(invalid_log.read_text(encoding="utf-8"))
    assert record["sample_id"] == "nonfinite"
    assert record["error_type"] == "NonFiniteCacheError"


def test_cache_failure_is_written_to_jsonl(tmp_path: Path, monkeypatch):
    step_path = tmp_path / "bad.step"
    step_path.write_text("not a valid step file\n", encoding="utf-8")
    split_path = tmp_path / "split.txt"
    split_path.write_text("bad.step\n", encoding="utf-8")
    invalid_log = tmp_path / "invalid.jsonl"

    config = load_config("configs/default.yaml")
    config["data"].update(
        {
            "root": str(tmp_path),
            "steps_dir": ".",
            "segs_dir": ".",
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(split_path),
            "labels_required": False,
            "overwrite_cache": True,
        }
    )

    dataset = StepSegDataset(config, split="train")

    def fail_extract(self, sample):
        raise RuntimeError("synthetic cache failure")

    monkeypatch.setattr(StepSegDataset, "_extract_to_cache", fail_extract)
    failures = dataset.build_cache(num_workers=0, invalid_log=invalid_log)

    assert len(failures) == 1
    records = [json.loads(line) for line in invalid_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["sample_id"] == "bad"
    assert records[0]["step_path"].endswith("bad.step")
    assert records[0]["error_type"] == "RuntimeError"
    assert records[0]["error"] == "synthetic cache failure"


def test_build_cache_skips_existing_cache_by_default(tmp_path: Path, monkeypatch):
    step_path = tmp_path / "done.step"
    step_path.write_text("already cached\n", encoding="utf-8")
    split_path = tmp_path / "split.txt"
    split_path.write_text("done.step\n", encoding="utf-8")

    config = load_config("configs/default.yaml")
    config["data"].update(
        {
            "root": str(tmp_path),
            "steps_dir": ".",
            "segs_dir": ".",
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(split_path),
            "labels_required": False,
            "overwrite_cache": False,
        }
    )

    dataset = StepSegDataset(config, split="train")
    dataset.samples[0].cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.samples[0].cache_path.write_bytes(b"existing cache")

    def fail_if_called(self, sample):
        raise AssertionError("existing cache should be skipped")

    monkeypatch.setattr(StepSegDataset, "_extract_to_cache", fail_if_called)
    failures = dataset.build_cache(num_workers=0)

    assert failures == []


def test_build_cache_overwrite_reprocesses_existing_cache(tmp_path: Path, monkeypatch):
    step_path = tmp_path / "done.step"
    step_path.write_text("already cached\n", encoding="utf-8")
    split_path = tmp_path / "split.txt"
    split_path.write_text("done.step\n", encoding="utf-8")

    config = load_config("configs/default.yaml")
    config["data"].update(
        {
            "root": str(tmp_path),
            "steps_dir": ".",
            "segs_dir": ".",
            "cache_dir": str(tmp_path / "cache"),
            "train_split": str(split_path),
            "labels_required": False,
            "overwrite_cache": True,
        }
    )

    dataset = StepSegDataset(config, split="train")
    dataset.samples[0].cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.samples[0].cache_path.write_bytes(b"existing cache")
    calls = []

    def record_extract(self, sample):
        calls.append(sample.sample_id)

    monkeypatch.setattr(StepSegDataset, "_extract_to_cache", record_extract)
    failures = dataset.build_cache(num_workers=0)

    assert failures == []
    assert calls == ["done"]
