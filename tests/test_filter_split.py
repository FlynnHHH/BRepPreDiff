from __future__ import annotations

import json
from pathlib import Path

from blendit.data.filter_split import filter_split


def test_filter_split_removes_invalid_step_entries(tmp_path: Path):
    split = tmp_path / "train.txt"
    split.write_text("00000001.step\n00000002.step\n", encoding="utf-8")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps({"sample_id": "00000002"}) + "\n", encoding="utf-8")
    output = tmp_path / "train_clean.txt"
    removed = tmp_path / "removed.txt"

    result = filter_split(split, [invalid], output, removed_output_path=removed)

    assert result.kept == 1
    assert result.removed == 1
    assert output.read_text(encoding="utf-8").splitlines() == ["00000001.step"]
    assert removed.read_text(encoding="utf-8").splitlines() == ["00000002.step"]


def test_filter_split_matches_hashed_cache_filenames(tmp_path: Path):
    split = tmp_path / "train.txt"
    split.write_text("00000001_aaaaaaaaaa.npz\n00000002_bbbbbbbbbb.npz\n", encoding="utf-8")
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps({"cache_path": "/tmp/cache/00000002_bbbbbbbbbb.npz"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "train_clean.txt"

    result = filter_split(split, [invalid], output)

    assert result.kept == 1
    assert result.removed == 1
    assert output.read_text(encoding="utf-8").splitlines() == ["00000001_aaaaaaaaaa.npz"]
