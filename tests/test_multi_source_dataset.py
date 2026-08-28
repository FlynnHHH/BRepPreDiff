from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


def _write_cache(path: Path, label: int) -> None:
    from brepprediff.data.graph import save_graph_npz

    save_graph_npz(
        path,
        {
            "face_cont": np.zeros((2, 18), dtype=np.float32),
            "face_surface_type": np.zeros((2,), dtype=np.int64),
            "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            "edge_cont": np.zeros((2, 9), dtype=np.float32),
            "edge_type": np.zeros((2,), dtype=np.int64),
            "edge_relation": np.zeros((2,), dtype=np.int64),
            "labels": np.full((2,), label, dtype=np.int64),
        },
    )


def _write_source_config(
    path: Path,
    cache_dir: Path,
    split_dir: Path,
    split_sizes: dict[str, int],
) -> None:
    cache_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, size in split_sizes.items():
        items = []
        for index in range(size):
            filename = f"{split}_{index}.npz"
            _write_cache(cache_dir / filename, index)
            items.append(filename)
        (split_dir / f"{split}.txt").write_text(
            "".join(f"{item}\n" for item in items),
            encoding="utf-8",
        )

    config = {
        "data": {
            "steps_dir": str(path.parent / "unused_steps"),
            "segs_dir": str(path.parent / "unused_labels"),
            "cache_dir": str(cache_dir),
            "train_split": str(split_dir / "train.txt"),
            "val_split": str(split_dir / "val.txt"),
            "test_split": str(split_dir / "test.txt"),
        },
        "brep": {
            "uv_grid_size": 1,
            "surface_type_vocab": 32,
            "edge_type_vocab": 32,
            "relation_type_vocab": 4,
        },
        "labels": {"ignore_index": -100},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_multi_source_dataset_combines_mapped_splits_and_never_loads_labels(
    tmp_path: Path,
):
    from brepprediff.data.dataset import MultiSourceDataset

    source_a = tmp_path / "source_a.yaml"
    source_b = tmp_path / "source_b.yaml"
    _write_source_config(
        source_a,
        tmp_path / "cache_a",
        tmp_path / "splits_a",
        {"train": 2, "val": 1, "test": 1},
    )
    _write_source_config(
        source_b,
        tmp_path / "cache_b",
        tmp_path / "splits_b",
        {"train": 1, "val": 1, "test": 2},
    )

    joint_path = tmp_path / "joint.yaml"
    config = {
        "data_config": str(joint_path),
        "data": {
            "strip_labels": True,
            "sources": [
                {
                    "name": "a",
                    "data_config": source_a.name,
                    "splits": {"train": ["train", "val", "test"]},
                },
                {
                    "name": "b",
                    "data_config": source_b.name,
                    "splits": {"train": ["train", "test"]},
                },
            ],
        },
        "brep": {
            "uv_grid_size": 1,
            "surface_type_vocab": 32,
            "edge_type_vocab": 32,
            "relation_type_vocab": 4,
        },
        "train": {"normalize_per_graph": False},
    }

    dataset = MultiSourceDataset(config, split="train")

    assert len(dataset) == 7
    assert dataset.component_names == [
        "a/train",
        "a/val",
        "a/test",
        "b/train",
        "b/test",
    ]
    assert dataset.component_sizes == [2, 1, 1, 1, 2]
    assert dataset[0].labels is None
    assert dataset[-1].labels is None
    assert dataset[0].sample_id.startswith("a/train:")
    assert dataset[-1].sample_id.startswith("b/test:")


def test_multi_source_dataset_rejects_incompatible_feature_grid(tmp_path: Path):
    from brepprediff.data.dataset import MultiSourceDataset

    source = tmp_path / "source.yaml"
    _write_source_config(
        source,
        tmp_path / "cache",
        tmp_path / "splits",
        {"train": 1, "val": 0, "test": 0},
    )
    config = {
        "data_config": str(tmp_path / "joint.yaml"),
        "data": {
            "strip_labels": True,
            "sources": [{"name": "bad", "data_config": source.name}],
        },
        "brep": {
            "uv_grid_size": 10,
            "surface_type_vocab": 32,
            "edge_type_vocab": 32,
            "relation_type_vocab": 4,
        },
        "train": {"normalize_per_graph": False},
    }

    with pytest.raises(ValueError, match="uv_grid_size"):
        MultiSourceDataset(config, split="train")
