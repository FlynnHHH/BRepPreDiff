import json
from pathlib import Path

import numpy as np
import pytest

from blendit.inference.finetune_visualize import (
    _binary_classification_metrics,
    _binary_transition_classes,
    _build_samples,
    _read_filletrec_json_classes,
)


def test_blendit_vbf_and_ebf_map_to_binary_transition():
    classes = np.asarray([0, 1, 2, 0, 2], dtype=np.int64)
    assert _binary_transition_classes(classes).tolist() == [0, 1, 1, 0, 1]


def test_native_binary_predictions_are_not_remapped():
    classes = np.asarray([0, 1, 1, 0], dtype=np.int64)
    assert _binary_transition_classes(classes, source_num_classes=2).tolist() == [0, 1, 1, 0]


def test_filletrec_json_labels_follow_occ_face_order(tmp_path: Path):
    label_path = tmp_path / "42.json"
    label_path.write_text(json.dumps([0, 1, 0, 1]), encoding="utf-8")
    assert _read_filletrec_json_classes(label_path, 4).tolist() == [0, 1, 0, 1]

    with pytest.raises(ValueError, match="label count mismatch"):
        _read_filletrec_json_classes(label_path, 3)


def test_binary_metrics_treat_transition_as_positive_class():
    prediction = np.asarray([0, 1, 1, 0], dtype=np.int64)
    target = np.asarray([0, 1, 0, 1], dtype=np.int64)

    metrics = _binary_classification_metrics(prediction, target)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_split_samples_use_independent_step_and_seg_directories(tmp_path: Path):
    steps_dir = tmp_path / "cad_storage"
    segs_dir = tmp_path / "label_storage"
    steps_dir.mkdir()
    segs_dir.mkdir()
    step_path = steps_dir / "part.step"
    seg_path = segs_dir / "part.seg"
    step_path.touch()
    seg_path.write_text("0\n", encoding="utf-8")
    split_path = tmp_path / "test.txt"
    split_path.write_text("part.step\n", encoding="utf-8")
    config = {
        "data": {
            "steps_dir": str(steps_dir),
            "segs_dir": str(segs_dir),
            "cache_dir": str(tmp_path / "cache"),
            "step_extensions": [".step", ".stp"],
        }
    }

    samples = _build_samples(
        config,
        split="test",
        split_file=str(split_path),
        cache_dir=None,
    )

    assert len(samples) == 1
    assert samples[0].step_path == step_path
    assert samples[0].seg_path == seg_path
