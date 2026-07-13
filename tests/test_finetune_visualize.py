import json
from pathlib import Path

import numpy as np
import pytest

from blendit.inference.finetune_visualize import (
    _binary_classification_metrics,
    _binary_transition_classes,
    _read_filletrec_json_classes,
)


def test_blendit_vbf_and_ebf_map_to_binary_transition():
    classes = np.asarray([0, 1, 2, 0, 2], dtype=np.int64)
    assert _binary_transition_classes(classes).tolist() == [0, 1, 1, 0, 1]


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
