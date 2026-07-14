from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_segmentation_metrics_from_confusion_matrix_uses_macro_f1():
    import torch

    from blendit.models import segmentation_metrics_from_confusion_matrix

    confusion = torch.tensor(
        [
            [2, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=torch.int64,
    )

    metrics = segmentation_metrics_from_confusion_matrix(confusion)

    assert metrics["acc"] == pytest.approx(4.0 / 6.0)
    assert metrics["f1"] == pytest.approx((0.8 + 2.0 / 3.0 + 0.5) / 3.0)
    assert metrics["weighted_f1"] == pytest.approx((2.0 * 0.8 + 2.0 * (2.0 / 3.0) + 2.0 * 0.5) / 6.0)


def test_segmentation_metrics_from_confusion_matrix_handles_empty_input():
    import torch

    from blendit.models import segmentation_metrics_from_confusion_matrix

    metrics = segmentation_metrics_from_confusion_matrix(torch.zeros((3, 3), dtype=torch.int64))

    assert metrics == {"acc": 0.0, "f1": 0.0, "weighted_f1": 0.0}
