from __future__ import annotations

import pytest
import torch

from brepprediff.training.evaluate import (
    classification_metrics_from_confusion,
    pair_step_seg_paths,
)


def test_evaluate_parser_collects_ensemble_checkpoints():
    from brepprediff.training.evaluate import build_arg_parser
    args = build_arg_parser().parse_args([
        '--checkpoint', 'primary.pt', '--ensemble-checkpoint', 'second.pt',
        '--ensemble-checkpoint', 'third.pt', '--rotation-tta', '4',
        '--graph-smoothing-alpha', '0.2',
    ])
    assert args.checkpoint == 'primary.pt'
    assert args.ensemble_checkpoint == ['second.pt', 'third.pt']
    assert args.rotation_tta == 4
    assert args.graph_smoothing_alpha == 0.2


def test_confidence_gated_graph_smoothing_preserves_confident_and_isolated_faces():
    import torch
    from brepprediff.training.evaluate import confidence_gated_graph_smoothing
    probabilities = torch.tensor([[1., 0.], [.4, .6], [0., 1.]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    smoothed = confidence_gated_graph_smoothing(probabilities, edge_index, 0.5)
    assert torch.equal(smoothed[0], probabilities[0])
    assert smoothed[1, 0] > probabilities[1, 0]
    assert torch.equal(smoothed[2], probabilities[2])
    assert torch.allclose(smoothed.sum(-1), torch.ones(3))


def test_classification_metrics_include_per_class_iou_and_transition_binary() -> None:
    confusion = torch.tensor(
        [
            [8, 1, 1],
            [2, 3, 1],
            [0, 1, 3],
        ],
        dtype=torch.int64,
    )

    metrics = classification_metrics_from_confusion(confusion)

    assert metrics["faces"] == 20
    assert metrics["accuracy"] == pytest.approx(14 / 20)
    assert metrics["per_class"][0]["precision"] == pytest.approx(8 / 10)
    assert metrics["per_class"][0]["recall"] == pytest.approx(8 / 10)
    assert metrics["per_class"][0]["iou"] == pytest.approx(8 / 12)
    assert metrics["transition_binary"] == {
        "accuracy": pytest.approx(16 / 20),
        "precision": pytest.approx(8 / 10),
        "recall": pytest.approx(8 / 10),
        "f1": pytest.approx(8 / 10),
        "iou": pytest.approx(8 / 12),
        "true_positive": 8,
        "true_negative": 8,
        "false_positive": 2,
        "false_negative": 2,
        "faces": 20,
    }


def test_classification_metrics_handle_empty_confusion() -> None:
    metrics = classification_metrics_from_confusion(torch.zeros((3, 3), dtype=torch.int64))

    assert metrics["faces"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["transition_binary"]["f1"] == 0.0


def test_classification_metrics_can_disable_blend_specific_binary_view() -> None:
    metrics = classification_metrics_from_confusion(
        torch.eye(25, dtype=torch.int64),
        include_transition_binary=False,
    )

    assert "transition_binary" not in metrics


def test_binary_classification_uses_transition_class_name() -> None:
    metrics = classification_metrics_from_confusion(torch.eye(2, dtype=torch.int64))

    assert [row["class_name"] for row in metrics["per_class"]] == [
        "NonTransition",
        "Transition",
    ]


def test_pair_step_seg_direct_files(tmp_path) -> None:
    step_path = tmp_path / "part.step"
    seg_path = tmp_path / "ground_truth.seg"
    step_path.write_text("step", encoding="utf-8")
    seg_path.write_text("0\n", encoding="utf-8")

    samples = pair_step_seg_paths(step_path, seg_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "part"
    assert samples[0].step_path == step_path
    assert samples[0].seg_path == seg_path


def test_pair_step_seg_directories_by_relative_path(tmp_path) -> None:
    step_root = tmp_path / "steps"
    seg_root = tmp_path / "labels"
    (step_root / "train").mkdir(parents=True)
    (seg_root / "train").mkdir(parents=True)
    (step_root / "train" / "part.STEP").write_text("step", encoding="utf-8")
    (seg_root / "train" / "part.seg").write_text("0\n", encoding="utf-8")

    samples = pair_step_seg_paths(step_root, seg_root)

    assert len(samples) == 1
    assert samples[0].sample_id == "train/part"
    assert samples[0].seg_path == (seg_root / "train" / "part.seg").resolve()


def test_pair_step_seg_directories_require_every_label(tmp_path) -> None:
    step_root = tmp_path / "steps"
    seg_root = tmp_path / "labels"
    step_root.mkdir()
    seg_root.mkdir()
    (step_root / "missing.stp").write_text("step", encoding="utf-8")
    (seg_root / "other.seg").write_text("0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No matching SEG/JSON"):
        pair_step_seg_paths(step_root, seg_root)
