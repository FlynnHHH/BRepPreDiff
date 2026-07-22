from __future__ import annotations

import torch


def confusion_matrix_from_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    if probabilities.shape[0] != labels.numel():
        raise ValueError(
            f"Probability/label count mismatch: probabilities={probabilities.shape[0]} "
            f"labels={labels.numel()}."
        )
    valid = labels != ignore_index
    if not valid.any():
        return torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
            device=probabilities.device,
        )
    valid_labels = labels[valid]
    predictions = probabilities[valid].argmax(dim=-1)
    return torch.bincount(
        valid_labels * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(confusion: torch.Tensor) -> dict[str, float]:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got shape={tuple(confusion.shape)}")
    counts = confusion.to(dtype=torch.float64)
    total = counts.sum()
    if total <= 0:
        return {"acc": 0.0, "f1": 0.0, "weighted_f1": 0.0}

    true_positive = counts.diag()
    support = counts.sum(dim=1)
    predicted = counts.sum(dim=0)
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-15)
    metrics = {
        "acc": float((true_positive.sum() / total).detach().cpu()),
        "f1": float(f1.mean().detach().cpu()),
        "weighted_f1": float(((f1 * support).sum() / total).detach().cpu()),
    }
    if confusion.shape[0] == 2:
        metrics.update(
            {
                "precision": float(precision[1].detach().cpu()),
                "recall": float(recall[1].detach().cpu()),
                "positive_f1": float(f1[1].detach().cpu()),
            }
        )
    return metrics
