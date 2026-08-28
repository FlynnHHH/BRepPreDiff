from __future__ import annotations

from typing import Any

import torch
from torch import nn

from brepprediff.data.graph import GraphBatch
from brepprediff.models.classification import (
    build_classification_model,
    classification_confusion_matrix,
    classification_metrics_from_confusion_matrix,
    classification_metrics_from_probabilities,
    compute_classification_label_diffusion_loss,
    compute_classification_loss,
    predict_classification_probabilities,
)
from brepprediff.models.downstream import LabelDiffusionModel, LabelDiffusionTrainingBatch
from brepprediff.models.segmentation import (
    build_segmentation_model,
    compute_label_diffusion_loss,
    compute_segmentation_loss,
    predict_segmentation_probabilities,
    segmentation_confusion_matrix,
    segmentation_metrics_from_confusion_matrix,
    segmentation_metrics_from_probabilities,
)
from brepprediff.task import CLASSIFICATION, task_type


def build_finetune_model(
    config: dict[str, Any],
    face_cont_dim: int,
    edge_cont_dim: int,
) -> nn.Module:
    if task_type(config) == CLASSIFICATION:
        return build_classification_model(config, face_cont_dim, edge_cont_dim)
    return build_segmentation_model(config, face_cont_dim, edge_cont_dim)


def compute_finetune_loss(
    logits: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if task_type(config) == CLASSIFICATION:
        return compute_classification_loss(logits, batch, config, class_weights)
    return compute_segmentation_loss(logits, batch, config, class_weights)


def compute_finetune_label_diffusion_loss(
    prediction: torch.Tensor,
    prepared: LabelDiffusionTrainingBatch,
    model: LabelDiffusionModel,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if task_type(config) == CLASSIFICATION:
        return compute_classification_label_diffusion_loss(
            prediction,
            prepared,
            model,
            class_weights,
        )
    return compute_label_diffusion_loss(prediction, prepared, model, class_weights)


def predict_finetune_probabilities(
    model: nn.Module,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    if task_type(config) == CLASSIFICATION:
        return predict_classification_probabilities(model, batch, config)
    return predict_segmentation_probabilities(model, batch, config)


def finetune_confusion_matrix(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    if task_type(config) == CLASSIFICATION:
        return classification_confusion_matrix(probabilities, batch, config)
    return segmentation_confusion_matrix(probabilities, batch, config)


def finetune_metrics_from_probabilities(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    if task_type(config) == CLASSIFICATION:
        return classification_metrics_from_probabilities(probabilities, batch, config)
    return segmentation_metrics_from_probabilities(probabilities, batch, config)


def finetune_metrics_from_confusion_matrix(
    confusion: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, float]:
    if task_type(config) == CLASSIFICATION:
        return classification_metrics_from_confusion_matrix(confusion)
    return segmentation_metrics_from_confusion_matrix(confusion)


__all__ = [
    "build_finetune_model",
    "compute_finetune_label_diffusion_loss",
    "compute_finetune_loss",
    "finetune_confusion_matrix",
    "finetune_metrics_from_confusion_matrix",
    "finetune_metrics_from_probabilities",
    "predict_finetune_probabilities",
]
