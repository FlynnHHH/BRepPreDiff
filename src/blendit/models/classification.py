from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from blendit.data.graph import GraphBatch
from blendit.models.downstream import (
    DownstreamEncoder,
    LabelDiffusionModel,
    LabelDiffusionTrainingBatch,
    label_diffusion_objective,
    predict_label_diffusion_probabilities,
    prepare_label_diffusion_training_batch,
)
from blendit.models.encoder import MLP
from blendit.models.metrics import (
    confusion_matrix_from_probabilities,
    metrics_from_confusion_matrix,
)
from blendit.task import CLASSIFICATION, task_type


def _require_classification(config: dict[str, Any]) -> None:
    configured_task = task_type(config)
    if configured_task != CLASSIFICATION:
        raise ValueError(
            f"Classification model received task={configured_task!r}; "
            "use build_segmentation_model or build_finetune_model."
        )


def mean_graph_pool(face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
    num_graphs = len(batch.sample_ids)
    pooled = face_embeddings.new_zeros((num_graphs, face_embeddings.shape[-1]))
    pooled.index_add_(0, batch.batch_index, face_embeddings)
    counts = (
        torch.bincount(batch.batch_index, minlength=num_graphs)
        .to(face_embeddings.dtype)
        .unsqueeze(-1)
    )
    return pooled / counts.clamp_min(1.0)


class ClassificationModel(DownstreamEncoder):
    task = CLASSIFICATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_classification(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        self.cls_head = MLP(
            hidden_dim,
            hidden_dim,
            int(model_cfg["num_classes"]),
            float(model_cfg["dropout"]),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.cls_head(mean_graph_pool(self.encode_faces(batch), batch))


class DiffusionClassificationModel(LabelDiffusionModel):
    task = CLASSIFICATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_classification(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)

    def encode_tokens(self, batch: GraphBatch) -> torch.Tensor:
        return mean_graph_pool(self.encode_faces(batch), batch)

    def token_counts(self, batch: GraphBatch) -> list[int]:
        return [1] * len(batch.sample_ids)


def build_classification_model(
    config: dict[str, Any],
    face_cont_dim: int,
    edge_cont_dim: int,
) -> ClassificationModel | DiffusionClassificationModel:
    _require_classification(config)
    head_type = str(config.get("model", {}).get("finetune_head", "mlp")).lower()
    if head_type == "mlp":
        return ClassificationModel(config, face_cont_dim, edge_cont_dim)
    if head_type == "diffusion":
        return DiffusionClassificationModel(config, face_cont_dim, edge_cont_dim)
    raise ValueError(f"Unsupported model.finetune_head: {head_type!r}")


def compute_classification_label_diffusion_loss(
    prediction: torch.Tensor,
    prepared: LabelDiffusionTrainingBatch,
    model: DiffusionClassificationModel,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics, _ = label_diffusion_objective(
        prediction,
        prepared,
        model,
        class_weights,
    )
    return loss, metrics


def predict_classification_probabilities(
    model: nn.Module,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    target_model = model.module if hasattr(model, "module") else model
    if isinstance(target_model, ClassificationModel):
        return F.softmax(target_model(batch), dim=-1)
    if isinstance(target_model, DiffusionClassificationModel):
        return predict_label_diffusion_probabilities(target_model, batch, config)
    raise TypeError(f"Unsupported classification model type: {type(target_model).__name__}")


def classification_confusion_matrix(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    _require_classification(config)
    if batch.labels is None:
        raise ValueError("Classification requires model labels.")
    expected_models = len(batch.sample_ids)
    if batch.labels.numel() != expected_models:
        raise ValueError(
            f"Classification requires one label per model: models={expected_models} "
            f"labels={batch.labels.numel()}."
        )
    return confusion_matrix_from_probabilities(
        probabilities,
        batch.labels,
        num_classes=int(config["model"]["num_classes"]),
        ignore_index=int(config.get("labels", {}).get("ignore_index", -100)),
    )


def classification_metrics_from_confusion_matrix(confusion: torch.Tensor) -> dict[str, float]:
    return metrics_from_confusion_matrix(confusion)


def classification_metrics_from_probabilities(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    return classification_metrics_from_confusion_matrix(
        classification_confusion_matrix(probabilities, batch, config)
    )


def compute_classification_loss(
    logits: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    _require_classification(config)
    if batch.labels is None:
        raise ValueError("Classification requires model labels.")
    expected_models = len(batch.sample_ids)
    if logits.shape[0] != expected_models or batch.labels.numel() != expected_models:
        raise ValueError(
            f"Classification requires one logit and label per model ({expected_models}): "
            f"logits={logits.shape[0]} labels={batch.labels.numel()}."
        )
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    loss = F.cross_entropy(logits, batch.labels, weight=class_weights, ignore_index=ignore_index)
    with torch.no_grad():
        valid = batch.labels != ignore_index
        accuracy = (
            (logits.argmax(dim=-1)[valid] == batch.labels[valid]).float().mean()
            if valid.any()
            else logits.new_tensor(0.0)
        )
    return loss, {
        "ce": float(loss.detach().cpu()),
        "acc": float(accuracy.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }


__all__ = [
    "ClassificationModel",
    "DiffusionClassificationModel",
    "build_classification_model",
    "classification_confusion_matrix",
    "classification_metrics_from_confusion_matrix",
    "classification_metrics_from_probabilities",
    "compute_classification_label_diffusion_loss",
    "compute_classification_loss",
    "mean_graph_pool",
    "predict_classification_probabilities",
    "prepare_label_diffusion_training_batch",
]
