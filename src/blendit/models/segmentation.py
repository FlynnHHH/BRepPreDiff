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
from blendit.task import SEGMENTATION, task_type


def _require_segmentation(config: dict[str, Any]) -> None:
    configured_task = task_type(config)
    if configured_task != SEGMENTATION:
        raise ValueError(
            f"Segmentation model received task={configured_task!r}; "
            "use build_classification_model or build_finetune_model."
        )


class SegmentationModel(DownstreamEncoder):
    task = SEGMENTATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_segmentation(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        self.seg_head = MLP(
            hidden_dim,
            hidden_dim,
            int(model_cfg["num_classes"]),
            float(model_cfg["dropout"]),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.seg_head(self.encode_faces(batch))


class DiffusionSegmentationModel(LabelDiffusionModel):
    task = SEGMENTATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_segmentation(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)

    def encode_tokens(self, batch: GraphBatch) -> torch.Tensor:
        return self.encode_faces(batch)

    def token_counts(self, batch: GraphBatch) -> list[int]:
        graph_ptr = batch.graph_ptr.detach().cpu().tolist()
        return [
            int(graph_ptr[index + 1] - graph_ptr[index])
            for index in range(len(batch.sample_ids))
        ]

    def encode(self, batch: GraphBatch) -> torch.Tensor:
        """Compatibility alias for the original DiffusionSegmentationModel API."""
        return self.encode_tokens(batch)


def build_segmentation_model(
    config: dict[str, Any],
    face_cont_dim: int,
    edge_cont_dim: int,
) -> SegmentationModel | DiffusionSegmentationModel:
    _require_segmentation(config)
    head_type = str(config.get("model", {}).get("finetune_head", "mlp")).lower()
    if head_type == "mlp":
        return SegmentationModel(config, face_cont_dim, edge_cont_dim)
    if head_type == "diffusion":
        return DiffusionSegmentationModel(config, face_cont_dim, edge_cont_dim)
    raise ValueError(f"Unsupported model.finetune_head: {head_type!r}")


def dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = labels != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)
    valid_logits = logits[valid]
    valid_labels = labels[valid]
    probabilities = F.softmax(valid_logits, dim=-1)
    target = F.one_hot(
        valid_labels.clamp(0, num_classes - 1),
        num_classes=num_classes,
    ).float()
    intersection = (probabilities * target).sum(dim=0)
    denominator = probabilities.sum(dim=0) + target.sum(dim=0)
    dice = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    return 1.0 - dice.mean()


def compute_label_diffusion_loss(
    prediction: torch.Tensor,
    prepared: LabelDiffusionTrainingBatch,
    model: DiffusionSegmentationModel,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics, x_start_prediction = label_diffusion_objective(
        prediction,
        prepared,
        model,
        class_weights,
    )
    with torch.no_grad():
        dsc = dice_loss(
            x_start_prediction,
            prepared.labels,
            model.num_classes,
            ignore_index=-100,
        )
    metrics["dice"] = float(dsc.detach().cpu())
    return loss, metrics


def predict_segmentation_probabilities(
    model: nn.Module,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    target_model = model.module if hasattr(model, "module") else model
    if isinstance(target_model, SegmentationModel):
        return F.softmax(target_model(batch), dim=-1)
    if isinstance(target_model, DiffusionSegmentationModel):
        return predict_label_diffusion_probabilities(target_model, batch, config)
    raise TypeError(f"Unsupported segmentation model type: {type(target_model).__name__}")


def segmentation_confusion_matrix(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    _require_segmentation(config)
    if batch.labels is None:
        raise ValueError("Segmentation requires face labels.")
    expected_faces = batch.face_cont.shape[0]
    if batch.labels.numel() != expected_faces:
        raise ValueError(
            f"Segmentation requires one label per face: faces={expected_faces} "
            f"labels={batch.labels.numel()}."
        )
    return confusion_matrix_from_probabilities(
        probabilities,
        batch.labels,
        num_classes=int(config["model"]["num_classes"]),
        ignore_index=int(config.get("labels", {}).get("ignore_index", -100)),
    )


def segmentation_metrics_from_confusion_matrix(confusion: torch.Tensor) -> dict[str, float]:
    return metrics_from_confusion_matrix(confusion)


def segmentation_metrics_from_probabilities(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    if batch.labels is None:
        raise ValueError("Segmentation requires face labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    num_classes = int(config["model"]["num_classes"])
    valid = batch.labels != ignore_index
    if not valid.any():
        return {"acc": 0.0, "f1": 0.0, "weighted_f1": 0.0, "dice": 1.0}
    metrics = segmentation_metrics_from_confusion_matrix(
        segmentation_confusion_matrix(probabilities, batch, config)
    )
    valid_probabilities = probabilities[valid]
    valid_labels = batch.labels[valid]
    targets = F.one_hot(valid_labels, num_classes=num_classes).float()
    intersection = (valid_probabilities * targets).sum(dim=0)
    denominator = valid_probabilities.sum(dim=0) + targets.sum(dim=0)
    dice = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    metrics["dice"] = float((1.0 - dice.mean()).detach().cpu())
    return metrics


def compute_segmentation_loss(
    logits: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    _require_segmentation(config)
    if batch.labels is None:
        raise ValueError("Segmentation requires face labels.")
    expected_faces = batch.face_cont.shape[0]
    if logits.shape[0] != expected_faces or batch.labels.numel() != expected_faces:
        raise ValueError(
            f"Segmentation requires one logit and label per face ({expected_faces}): "
            f"logits={logits.shape[0]} labels={batch.labels.numel()}."
        )
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    num_classes = int(config["model"]["num_classes"])
    ce = F.cross_entropy(logits, batch.labels, weight=class_weights, ignore_index=ignore_index)
    dsc = dice_loss(logits, batch.labels, num_classes, ignore_index)
    total = ce + float(config["train"].get("dice_loss_weight", 0.3)) * dsc
    with torch.no_grad():
        valid = batch.labels != ignore_index
        accuracy = (
            (logits.argmax(dim=-1)[valid] == batch.labels[valid]).float().mean()
            if valid.any()
            else logits.new_tensor(0.0)
        )
    return total, {
        "ce": float(ce.detach().cpu()),
        "dice": float(dsc.detach().cpu()),
        "acc": float(accuracy.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


__all__ = [
    "DiffusionSegmentationModel",
    "LabelDiffusionTrainingBatch",
    "SegmentationModel",
    "build_segmentation_model",
    "compute_label_diffusion_loss",
    "compute_segmentation_loss",
    "dice_loss",
    "predict_segmentation_probabilities",
    "prepare_label_diffusion_training_batch",
    "segmentation_confusion_matrix",
    "segmentation_metrics_from_confusion_matrix",
    "segmentation_metrics_from_probabilities",
]
