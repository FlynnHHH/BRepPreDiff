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
    counts = _graph_counts(face_embeddings, batch)
    return pooled / counts.clamp_min(1.0)


def _graph_counts(face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
    return (
        torch.bincount(batch.batch_index, minlength=len(batch.sample_ids))
        .to(face_embeddings.dtype)
        .unsqueeze(-1)
    )


def _max_graph_pool(face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
    pooled = face_embeddings.new_full(
        (len(batch.sample_ids), face_embeddings.shape[-1]),
        -torch.inf,
    )
    graph_indices = batch.batch_index.unsqueeze(-1).expand_as(face_embeddings)
    pooled.scatter_reduce_(
        0,
        graph_indices,
        face_embeddings,
        reduce="amax",
        include_self=True,
    )
    return pooled


def _std_graph_pool(face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
    mean = mean_graph_pool(face_embeddings, batch)
    mean_square = face_embeddings.new_zeros(mean.shape)
    mean_square.index_add_(0, batch.batch_index, face_embeddings.square())
    mean_square = mean_square / _graph_counts(face_embeddings, batch).clamp_min(1.0)
    return (mean_square - mean.square()).clamp_min(1.0e-12).sqrt()


class MeanGraphPool(nn.Module):
    def forward(self, face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        return mean_graph_pool(face_embeddings, batch)


class MeanStatisticGraphPool(nn.Module):
    """Fuse mean pooling with a complementary max or standard-deviation statistic.

    The residual projection is zero-initialized so fine-tuning starts from the exact
    mean-pooling baseline instead of perturbing a pretrained encoder immediately.
    """

    def __init__(self, hidden_dim: int, dropout: float, statistic: str) -> None:
        super().__init__()
        if statistic not in {"max", "std"}:
            raise ValueError(f"Unsupported graph statistic: {statistic!r}")
        self.statistic = statistic
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        final_projection = self.residual[-1]
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(self, face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        mean = mean_graph_pool(face_embeddings, batch)
        if self.statistic == "max":
            complement = _max_graph_pool(face_embeddings, batch)
        else:
            complement = _std_graph_pool(face_embeddings, batch)
        return mean + self.residual(torch.cat((mean, complement), dim=-1))


class ResidualAttentionGraphPool(nn.Module):
    """Learn face importance without discarding the stable mean representation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        final_score = self.score[-1]
        nn.init.zeros_(final_score.weight)
        nn.init.zeros_(final_score.bias)
        # Begin with a conservative 20% residual-attention mixing coefficient.
        self.mix_logit = nn.Parameter(torch.tensor(-1.3862944))

    def forward(self, face_embeddings: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        mean = mean_graph_pool(face_embeddings, batch)
        logits = self.score(face_embeddings).squeeze(-1).clamp(min=-10.0, max=10.0)
        graph_max = logits.new_full((len(batch.sample_ids),), -torch.inf)
        graph_max.scatter_reduce_(
            0,
            batch.batch_index,
            logits,
            reduce="amax",
            include_self=True,
        )
        weights = torch.exp(logits - graph_max[batch.batch_index])
        denominators = logits.new_zeros((len(batch.sample_ids),))
        denominators.index_add_(0, batch.batch_index, weights)
        weights = weights / denominators[batch.batch_index].clamp_min(1.0e-8)
        attended = face_embeddings.new_zeros(mean.shape)
        attended.index_add_(0, batch.batch_index, face_embeddings * weights.unsqueeze(-1))
        return mean + self.mix_logit.sigmoid() * (attended - mean)


def build_graph_pool(model_cfg: dict[str, Any]) -> nn.Module:
    hidden_dim = int(model_cfg["hidden_dim"])
    dropout = float(model_cfg["dropout"])
    pooling = str(model_cfg.get("graph_pooling", "mean")).lower()
    if pooling == "mean":
        return MeanGraphPool()
    if pooling == "mean_max":
        return MeanStatisticGraphPool(hidden_dim, dropout, statistic="max")
    if pooling == "mean_std":
        return MeanStatisticGraphPool(hidden_dim, dropout, statistic="std")
    if pooling == "residual_attention":
        return ResidualAttentionGraphPool(hidden_dim)
    supported = ["mean", "mean_max", "mean_std", "residual_attention"]
    raise ValueError(
        f"Unsupported model.graph_pooling {pooling!r}; expected one of {supported}."
    )


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
        # Construct the shared classification head before method-specific pooling
        # modules so a fixed seed gives every pooling ablation the same head weights.
        self.graph_pool = build_graph_pool(model_cfg)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        return self.cls_head(self.graph_pool(self.encode_faces(batch), batch))


class DiffusionClassificationModel(LabelDiffusionModel):
    task = CLASSIFICATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_classification(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)
        self.graph_pool = build_graph_pool(config["model"])

    def encode_tokens(self, batch: GraphBatch) -> torch.Tensor:
        return self.graph_pool(self.encode_faces(batch), batch)

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
    "MeanGraphPool",
    "MeanStatisticGraphPool",
    "ResidualAttentionGraphPool",
    "build_graph_pool",
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
