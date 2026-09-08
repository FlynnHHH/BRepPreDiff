from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from brepprediff.data.graph import GraphBatch
from brepprediff.models.downstream import (
    DownstreamEncoder,
    LabelDiffusionModel,
    LabelDiffusionTrainingBatch,
    label_diffusion_objective,
    predict_label_diffusion_probabilities,
    prepare_label_diffusion_training_batch,
)
from brepprediff.models.encoder import MLP, EdgeAttentionLayer, sinusoidal_timestep_embedding
from brepprediff.models.metrics import (
    confusion_matrix_from_probabilities,
    metrics_from_confusion_matrix,
)
from brepprediff.task import SEGMENTATION, task_type


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
        self.boundary_head = None
        self.operation_head = None
        operation_map = model_cfg.get('operation_class_map')
        if operation_map is not None:
            if len(operation_map) != int(model_cfg['num_classes']) or min(operation_map) < 0:
                raise ValueError('operation_class_map must map every class to a nonnegative operation ID.')
            if set(operation_map) != set(range(max(operation_map) + 1)):
                raise ValueError('Operation IDs must be contiguous.')
            self.operation_head = MLP(hidden_dim, hidden_dim, max(operation_map) + 1,
                                      float(model_cfg['dropout']))
        self.boundary_refine = bool(model_cfg.get("boundary_refine", False))
        if bool(model_cfg.get("boundary_aux", False)) or self.boundary_refine:
            self.boundary_head = MLP(hidden_dim * 2 + edge_cont_dim, hidden_dim, 1, 0.0)
            if self.boundary_refine:
                self.refine_message = MLP(hidden_dim, hidden_dim, hidden_dim, 0.0)
                self.refine_norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch: GraphBatch, *, return_aux: bool = False):
        face_h = self.encode_faces(batch)
        boundary_logits = None
        if self.boundary_head is not None:
            src, dst = batch.edge_index
            boundary_logits = self.boundary_head(torch.cat((
                face_h[src] + face_h[dst], (face_h[src] - face_h[dst]).abs(), batch.edge_cont,
            ), dim=-1)).squeeze(-1)
            if self.boundary_refine:
                message = self.refine_message(face_h)[src] * (1 - boundary_logits.sigmoid())[:, None]
                aggregate = torch.zeros_like(face_h)
                aggregate.index_add_(0, dst, message)
                degree = torch.bincount(dst, minlength=face_h.shape[0]).clamp_min(1)
                face_h = self.refine_norm(face_h + aggregate / degree[:, None])
        logits = self.seg_head(face_h)
        if return_aux:
            return {"logits": logits, "boundary_logits": boundary_logits,
                    "operation_logits": self.operation_head(face_h) if self.operation_head is not None else None}
        return logits


class GraphLabelDenoiser(nn.Module):
    """Joint face-label denoising with immutable geometry conditions."""

    def __init__(self, classes: int, hidden: int, heads: int, depth: int, out_channels: int):
        super().__init__()
        self.in_channels = classes
        self.hidden_dim = hidden
        self.label_projection = nn.Linear(classes, hidden)
        self.time_projection = MLP(hidden, hidden, hidden)
        self.layers = nn.ModuleList([EdgeAttentionLayer(hidden, 0.0, num_heads=heads)
                                     for _ in range(depth)])
        self.output = nn.Linear(hidden, out_channels)

    def forward(self, x_t, timesteps, condition, edge_h, edge_index):
        h = self.label_projection(x_t) + condition
        h = h + self.time_projection(sinusoidal_timestep_embedding(timesteps, self.hidden_dim))
        for layer in self.layers:
            h, _ = layer(h, edge_h, edge_index)
        return self.output(h)


class DiffusionSegmentationModel(LabelDiffusionModel):
    task = SEGMENTATION

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        _require_segmentation(config)
        super().__init__(config, face_cont_dim, edge_cont_dim)
        self.structured_labels = bool(config.get('label_diffusion', {}).get('structured', False))
        if self.structured_labels:
            self.diffusion_head = GraphLabelDenoiser(
                self.num_classes, int(config['model']['hidden_dim']),
                int(config['model'].get('num_heads', 4)),
                int(config['label_diffusion'].get('graph_head_depth', 2)),
                self.num_classes * (2 if self.prediction_type == 'x_start_epsilon' else 1),
            )

    def encode_graph_condition(self, batch):
        return self.encoder(batch.face_cont, batch.face_surface_type, batch.edge_index,
                            batch.edge_cont, batch.edge_type, batch.edge_relation,
                            graph_ptr=batch.graph_ptr)

    def forward(self, batch, noisy_labels, timesteps, token_indices):
        if not self.structured_labels:
            return super().forward(batch, noisy_labels, timesteps, token_indices)
        if token_indices.numel() != batch.face_cont.shape[0] or not torch.equal(
            token_indices, torch.arange(batch.face_cont.shape[0], device=token_indices.device)
        ):
            raise ValueError('Structured denoising requires all faces in original order.')
        condition, edge_h = self.encode_graph_condition(batch)
        return self.diffusion_head(noisy_labels, timesteps, condition, edge_h, batch.edge_index)

    def sample(self, batch, **kwargs):
        if not self.structured_labels:
            return super().sample(batch, **kwargs)
        condition, edge_h = self.encode_graph_condition(batch)
        head = self.diffusion_head

        class BoundDenoiser:
            in_channels = head.in_channels

            def __call__(self, x_t, timesteps, face_condition):
                return head(x_t, timesteps, face_condition, edge_h, batch.edge_index)

        return self.schedule.sample(BoundDenoiser(), condition,
                                    prediction_type=self.prediction_type, **kwargs)

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
    boundary_logits = None
    operation_logits = None
    if isinstance(logits, dict):
        boundary_logits = logits['boundary_logits']
        operation_logits = logits.get('operation_logits')
        logits = logits['logits']
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
    boundary_loss = logits.sum() * 0.0
    operation_loss = logits.sum() * 0.0
    if operation_logits is not None:
        mapping = torch.as_tensor(config['model']['operation_class_map'], device=logits.device)
        valid = batch.labels != ignore_index
        targets = mapping[batch.labels.masked_fill(~valid, 0)]
        losses = F.cross_entropy(operation_logits, targets, reduction='none')
        operation_loss = (losses * valid).sum() / valid.sum().clamp_min(1)
        total = total + float(config['train'].get('operation_loss_weight', 0.2)) * operation_loss
    if boundary_logits is not None:
        src, dst = batch.edge_index
        edge_valid = (batch.labels[src] != ignore_index) & (batch.labels[dst] != ignore_index)
        targets = (batch.labels[src] != batch.labels[dst]).to(logits.dtype)
        per_edge = F.binary_cross_entropy_with_logits(boundary_logits, targets, reduction='none')
        boundary_loss = (per_edge * edge_valid).sum() / edge_valid.sum().clamp_min(1)
        total = total + float(config['train'].get('boundary_loss_weight', 0.1)) * boundary_loss
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
        "boundary": float(boundary_loss.detach().cpu()),
        "operation": float(operation_loss.detach().cpu()),
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
