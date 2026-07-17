from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from blendit.data.graph import GraphBatch
from blendit.models.encoder import BRepGraphEncoder, MLP
from blendit.models.diffusion import (
    ConditionalDenoisingMLP,
    LabelDiffusionSchedule,
    bipolar_one_hot,
)


class SegmentationModel(nn.Module):
    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        brep_cfg = config["brep"]
        hidden_dim = int(model_cfg["hidden_dim"])
        self.encoder = BRepGraphEncoder(
            face_cont_dim,
            edge_cont_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            surface_type_vocab=int(brep_cfg["surface_type_vocab"]),
            edge_type_vocab=int(brep_cfg["edge_type_vocab"]),
            relation_type_vocab=int(brep_cfg["relation_type_vocab"]),
        )
        self.seg_head = MLP(hidden_dim, hidden_dim, int(model_cfg["num_classes"]), float(model_cfg["dropout"]))

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        node_h, _ = self.encoder(
            batch.face_cont,
            batch.face_surface_type,
            batch.edge_index,
            batch.edge_cont,
            batch.edge_type,
            batch.edge_relation,
        )
        return self.seg_head(node_h)


class DiffusionSegmentationModel(nn.Module):
    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        brep_cfg = config["brep"]
        diffusion_cfg = config["label_diffusion"]
        if str(diffusion_cfg.get("target_encoding", "bipolar_one_hot")) != "bipolar_one_hot":
            raise ValueError("label_diffusion.target_encoding must be 'bipolar_one_hot'.")
        self.prediction_type = str(diffusion_cfg.get("prediction_type", "epsilon"))
        self.x_start_loss_weight = float(diffusion_cfg.get("x_start_loss_weight", 1.0))
        self.epsilon_loss_weight = float(diffusion_cfg.get("epsilon_loss_weight", 1.0))
        if self.x_start_loss_weight < 0.0 or self.epsilon_loss_weight < 0.0:
            raise ValueError("Label diffusion loss weights must be non-negative.")
        supported_prediction_types = {"epsilon", "x_start", "x_start_epsilon"}
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                "label_diffusion.prediction_type must be one of "
                f"{sorted(supported_prediction_types)}, got {self.prediction_type!r}."
            )
        if str(diffusion_cfg.get("sampling_method", "ddim")) != "ddim":
            raise ValueError("label_diffusion.sampling_method must be 'ddim'.")
        hidden_dim = int(model_cfg["hidden_dim"])
        self.num_classes = int(model_cfg["num_classes"])
        self.encoder = BRepGraphEncoder(
            face_cont_dim,
            edge_cont_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            surface_type_vocab=int(brep_cfg["surface_type_vocab"]),
            edge_type_vocab=int(brep_cfg["edge_type_vocab"]),
            relation_type_vocab=int(brep_cfg["relation_type_vocab"]),
        )
        self.diffusion_head = ConditionalDenoisingMLP(
            self.num_classes,
            hidden_dim,
            width=int(diffusion_cfg.get("head_width", hidden_dim)),
            depth=int(diffusion_cfg.get("head_depth", 3)),
            dropout=float(diffusion_cfg.get("head_dropout", 0.0)),
            out_channels=self.num_classes * (2 if self.prediction_type == "x_start_epsilon" else 1),
        )
        self.schedule = LabelDiffusionSchedule(
            int(diffusion_cfg.get("train_timesteps", 1000)),
            noise_schedule=str(diffusion_cfg.get("noise_schedule", "cosine")),
            beta_start=float(diffusion_cfg.get("beta_start", 1.0e-4)),
            beta_end=float(diffusion_cfg.get("beta_end", 0.02)),
        )

    def encode(self, batch: GraphBatch) -> torch.Tensor:
        node_h, _ = self.encoder(
            batch.face_cont,
            batch.face_surface_type,
            batch.edge_index,
            batch.edge_cont,
            batch.edge_type,
            batch.edge_relation,
        )
        return node_h

    def forward(
        self,
        batch: GraphBatch,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
        face_indices: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode(batch)[face_indices]
        return self.diffusion_head(noisy_labels, timesteps, condition)

    def sample(
        self,
        batch: GraphBatch,
        *,
        initial_noise: torch.Tensor | None = None,
        steps: int,
        eta: float = 0.0,
        temperature: float = 1.0,
        clip_x_start: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode(batch)
        return self.schedule.sample(
            self.diffusion_head,
            condition,
            initial_noise=initial_noise,
            steps=steps,
            eta=eta,
            temperature=temperature,
            clip_x_start=clip_x_start,
            prediction_type=self.prediction_type,
            generator=generator,
        )


def build_segmentation_model(
    config: dict[str, Any],
    face_cont_dim: int,
    edge_cont_dim: int,
) -> SegmentationModel | DiffusionSegmentationModel:
    head_type = str(config.get("model", {}).get("finetune_head", "mlp")).lower()
    if head_type == "mlp":
        return SegmentationModel(config, face_cont_dim, edge_cont_dim)
    if head_type == "diffusion":
        return DiffusionSegmentationModel(config, face_cont_dim, edge_cont_dim)
    raise ValueError(f"Unsupported model.finetune_head: {head_type!r}")


@dataclass(frozen=True)
class LabelDiffusionTrainingBatch:
    x_start: torch.Tensor
    x_t: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    face_indices: torch.Tensor
    labels: torch.Tensor


def prepare_label_diffusion_training_batch(
    model: DiffusionSegmentationModel,
    batch: GraphBatch,
    config: dict[str, Any],
) -> LabelDiffusionTrainingBatch:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires face labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    valid_indices = torch.nonzero(batch.labels != ignore_index, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("Fine-tuning batch does not contain any valid face labels.")

    repeats = max(1, int(config["label_diffusion"].get("noise_samples_per_token", 1)))
    face_indices = valid_indices.repeat(repeats)
    labels = batch.labels[face_indices]
    x_start = bipolar_one_hot(labels, model.num_classes).to(dtype=batch.face_cont.dtype)
    timesteps = torch.randint(
        0,
        model.schedule.timesteps,
        (x_start.shape[0],),
        dtype=torch.long,
        device=x_start.device,
    )
    x_t, noise = model.schedule.q_sample(x_start, timesteps)
    return LabelDiffusionTrainingBatch(
        x_start=x_start,
        x_t=x_t,
        noise=noise,
        timesteps=timesteps,
        face_indices=face_indices,
        labels=labels,
    )


def compute_label_diffusion_loss(
    prediction: torch.Tensor,
    prepared: LabelDiffusionTrainingBatch,
    model: DiffusionSegmentationModel,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction_type = model.prediction_type
    x_start_prediction, epsilon_prediction = model.schedule.model_predictions(
        prepared.x_t,
        prepared.timesteps,
        prediction,
        prediction_type,
    )

    component_losses: dict[str, torch.Tensor] = {}
    if prediction_type in {"x_start", "x_start_epsilon"}:
        component_losses["x_start_mse"] = (x_start_prediction - prepared.x_start).square().mean(dim=-1)
    if prediction_type in {"epsilon", "x_start_epsilon"}:
        component_losses["epsilon_mse"] = (epsilon_prediction - prepared.noise).square().mean(dim=-1)

    component_weights = {
        "x_start_mse": model.x_start_loss_weight,
        "epsilon_mse": model.epsilon_loss_weight,
    }
    active_weight = sum(component_weights[name] for name in component_losses)
    if active_weight <= 0.0:
        raise ValueError("Active label diffusion loss weights must sum to a positive value.")
    per_face_loss = sum(
        component_weights[name] * values for name, values in component_losses.items()
    ) / active_weight
    if class_weights is None:
        loss = per_face_loss.mean()
    else:
        face_weights = class_weights[prepared.labels]
        loss = (per_face_loss * face_weights).sum() / face_weights.sum().clamp_min(1.0e-8)

    with torch.no_grad():
        predictions = x_start_prediction.argmax(dim=-1)
        accuracy = (predictions == prepared.labels).float().mean()
        dsc = dice_loss(x_start_prediction, prepared.labels, model.num_classes, ignore_index=-100)
    metrics = {
        "diffusion_mse": float(loss.detach().cpu()),
        "dice": float(dsc.detach().cpu()),
        "acc": float(accuracy.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    for name, values in component_losses.items():
        metrics[name] = float(values.mean().detach().cpu())
    return loss, metrics


def _stable_initial_noise(
    batch: GraphBatch,
    *,
    num_classes: int,
    base_seed: int,
    sample_index: int,
) -> torch.Tensor:
    graph_ptr = batch.graph_ptr.detach().cpu().tolist()
    chunks: list[torch.Tensor] = []
    for graph_index, sample_id in enumerate(batch.sample_ids):
        face_count = int(graph_ptr[graph_index + 1] - graph_ptr[graph_index])
        digest = hashlib.sha256(f"{base_seed}:{sample_index}:{sample_id}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
        generator = torch.Generator(device=batch.face_cont.device)
        generator.manual_seed(seed)
        chunks.append(
            torch.randn(
                (face_count, num_classes),
                dtype=batch.face_cont.dtype,
                device=batch.face_cont.device,
                generator=generator,
            )
        )
    return torch.cat(chunks, dim=0)


def predict_segmentation_probabilities(
    model: nn.Module,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    target_model = model.module if hasattr(model, "module") else model
    if isinstance(target_model, SegmentationModel):
        return F.softmax(target_model(batch), dim=-1)
    if not isinstance(target_model, DiffusionSegmentationModel):
        raise TypeError(f"Unsupported segmentation model type: {type(target_model).__name__}")

    diffusion_cfg = config["label_diffusion"]
    inference_samples = max(1, int(diffusion_cfg.get("inference_samples", 1)))
    score_temperature = max(float(diffusion_cfg.get("score_temperature", 1.0)), 1.0e-6)
    probabilities = batch.face_cont.new_zeros((batch.face_cont.shape[0], target_model.num_classes))
    for sample_index in range(inference_samples):
        initial_noise = _stable_initial_noise(
            batch,
            num_classes=target_model.num_classes,
            base_seed=int(diffusion_cfg.get("seed", config.get("seed", 42))),
            sample_index=sample_index,
        )
        sampled_x_start = target_model.sample(
            batch,
            initial_noise=initial_noise,
            steps=int(diffusion_cfg.get("sampling_steps", 25)),
            eta=float(diffusion_cfg.get("ddim_eta", 0.0)),
            temperature=float(diffusion_cfg.get("sampling_temperature", 1.0)),
            clip_x_start=bool(diffusion_cfg.get("clip_x_start", True)),
        )
        probabilities.add_(F.softmax(sampled_x_start / score_temperature, dim=-1))
    return probabilities / inference_samples


def dice_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = labels != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)
    logits = logits[valid]
    labels = labels[valid]
    probs = F.softmax(logits, dim=-1)
    target = F.one_hot(labels.clamp(0, num_classes - 1), num_classes=num_classes).float()
    intersection = (probs * target).sum(dim=0)
    denominator = probs.sum(dim=0) + target.sum(dim=0)
    dice = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    return 1.0 - dice.mean()


def segmentation_metrics_from_probabilities(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> dict[str, float]:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires face labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    num_classes = int(config["model"]["num_classes"])
    valid = batch.labels != ignore_index
    if not valid.any():
        return {"acc": 0.0, "dice": 1.0, "f1": 0.0, "weighted_f1": 0.0}
    valid_probabilities = probabilities[valid]
    valid_labels = batch.labels[valid]
    targets = F.one_hot(valid_labels, num_classes=num_classes).float()
    intersection = (valid_probabilities * targets).sum(dim=0)
    denominator = valid_probabilities.sum(dim=0) + targets.sum(dim=0)
    dice = (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
    metrics = segmentation_metrics_from_confusion_matrix(
        segmentation_confusion_matrix(probabilities, batch, config)
    )
    metrics["dice"] = float((1.0 - dice.mean()).detach().cpu())
    return metrics


def segmentation_confusion_matrix(
    probabilities: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires face labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    num_classes = int(config["model"]["num_classes"])
    valid = batch.labels != ignore_index
    if not valid.any():
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=probabilities.device)
    labels = batch.labels[valid]
    predictions = probabilities[valid].argmax(dim=-1)
    return torch.bincount(
        labels * num_classes + predictions,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def segmentation_metrics_from_confusion_matrix(confusion: torch.Tensor) -> dict[str, float]:
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
        # `f1` is the unweighted mean over all configured segmentation classes.
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


def compute_segmentation_loss(
    logits: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires face labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    num_classes = int(config["model"]["num_classes"])
    ce = F.cross_entropy(logits, batch.labels, weight=class_weights, ignore_index=ignore_index)
    dsc = dice_loss(logits, batch.labels, num_classes, ignore_index)
    total = ce + float(config["train"].get("dice_loss_weight", 0.3)) * dsc
    with torch.no_grad():
        valid = batch.labels != ignore_index
        if valid.any():
            pred = logits.argmax(dim=-1)
            acc = (pred[valid] == batch.labels[valid]).float().mean()
        else:
            acc = logits.new_tensor(0.0)
    return total, {
        "ce": float(ce.detach().cpu()),
        "dice": float(dsc.detach().cpu()),
        "acc": float(acc.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
