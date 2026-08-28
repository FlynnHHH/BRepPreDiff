from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from blendit.data.graph import GraphBatch
from blendit.models.diffusion import (
    ConditionalDenoisingMLP,
    LabelDiffusionSchedule,
    bipolar_one_hot,
)
from blendit.models.encoder import BRepGraphEncoder


class DownstreamEncoder(nn.Module):
    """Shared B-Rep encoder construction for downstream task models."""

    task: str

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__()
        self.encoder = BRepGraphEncoder.from_config(config, face_cont_dim, edge_cont_dim)

    def encode_faces(self, batch: GraphBatch) -> torch.Tensor:
        face_embeddings, _ = self.encoder(
            batch.face_cont,
            batch.face_surface_type,
            batch.edge_index,
            batch.edge_cont,
            batch.edge_type,
            batch.edge_relation,
            graph_ptr=batch.graph_ptr,
        )
        return face_embeddings


class LabelDiffusionModel(DownstreamEncoder):
    """Task-neutral DiffLoss machinery; subclasses define the prediction tokens."""

    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__(config, face_cont_dim, edge_cont_dim)
        model_cfg = config["model"]
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

    def encode_tokens(self, batch: GraphBatch) -> torch.Tensor:
        raise NotImplementedError

    def token_counts(self, batch: GraphBatch) -> list[int]:
        raise NotImplementedError

    def token_count(self, batch: GraphBatch) -> int:
        return sum(self.token_counts(batch))

    def forward(
        self,
        batch: GraphBatch,
        noisy_labels: torch.Tensor,
        timesteps: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.encode_tokens(batch)[token_indices]
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
        condition = self.encode_tokens(batch)
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


@dataclass(frozen=True)
class LabelDiffusionTrainingBatch:
    x_start: torch.Tensor
    x_t: torch.Tensor
    noise: torch.Tensor
    timesteps: torch.Tensor
    token_indices: torch.Tensor
    labels: torch.Tensor

    @property
    def face_indices(self) -> torch.Tensor:
        """Compatibility alias for older segmentation callers."""
        return self.token_indices


def prepare_label_diffusion_training_batch(
    model: LabelDiffusionModel,
    batch: GraphBatch,
    config: dict[str, Any],
) -> LabelDiffusionTrainingBatch:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires labels.")
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    valid_indices = torch.nonzero(batch.labels != ignore_index, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise ValueError("Fine-tuning batch does not contain any valid labels.")

    token_count = model.token_count(batch)
    if batch.labels.numel() != token_count:
        raise ValueError(
            f"Label count {batch.labels.numel()} does not match {model.task} "
            f"token count {token_count}."
        )

    repeats = max(1, int(config["label_diffusion"].get("noise_samples_per_token", 1)))
    token_indices = valid_indices.repeat(repeats)
    labels = batch.labels[token_indices]
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
        token_indices=token_indices,
        labels=labels,
    )


def label_diffusion_objective(
    prediction: torch.Tensor,
    prepared: LabelDiffusionTrainingBatch,
    model: LabelDiffusionModel,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    x_start_prediction, epsilon_prediction = model.schedule.model_predictions(
        prepared.x_t,
        prepared.timesteps,
        prediction,
        model.prediction_type,
    )

    component_losses: dict[str, torch.Tensor] = {}
    if model.prediction_type in {"x_start", "x_start_epsilon"}:
        component_losses["x_start_mse"] = (
            x_start_prediction - prepared.x_start
        ).square().mean(dim=-1)
    if model.prediction_type in {"epsilon", "x_start_epsilon"}:
        component_losses["epsilon_mse"] = (
            epsilon_prediction - prepared.noise
        ).square().mean(dim=-1)

    component_weights = {
        "x_start_mse": model.x_start_loss_weight,
        "epsilon_mse": model.epsilon_loss_weight,
    }
    active_weight = sum(component_weights[name] for name in component_losses)
    if active_weight <= 0.0:
        raise ValueError("Active label diffusion loss weights must sum to a positive value.")
    # Loss weights are literal coefficients.  In particular, weights 1.0 and
    # 0.5 implement L = L_x_start + 0.5 * L_epsilon rather than a normalized
    # weighted average.
    per_token_loss = sum(
        component_weights[name] * values for name, values in component_losses.items()
    )
    if class_weights is None:
        loss = per_token_loss.mean()
    else:
        token_weights = class_weights[prepared.labels]
        loss = (per_token_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0e-8)

    with torch.no_grad():
        predictions = x_start_prediction.argmax(dim=-1)
        accuracy = (predictions == prepared.labels).float().mean()
    metrics = {
        "diffusion_mse": float(loss.detach().cpu()),
        "acc": float(accuracy.detach().cpu()),
        "total": float(loss.detach().cpu()),
    }
    for name, values in component_losses.items():
        metrics[name] = float(values.mean().detach().cpu())
    return loss, metrics, x_start_prediction


def stable_initial_noise(
    model: LabelDiffusionModel,
    batch: GraphBatch,
    *,
    base_seed: int,
    sample_index: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for sample_id, token_count in zip(batch.sample_ids, model.token_counts(batch)):
        digest = hashlib.sha256(
            f"{base_seed}:{sample_index}:{sample_id}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)
        generator = torch.Generator(device=batch.face_cont.device)
        generator.manual_seed(seed)
        chunks.append(
            torch.randn(
                (token_count, model.num_classes),
                dtype=batch.face_cont.dtype,
                device=batch.face_cont.device,
                generator=generator,
            )
        )
    return torch.cat(chunks, dim=0)


def predict_label_diffusion_probabilities(
    model: LabelDiffusionModel,
    batch: GraphBatch,
    config: dict[str, Any],
) -> torch.Tensor:
    diffusion_cfg = config["label_diffusion"]
    inference_samples = max(1, int(diffusion_cfg.get("inference_samples", 1)))
    score_temperature = max(float(diffusion_cfg.get("score_temperature", 1.0)), 1.0e-6)
    configured_score_bias = diffusion_cfg.get("class_score_bias")
    score_bias = None
    if configured_score_bias is not None:
        if len(configured_score_bias) != model.num_classes:
            raise ValueError(
                "label_diffusion.class_score_bias must contain one value per class: "
                f"expected={model.num_classes} got={len(configured_score_bias)}"
            )
        score_bias = batch.face_cont.new_tensor(configured_score_bias)
    probabilities = batch.face_cont.new_zeros((model.token_count(batch), model.num_classes))
    for sample_index in range(inference_samples):
        initial_noise = stable_initial_noise(
            model,
            batch,
            base_seed=int(diffusion_cfg.get("seed", config.get("seed", 42))),
            sample_index=sample_index,
        )
        sampled_x_start = model.sample(
            batch,
            initial_noise=initial_noise,
            steps=int(diffusion_cfg.get("sampling_steps", 25)),
            eta=float(diffusion_cfg.get("ddim_eta", 0.0)),
            temperature=float(diffusion_cfg.get("sampling_temperature", 1.0)),
            clip_x_start=bool(diffusion_cfg.get("clip_x_start", True)),
        )
        scores = sampled_x_start / score_temperature
        if score_bias is not None:
            scores = scores + score_bias
        probabilities.add_(F.softmax(scores, dim=-1))
    return probabilities / inference_samples
