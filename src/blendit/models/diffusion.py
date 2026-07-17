from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from blendit.data.graph import GraphBatch
from blendit.models.encoder import BRepGraphEncoder, MLP, sinusoidal_timestep_embedding


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        scale_clean = self.sqrt_alpha_bar[t].unsqueeze(-1)
        scale_noise = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return scale_clean * x_start + scale_noise * noise, noise


def bipolar_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encode class ids as vertices in {-1, +1}^C."""
    return F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32).mul_(2.0).sub_(1.0)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + s) / (1.0 + s)) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1.0e-8, 0.999).to(dtype=torch.float32)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    extracted = values.gather(0, timesteps)
    return extracted.reshape((timesteps.shape[0],) + (1,) * (reference.ndim - 1))


class LabelDiffusionSchedule(nn.Module):
    """Forward label noising and deterministic/stochastic DDIM sampling."""

    def __init__(
        self,
        timesteps: int,
        *,
        noise_schedule: str = "cosine",
        beta_start: float = 1.0e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if noise_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif noise_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unsupported label diffusion noise schedule: {noise_schedule!r}")

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.noise_schedule = noise_schedule
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar), persistent=False)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        scale_clean = _extract(self.sqrt_alpha_bar, timesteps, x_start)
        scale_noise = _extract(self.sqrt_one_minus_alpha_bar, timesteps, x_start)
        return scale_clean * x_start + scale_noise * noise, noise

    def predict_x_start(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        scale_clean = _extract(self.sqrt_alpha_bar, timesteps, x_t)
        scale_noise = _extract(self.sqrt_one_minus_alpha_bar, timesteps, x_t)
        return (x_t - scale_noise * epsilon) / scale_clean.clamp_min(1.0e-8)

    def predict_epsilon(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        x_start: torch.Tensor,
    ) -> torch.Tensor:
        scale_clean = _extract(self.sqrt_alpha_bar, timesteps, x_t)
        scale_noise = _extract(self.sqrt_one_minus_alpha_bar, timesteps, x_t)
        return (x_t - scale_clean * x_start) / scale_noise.clamp_min(1.0e-8)

    def model_predictions(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        prediction: torch.Tensor,
        prediction_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a configured model output into consistent x_0 and epsilon tensors."""
        if prediction_type == "epsilon":
            epsilon = prediction
            x_start = self.predict_x_start(x_t, timesteps, epsilon)
        elif prediction_type == "x_start":
            x_start = prediction
            epsilon = self.predict_epsilon(x_t, timesteps, x_start)
        elif prediction_type == "x_start_epsilon":
            if prediction.shape[-1] != x_t.shape[-1] * 2:
                raise ValueError(
                    "x_start_epsilon prediction must have twice the label dimension; "
                    f"got prediction={tuple(prediction.shape)} x_t={tuple(x_t.shape)}"
                )
            x_start, epsilon = prediction.chunk(2, dim=-1)
        else:
            raise ValueError(f"Unsupported label diffusion prediction_type: {prediction_type!r}")
        return x_start, epsilon

    def sampling_timesteps(self, steps: int, device: torch.device) -> torch.Tensor:
        if steps <= 0:
            raise ValueError(f"sampling steps must be positive, got {steps}")
        steps = min(int(steps), self.timesteps)
        return torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long()

    def sample(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        condition: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        steps: int,
        eta: float = 0.0,
        temperature: float = 1.0,
        clip_x_start: bool = True,
        prediction_type: str = "epsilon",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if initial_noise is None:
            x_t = torch.randn(
                (condition.shape[0], denoiser.in_channels),
                dtype=condition.dtype,
                device=condition.device,
                generator=generator,
            )
        else:
            x_t = initial_noise.to(dtype=condition.dtype, device=condition.device)
        x_t = x_t * float(temperature)

        sampling_times = self.sampling_timesteps(steps, condition.device)
        for index, timestep in enumerate(sampling_times):
            t = torch.full((condition.shape[0],), int(timestep.item()), device=condition.device, dtype=torch.long)
            prediction = denoiser(x_t, t, condition)
            x_start, epsilon = self.model_predictions(x_t, t, prediction, prediction_type)
            if clip_x_start:
                x_start = x_start.clamp(-1.0, 1.0)

            if index + 1 == sampling_times.numel():
                x_t = x_start
                continue

            previous_timestep = sampling_times[index + 1]
            alpha_t = self.alpha_bar[timestep]
            alpha_previous = self.alpha_bar[previous_timestep]
            sigma = float(eta) * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp_min(0.0)
                * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
            )
            direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0)) * epsilon
            if float(eta) > 0.0:
                step_noise = torch.randn(
                    x_t.shape,
                    dtype=x_t.dtype,
                    device=x_t.device,
                    generator=generator,
                )
            else:
                step_noise = torch.zeros_like(x_t)
            x_t = torch.sqrt(alpha_previous) * x_start + direction + sigma * step_noise
        return x_t


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal_embedding(timesteps, self.frequency_dim))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class AdaLNResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1.0e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 3))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.norm(x), shift, scale))


class ConditionalDenoisingMLP(nn.Module):
    """Small MAR-style AdaLN MLP predicting a configured target for one label token."""

    def __init__(
        self,
        in_channels: int,
        condition_dim: int,
        *,
        width: int,
        depth: int,
        dropout: float = 0.0,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels if out_channels is not None else in_channels)
        self.input_projection = nn.Linear(in_channels, width)
        self.condition_projection = nn.Linear(condition_dim, width)
        self.time_embedding = TimestepEmbedder(width)
        self.blocks = nn.ModuleList([AdaLNResidualBlock(width, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1.0e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
        self.output_projection = nn.Linear(width, self.out_channels)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.time_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedding.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(x_t)
        adaptive_condition = self.time_embedding(timesteps) + self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden, adaptive_condition)
        shift, scale = self.final_modulation(adaptive_condition).chunk(2, dim=-1)
        hidden = _modulate(self.final_norm(hidden), shift, scale)
        return self.output_projection(hidden)


class DiffusionPretrainModel(nn.Module):
    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        brep_cfg = config["brep"]
        hidden_dim = int(model_cfg["hidden_dim"])
        time_dim = int(model_cfg.get("time_dim", hidden_dim))
        self.num_classes = int(model_cfg["num_classes"])
        self.time_dim = time_dim
        self.use_coarse_label_head = bool(model_cfg.get("use_coarse_label_head", True))

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
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
        self.face_noise_head = MLP(hidden_dim, hidden_dim, face_cont_dim, float(model_cfg["dropout"]))
        self.face_recon_head = MLP(hidden_dim, hidden_dim, face_cont_dim, float(model_cfg["dropout"]))
        self.surface_head = MLP(hidden_dim, hidden_dim, int(brep_cfg["surface_type_vocab"]), float(model_cfg["dropout"]))
        self.coarse_label_head = (
            MLP(hidden_dim, hidden_dim, self.num_classes, float(model_cfg["dropout"]))
            if self.use_coarse_label_head
            else None
        )

        edge_context_dim = hidden_dim * 3
        self.edge_context = MLP(edge_context_dim, hidden_dim, hidden_dim, float(model_cfg["dropout"]))
        self.edge_noise_head = MLP(hidden_dim, hidden_dim, edge_cont_dim, float(model_cfg["dropout"]))
        self.edge_recon_head = MLP(hidden_dim, hidden_dim, edge_cont_dim, float(model_cfg["dropout"]))
        self.edge_type_head = MLP(hidden_dim, hidden_dim, int(brep_cfg["edge_type_vocab"]), float(model_cfg["dropout"]))
        self.relation_head = MLP(hidden_dim, hidden_dim, int(brep_cfg["relation_type_vocab"]), float(model_cfg["dropout"]))

    def forward(
        self,
        batch: GraphBatch,
        face_cont_noisy: torch.Tensor,
        edge_cont_noisy: torch.Tensor,
        face_timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        time_h = self.time_mlp(sinusoidal_timestep_embedding(face_timesteps, self.time_dim))
        node_h, edge_h = self.encoder(
            face_cont_noisy,
            batch.face_surface_type,
            batch.edge_index,
            edge_cont_noisy,
            batch.edge_type,
            batch.edge_relation,
            time_h=time_h,
        )
        outputs = {
            "face_noise": self.face_noise_head(node_h),
            "face_recon": self.face_recon_head(node_h),
            "surface_logits": self.surface_head(node_h),
        }
        if self.coarse_label_head is not None:
            outputs["coarse_label_logits"] = self.coarse_label_head(node_h)

        if batch.edge_index.numel() == 0:
            outputs.update(
                {
                    "edge_noise": edge_cont_noisy.new_empty((0, edge_cont_noisy.shape[-1])),
                    "edge_recon": edge_cont_noisy.new_empty((0, edge_cont_noisy.shape[-1])),
                    "edge_type_logits": edge_cont_noisy.new_empty((0, self.edge_type_head.net[-1].out_features)),
                    "relation_logits": edge_cont_noisy.new_empty((0, self.relation_head.net[-1].out_features)),
                }
            )
            return outputs

        src, dst = batch.edge_index[0], batch.edge_index[1]
        edge_context = self.edge_context(torch.cat([node_h[src], node_h[dst], edge_h], dim=-1))
        outputs.update(
            {
                "edge_noise": self.edge_noise_head(edge_context),
                "edge_recon": self.edge_recon_head(edge_context),
                "edge_type_logits": self.edge_type_head(edge_context),
                "relation_logits": self.relation_head(edge_context),
            }
        )
        return outputs


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_prob).sum(dim=-1).mean()


def make_soft_targets(labels: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    num_classes = int(config["model"]["num_classes"])
    ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
    mapping = config["labels"].get("coarse_soft_targets", {})
    soft = torch.zeros((labels.shape[0], num_classes), dtype=torch.float32, device=labels.device)
    valid = labels != ignore_index
    for cls in range(num_classes):
        values = mapping.get(cls, mapping.get(str(cls)))
        if values is None:
            values = [0.0] * num_classes
            values[cls] = 1.0
        cls_mask = valid & (labels == cls)
        if cls_mask.any():
            soft[cls_mask] = torch.tensor(values, dtype=torch.float32, device=labels.device)
    return soft, valid


def compute_pretrain_loss(
    outputs: dict[str, torch.Tensor],
    batch: GraphBatch,
    *,
    face_noise: torch.Tensor,
    edge_noise: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    diff_cfg = config["diffusion"]
    losses: dict[str, torch.Tensor] = {}

    losses["face_noise"] = F.mse_loss(outputs["face_noise"], face_noise)
    losses["face_recon"] = F.l1_loss(outputs["face_recon"], batch.face_cont)
    losses["surface"] = F.cross_entropy(outputs["surface_logits"], batch.face_surface_type)

    if batch.edge_cont.numel() > 0:
        losses["edge_noise"] = F.mse_loss(outputs["edge_noise"], edge_noise)
        losses["edge_recon"] = F.l1_loss(outputs["edge_recon"], batch.edge_cont)
        losses["edge_type"] = F.cross_entropy(outputs["edge_type_logits"], batch.edge_type)
        losses["relation"] = F.cross_entropy(outputs["relation_logits"], batch.edge_relation)
    else:
        zero = batch.face_cont.new_tensor(0.0)
        losses["edge_noise"] = zero
        losses["edge_recon"] = zero
        losses["edge_type"] = zero
        losses["relation"] = zero

    if "coarse_label_logits" in outputs and batch.labels is not None:
        soft_targets, valid = make_soft_targets(batch.labels, config)
        if valid.any():
            losses["coarse_label"] = soft_cross_entropy(outputs["coarse_label_logits"][valid], soft_targets[valid])
        else:
            losses["coarse_label"] = batch.face_cont.new_tensor(0.0)
    elif "coarse_label_logits" in outputs:
        losses["coarse_label"] = batch.face_cont.new_tensor(0.0)

    noise_loss = losses["face_noise"] + losses["edge_noise"]
    recon_loss = losses["face_recon"] + losses["edge_recon"]
    categorical_loss = losses["surface"] + losses["edge_type"]
    relation_loss = losses["relation"]

    total = (
        float(diff_cfg["noise_loss_weight"]) * noise_loss
        + float(diff_cfg["recon_loss_weight"]) * recon_loss
        + float(diff_cfg["categorical_loss_weight"]) * categorical_loss
        + float(diff_cfg["relation_loss_weight"]) * relation_loss
    )
    if "coarse_label" in losses:
        total = total + float(diff_cfg["coarse_label_loss_weight"]) * losses["coarse_label"]
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics
