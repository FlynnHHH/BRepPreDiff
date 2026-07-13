from __future__ import annotations

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


class DiffusionPretrainModel(nn.Module):
    def __init__(self, config: dict[str, Any], face_cont_dim: int, edge_cont_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        brep_cfg = config["brep"]
        hidden_dim = int(model_cfg["hidden_dim"])
        time_dim = int(model_cfg.get("time_dim", hidden_dim))
        self.num_classes = int(model_cfg["num_classes"])
        self.time_dim = time_dim

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
        self.coarse_label_head = MLP(hidden_dim, hidden_dim, self.num_classes, float(model_cfg["dropout"]))

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
            "coarse_label_logits": self.coarse_label_head(node_h),
        }

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
    ignore_index = int(config["train"].get("ignore_index", -100))
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

    if batch.labels is not None:
        soft_targets, valid = make_soft_targets(batch.labels, config)
        if valid.any():
            losses["coarse_label"] = soft_cross_entropy(outputs["coarse_label_logits"][valid], soft_targets[valid])
        else:
            losses["coarse_label"] = batch.face_cont.new_tensor(0.0)
    else:
        losses["coarse_label"] = batch.face_cont.new_tensor(0.0)

    noise_loss = losses["face_noise"] + losses["edge_noise"]
    recon_loss = losses["face_recon"] + losses["edge_recon"]
    categorical_loss = losses["surface"] + losses["edge_type"]
    relation_loss = losses["relation"]
    coarse_loss = losses["coarse_label"]

    total = (
        float(diff_cfg["noise_loss_weight"]) * noise_loss
        + float(diff_cfg["recon_loss_weight"]) * recon_loss
        + float(diff_cfg["categorical_loss_weight"]) * categorical_loss
        + float(diff_cfg["relation_loss_weight"]) * relation_loss
        + float(diff_cfg["coarse_label_loss_weight"]) * coarse_loss
    )
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics
