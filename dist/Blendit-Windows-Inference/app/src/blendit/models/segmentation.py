from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from blendit.data.graph import GraphBatch
from blendit.models.encoder import BRepGraphEncoder, MLP


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


def compute_segmentation_loss(
    logits: torch.Tensor,
    batch: GraphBatch,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if batch.labels is None:
        raise ValueError("Fine-tuning requires face labels.")
    ignore_index = int(config["train"].get("ignore_index", -100))
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
