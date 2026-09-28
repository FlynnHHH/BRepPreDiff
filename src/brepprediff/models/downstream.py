from __future__ import annotations

from typing import Any

import torch
from torch import nn

from brepprediff.data.graph import GraphBatch
from brepprediff.models.encoder import BRepGraphEncoder


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
