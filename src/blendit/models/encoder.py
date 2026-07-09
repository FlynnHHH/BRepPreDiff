from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm_msg = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_h: torch.Tensor, edge_h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(node_h)
        else:
            src, dst = edge_index[0], edge_index[1]
            messages = self.message(torch.cat([node_h[src], edge_h], dim=-1))
            aggregated = torch.zeros_like(node_h)
            aggregated.index_add_(0, dst, messages)
            degree = torch.zeros((node_h.shape[0], 1), dtype=node_h.dtype, device=node_h.device)
            degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=node_h.dtype, device=node_h.device))
            aggregated = aggregated / degree.clamp_min(1.0)

        node_h = self.norm_msg(node_h + self.dropout(aggregated))
        node_h = self.norm_ffn(node_h + self.dropout(self.ffn(node_h)))
        return node_h


class BRepGraphEncoder(nn.Module):
    def __init__(
        self,
        face_cont_dim: int,
        edge_cont_dim: int,
        *,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        surface_type_vocab: int,
        edge_type_vocab: int,
        relation_type_vocab: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.surface_type_vocab = surface_type_vocab
        self.edge_type_vocab = edge_type_vocab
        self.relation_type_vocab = relation_type_vocab

        self.face_cont_proj = nn.Linear(face_cont_dim, hidden_dim)
        self.surface_emb = nn.Embedding(surface_type_vocab, hidden_dim)
        self.edge_cont_proj = nn.Linear(edge_cont_dim, hidden_dim)
        self.edge_type_emb = nn.Embedding(edge_type_vocab, hidden_dim)
        self.edge_relation_emb = nn.Embedding(relation_type_vocab, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([GraphMessageLayer(hidden_dim, dropout) for _ in range(num_layers)])

    def embed_edges(
        self,
        edge_cont: torch.Tensor,
        edge_type: torch.Tensor,
        edge_relation: torch.Tensor,
    ) -> torch.Tensor:
        edge_type = edge_type.clamp(0, self.edge_type_vocab - 1)
        edge_relation = edge_relation.clamp(0, self.relation_type_vocab - 1)
        edge_h = (
            self.edge_cont_proj(edge_cont)
            + self.edge_type_emb(edge_type)
            + self.edge_relation_emb(edge_relation)
        )
        return self.edge_norm(edge_h)

    def forward(
        self,
        face_cont: torch.Tensor,
        face_surface_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_cont: torch.Tensor,
        edge_type: torch.Tensor,
        edge_relation: torch.Tensor,
        *,
        time_h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_surface_type = face_surface_type.clamp(0, self.surface_type_vocab - 1)
        node_h = self.face_cont_proj(face_cont) + self.surface_emb(face_surface_type)
        if time_h is not None:
            node_h = node_h + time_h
        node_h = self.input_norm(node_h)
        edge_h = self.embed_edges(edge_cont, edge_type, edge_relation)
        for layer in self.layers:
            node_h = layer(node_h, edge_h, edge_index)
        return node_h, edge_h
