from __future__ import annotations

import math
from typing import Any

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


class FeedForward(nn.Module):
    """SwiGLU feed-forward block used by the attention encoder variants."""

    def __init__(self, hidden_dim: int, dropout: float, *, expansion: int = 4) -> None:
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.input_projection = nn.Linear(hidden_dim, inner_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(inner_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values, gates = self.input_projection(inputs).chunk(2, dim=-1)
        hidden = values * torch.nn.functional.silu(gates)
        return self.output_projection(self.dropout(hidden))


def segmented_softmax(
    logits: torch.Tensor,
    indexes: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Numerically stable softmax over rows sharing the same segment index."""
    maxima = logits.new_full((num_segments, logits.shape[-1]), float("-inf"))
    expanded_indexes = indexes.unsqueeze(-1).expand_as(logits)
    maxima.scatter_reduce_(
        0,
        expanded_indexes,
        logits,
        reduce="amax",
        include_self=True,
    )
    exponentials = torch.exp(logits - maxima[indexes])
    denominators = logits.new_zeros((num_segments, logits.shape[-1]))
    denominators.index_add_(0, indexes, exponentials)
    return exponentials / denominators[indexes].clamp_min(1.0e-8)


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

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return node_h, edge_h


class EdgeAttentionLayer(nn.Module):
    """Sparse multi-head attention with edge-conditioned keys, values and logits."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        *,
        num_heads: int,
        update_edges: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.update_edges = update_edges
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_key = nn.Linear(hidden_dim, hidden_dim)
        self.edge_value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(hidden_dim, num_heads)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.edge_update = (
            MLP(hidden_dim * 3, hidden_dim * 2, hidden_dim, dropout)
            if update_edges
            else None
        )
        self.edge_norm = nn.LayerNorm(hidden_dim) if update_edges else None

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            attention_output = torch.zeros_like(node_h)
        else:
            src, dst = edge_index[0], edge_index[1]
            normalized = self.norm_attention(node_h)
            queries = self.query(normalized).view(
                -1, self.num_heads, self.head_dim
            )[dst]
            keys = (
                self.key(normalized)[src] + self.edge_key(edge_h)
            ).view(-1, self.num_heads, self.head_dim)
            values = (
                self.value(normalized)[src] + self.edge_value(edge_h)
            ).view(-1, self.num_heads, self.head_dim)
            logits = (queries * keys).sum(dim=-1) / math.sqrt(self.head_dim)
            logits = logits + self.edge_bias(edge_h)
            weights = segmented_softmax(logits, dst, node_h.shape[0])
            aggregated = node_h.new_zeros(
                (node_h.shape[0], self.num_heads, self.head_dim)
            )
            aggregated.index_add_(0, dst, values * weights.unsqueeze(-1))
            attention_output = self.output_projection(aggregated.flatten(1))

        node_h = node_h + self.dropout(attention_output)
        node_h = node_h + self.dropout(self.ffn(self.norm_ffn(node_h)))
        if self.edge_update is not None and edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_delta = self.edge_update(
                torch.cat([node_h[src], node_h[dst], edge_h], dim=-1)
            )
            assert self.edge_norm is not None
            edge_h = self.edge_norm(edge_h + self.dropout(edge_delta))
        return node_h, edge_h


class BRepGraphEncoder(nn.Module):
    SUPPORTED_TYPES = {"ffn", "message_passing", "edge_update_attention"}
    ENCODER_TYPE_ALIASES = {"ffn": "message_passing"}

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
        encoder_type: str = "message_passing",
        num_heads: int = 4,
        grid_encoder: bool = False,
        multiscale_context: bool = False,
    ) -> None:
        super().__init__()
        encoder_type = self.ENCODER_TYPE_ALIASES.get(encoder_type, encoder_type)
        if encoder_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported encoder_type {encoder_type!r}; "
                f"expected one of {sorted(self.SUPPORTED_TYPES)}."
            )
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
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
        layer_type = GraphMessageLayer if encoder_type == "message_passing" else None
        self.layers = nn.ModuleList(
            [
                layer_type(hidden_dim, dropout)
                if layer_type is not None
                else EdgeAttentionLayer(
                    hidden_dim,
                    dropout,
                    num_heads=num_heads,
                    update_edges=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.grid_encoder = grid_encoder
        self.multiscale_context = multiscale_context
        if grid_encoder:
            self.grid_size = math.isqrt((face_cont_dim - 11) // 7)
            if face_cont_dim != 11 + self.grid_size**2 * 7 or (edge_cont_dim - 3) % 6:
                raise ValueError("Grid encoder requires OCC-grid-v2 features.")
            self.face_grid_net = nn.Sequential(
                nn.Conv2d(7, 32, 3, padding=1), nn.SiLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hidden_dim),
            )
            self.edge_grid_net = nn.Sequential(
                nn.Conv1d(6, 32, 3, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, hidden_dim),
            )
        if multiscale_context:
            self.scale_logits = nn.Parameter(torch.zeros(num_layers + 1))
            self.context_projection = MLP(hidden_dim * 3, hidden_dim, hidden_dim, dropout)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        face_cont_dim: int,
        edge_cont_dim: int,
    ) -> "BRepGraphEncoder":
        model_cfg = config["model"]
        brep_cfg = config["brep"]
        return cls(
            face_cont_dim,
            edge_cont_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            surface_type_vocab=int(brep_cfg["surface_type_vocab"]),
            edge_type_vocab=int(brep_cfg["edge_type_vocab"]),
            relation_type_vocab=int(brep_cfg["relation_type_vocab"]),
            encoder_type=str(model_cfg.get("encoder_type", "message_passing")),
            num_heads=int(model_cfg.get("num_heads", 4)),
            grid_encoder=bool(model_cfg.get("grid_encoder", False)),
            multiscale_context=bool(model_cfg.get("multiscale_context", False)),
        )

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
        graph_ptr: torch.Tensor | None = None,
        face_type_mask: torch.Tensor | None = None,
        edge_type_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        face_surface_type = face_surface_type.clamp(0, self.surface_type_vocab - 1)
        node_h = self.face_cont_proj(face_cont) + self.surface_emb(face_surface_type)
        if face_type_mask is not None:
            node_h = node_h - self.surface_emb(face_surface_type) * face_type_mask.unsqueeze(-1)
        if self.grid_encoder:
            grids = face_cont[:, 11:].reshape(-1, self.grid_size, self.grid_size, 7)
            node_h = node_h + self.face_grid_net(grids.permute(0, 3, 1, 2))
        if time_h is not None:
            node_h = node_h + time_h
        node_h = self.input_norm(node_h)
        edge_h = self.embed_edges(edge_cont, edge_type, edge_relation)
        if edge_type_mask is not None:
            # Replace both categorical contributions before LayerNorm.
            edge_input = self.edge_cont_proj(edge_cont)
            categorical = self.edge_type_emb(edge_type) + self.edge_relation_emb(edge_relation)
            edge_h = self.edge_norm(edge_input + categorical * (~edge_type_mask).unsqueeze(-1))
        if self.grid_encoder and edge_cont.shape[0]:
            edge_h = edge_h + self.edge_grid_net(edge_cont[:, 3:].reshape(edge_cont.shape[0], -1, 6).transpose(1, 2))
        scales = [node_h] if self.multiscale_context else None
        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h, edge_index)
            if scales is not None:
                scales.append(node_h)
        if scales is not None:
            node_h = (torch.stack(scales) * self.scale_logits.softmax(0)[:, None, None]).sum(0)
            if graph_ptr is None:
                raise ValueError("Multiscale context requires graph_ptr.")
            counts = graph_ptr[1:] - graph_ptr[:-1]
            indexes = torch.repeat_interleave(torch.arange(counts.numel(), device=node_h.device), counts)
            mean = node_h.new_zeros((counts.numel(), node_h.shape[-1]))
            mean.index_add_(0, indexes, node_h)
            mean = mean / counts[:, None].clamp_min(1)
            maximum = torch.full_like(mean, -torch.inf)
            maximum.scatter_reduce_(0, indexes[:, None].expand_as(node_h), node_h, reduce="amax", include_self=True)
            node_h = node_h + self.context_projection(torch.cat((node_h, mean[indexes], maximum[indexes]), -1))
        return node_h, edge_h
