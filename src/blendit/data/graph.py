from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch


@dataclass
class BRepGraph:
    face_cont: torch.Tensor
    face_surface_type: torch.Tensor
    edge_index: torch.Tensor
    edge_cont: torch.Tensor
    edge_type: torch.Tensor
    edge_relation: torch.Tensor
    labels: torch.Tensor | None
    sample_id: str

    @property
    def num_faces(self) -> int:
        return int(self.face_cont.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class GraphBatch:
    face_cont: torch.Tensor
    face_surface_type: torch.Tensor
    edge_index: torch.Tensor
    edge_cont: torch.Tensor
    edge_type: torch.Tensor
    edge_relation: torch.Tensor
    labels: torch.Tensor | None
    batch_index: torch.Tensor
    edge_batch_index: torch.Tensor
    graph_ptr: torch.Tensor
    sample_ids: list[str]

    def to(self, device: torch.device | str) -> "GraphBatch":
        labels = self.labels.to(device) if self.labels is not None else None
        return GraphBatch(
            face_cont=self.face_cont.to(device),
            face_surface_type=self.face_surface_type.to(device),
            edge_index=self.edge_index.to(device),
            edge_cont=self.edge_cont.to(device),
            edge_type=self.edge_type.to(device),
            edge_relation=self.edge_relation.to(device),
            labels=labels,
            batch_index=self.batch_index.to(device),
            edge_batch_index=self.edge_batch_index.to(device),
            graph_ptr=self.graph_ptr.to(device),
            sample_ids=self.sample_ids,
        )


def normalize_graph_features(
    graph: BRepGraph,
    eps: float = 1.0e-6,
    *,
    uv_grid_size: int | None = None,
    edge_u_grid_size: int | None = None,
) -> BRepGraph:
    face = graph.face_cont
    edge = graph.edge_cont
    face_mean = face.mean(dim=0, keepdim=True)
    face_std = face.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    face = (face - face_mean) / face_std
    if uv_grid_size is not None:
        face_grid = graph.face_cont[:, 11:].reshape(graph.num_faces, uv_grid_size**2, 7)
        normalized_grid = face[:, 11:].reshape(graph.num_faces, uv_grid_size**2, 7)
        # The trimming mask is categorical and must remain exactly zero or one.
        normalized_grid[:, :, 6] = face_grid[:, :, 6]

    if edge.numel() > 0:
        edge_mean = edge.mean(dim=0, keepdim=True)
        edge_std = edge.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        edge = (edge - edge_mean) / edge_std
        if edge_u_grid_size is not None:
            edge_grid = graph.edge_cont[:, 3:].reshape(-1, edge_u_grid_size, 6)
            normalized_edge_grid = edge[:, 3:].reshape(-1, edge_u_grid_size, 6)
            tangents = edge_grid[:, :, 3:6]
            normalized_edge_grid[:, :, 3:6] = tangents / tangents.norm(
                dim=-1, keepdim=True
            ).clamp_min(eps)
    return BRepGraph(
        face_cont=face,
        face_surface_type=graph.face_surface_type,
        edge_index=graph.edge_index,
        edge_cont=edge,
        edge_type=graph.edge_type,
        edge_relation=graph.edge_relation,
        labels=graph.labels,
        sample_id=graph.sample_id,
    )


@dataclass(frozen=True)
class GlobalFeatureStats:
    face_mean: torch.Tensor
    face_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor
    face_count: int
    edge_count: int
    uv_grid_size: int


def load_global_feature_stats(path: str | Path) -> GlobalFeatureStats:
    stats_path = Path(path)
    with stats_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported feature-statistics version in {stats_path}.")
    return GlobalFeatureStats(
        face_mean=torch.tensor(payload["face_mean"], dtype=torch.float32),
        face_std=torch.tensor(payload["face_std"], dtype=torch.float32),
        edge_mean=torch.tensor(payload["edge_mean"], dtype=torch.float32),
        edge_std=torch.tensor(payload["edge_std"], dtype=torch.float32),
        face_count=int(payload["face_count"]),
        edge_count=int(payload["edge_count"]),
        uv_grid_size=int(payload["uv_grid_size"]),
    )


def typewise_global_standardize_graph_features(
    graph: BRepGraph,
    stats: GlobalFeatureStats,
    *,
    uv_grid_size: int,
    edge_u_grid_size: int | None = None,
    legacy: bool = False,
    eps: float = 1.0e-6,
) -> BRepGraph:
    """Apply global z-scores except to directional/bounded feature channels."""
    if stats.uv_grid_size != uv_grid_size:
        raise ValueError(
            f"Feature statistics use uv_grid_size={stats.uv_grid_size}, "
            f"but the dataset uses {uv_grid_size}."
        )
    if stats.face_mean.numel() != graph.face_cont.shape[-1]:
        raise ValueError("Face feature statistics have an incompatible dimension.")
    if stats.edge_mean.numel() != graph.edge_cont.shape[-1]:
        raise ValueError("Edge feature statistics have an incompatible dimension.")

    face_std = stats.face_std.clamp_min(eps)
    face = (graph.face_cont - stats.face_mean) / face_std
    # Surface normals are directional unit vectors, so a component-wise z-score
    # would destroy their geometry. Re-normalize each xyz vector instead.
    center_normals = graph.face_cont[:, 4:7]
    face[:, 4:7] = center_normals / center_normals.norm(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    grid_channels = 6 if legacy else 7
    grid_input = graph.face_cont[:, 11:].reshape(
        graph.num_faces, uv_grid_size * uv_grid_size, grid_channels
    )
    grid_output = face[:, 11:].reshape(
        graph.num_faces, uv_grid_size * uv_grid_size, grid_channels
    )
    grid_normals = grid_input[:, :, 3:6]
    grid_output[:, :, 3:6] = grid_normals / grid_normals.norm(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    if not legacy:
        grid_output[:, :, 6] = grid_input[:, :, 6]

    edge = (
        graph.edge_cont.clone()
        if legacy
        else (graph.edge_cont - stats.edge_mean) / stats.edge_std.clamp_min(eps)
    )
    if edge.numel() > 0:
        # edge_cont = [log1p(length), angle/pi, normal dot product]. Only the
        # unbounded length channel needs a fitted global z-score.
        edge[:, 0] = (graph.edge_cont[:, 0] - stats.edge_mean[0]) / stats.edge_std[
            0
        ].clamp_min(eps)
        edge[:, 1] = graph.edge_cont[:, 1].clamp(0.0, 1.0)
        edge[:, 2] = graph.edge_cont[:, 2].clamp(-1.0, 1.0)
        resolved_edge_grid_size = edge_u_grid_size or uv_grid_size
        if not legacy:
            edge_grid_input = graph.edge_cont[:, 3:].reshape(
                -1, resolved_edge_grid_size, 6
            )
            edge_grid_output = edge[:, 3:].reshape(-1, resolved_edge_grid_size, 6)
            edge_tangents = edge_grid_input[:, :, 3:6]
            edge_grid_output[:, :, 3:6] = edge_tangents / edge_tangents.norm(
                dim=-1, keepdim=True
            ).clamp_min(eps)

    return BRepGraph(
        face_cont=face,
        face_surface_type=graph.face_surface_type,
        edge_index=graph.edge_index,
        edge_cont=edge,
        edge_type=graph.edge_type,
        edge_relation=graph.edge_relation,
        labels=graph.labels,
        sample_id=graph.sample_id,
    )


def collate_graphs(graphs: list[BRepGraph]) -> GraphBatch:
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch.")

    face_cont = []
    face_surface_type = []
    edge_index = []
    edge_cont = []
    edge_type = []
    edge_relation = []
    labels = []
    batch_index = []
    edge_batch_index = []
    graph_ptr = [0]
    sample_ids = []
    face_offset = 0
    has_labels = all(g.labels is not None for g in graphs)

    for graph_id, graph in enumerate(graphs):
        n_faces = graph.num_faces
        face_cont.append(graph.face_cont)
        face_surface_type.append(graph.face_surface_type)
        if graph.num_edges > 0:
            edge_index.append(graph.edge_index + face_offset)
            edge_cont.append(graph.edge_cont)
            edge_type.append(graph.edge_type)
            edge_relation.append(graph.edge_relation)
            edge_batch_index.append(torch.full((graph.num_edges,), graph_id, dtype=torch.long))
        if has_labels and graph.labels is not None:
            labels.append(graph.labels)
        batch_index.append(torch.full((n_faces,), graph_id, dtype=torch.long))
        face_offset += n_faces
        graph_ptr.append(face_offset)
        sample_ids.append(graph.sample_id)

    if edge_index:
        batched_edge_index = torch.cat(edge_index, dim=1)
        batched_edge_cont = torch.cat(edge_cont, dim=0)
        batched_edge_type = torch.cat(edge_type, dim=0)
        batched_edge_relation = torch.cat(edge_relation, dim=0)
        batched_edge_batch_index = torch.cat(edge_batch_index, dim=0)
    else:
        edge_dim = graphs[0].edge_cont.shape[-1]
        batched_edge_index = torch.empty((2, 0), dtype=torch.long)
        batched_edge_cont = torch.empty((0, edge_dim), dtype=torch.float32)
        batched_edge_type = torch.empty((0,), dtype=torch.long)
        batched_edge_relation = torch.empty((0,), dtype=torch.long)
        batched_edge_batch_index = torch.empty((0,), dtype=torch.long)

    batched_labels = torch.cat(labels, dim=0) if has_labels else None
    return GraphBatch(
        face_cont=torch.cat(face_cont, dim=0),
        face_surface_type=torch.cat(face_surface_type, dim=0),
        edge_index=batched_edge_index,
        edge_cont=batched_edge_cont,
        edge_type=batched_edge_type,
        edge_relation=batched_edge_relation,
        labels=batched_labels,
        batch_index=torch.cat(batch_index, dim=0),
        edge_batch_index=batched_edge_batch_index,
        graph_ptr=torch.tensor(graph_ptr, dtype=torch.long),
        sample_ids=sample_ids,
    )


def graph_from_arrays(arrays: dict[str, Any], sample_id: str) -> BRepGraph:
    labels_arr = arrays.get("labels")
    labels = None
    if labels_arr is not None and np.asarray(labels_arr).size > 0:
        labels = torch.as_tensor(labels_arr, dtype=torch.long)
    return BRepGraph(
        face_cont=torch.as_tensor(arrays["face_cont"], dtype=torch.float32),
        face_surface_type=torch.as_tensor(arrays["face_surface_type"], dtype=torch.long),
        edge_index=torch.as_tensor(arrays["edge_index"], dtype=torch.long),
        edge_cont=torch.as_tensor(arrays["edge_cont"], dtype=torch.float32),
        edge_type=torch.as_tensor(arrays["edge_type"], dtype=torch.long),
        edge_relation=torch.as_tensor(arrays["edge_relation"], dtype=torch.long),
        labels=labels,
        sample_id=sample_id,
    )


def load_graph_npz(
    path: str | Path,
    sample_id: str,
    *,
    load_labels: bool = True,
) -> BRepGraph:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            key: data[key]
            for key in data.files
            if load_labels or key != "labels"
        }
    return graph_from_arrays(arrays, sample_id)


def save_graph_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f".{out_path.name}.",
        suffix=".tmp.npz",
        dir=out_path.parent,
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
