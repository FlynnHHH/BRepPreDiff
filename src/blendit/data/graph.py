from __future__ import annotations

from dataclasses import dataclass
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


def normalize_graph_features(graph: BRepGraph, eps: float = 1.0e-6) -> BRepGraph:
    face = graph.face_cont
    edge = graph.edge_cont
    face_mean = face.mean(dim=0, keepdim=True)
    face_std = face.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    face = (face - face_mean) / face_std

    if edge.numel() > 0:
        edge_mean = edge.mean(dim=0, keepdim=True)
        edge_std = edge.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        edge = (edge - edge_mean) / edge_std
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


def load_graph_npz(path: str | Path, sample_id: str) -> BRepGraph:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
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
