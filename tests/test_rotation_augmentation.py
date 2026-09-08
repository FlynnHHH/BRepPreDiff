from __future__ import annotations

import torch

from brepprediff.data.graph import BRepGraph, rotate_graph_geometry


def _graph() -> BRepGraph:
    face = torch.zeros(2, 18)
    face[:, 1:4] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    face[:, 4:7] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    face[:, 11:14] = face[:, 1:4]
    face[:, 14:17] = face[:, 4:7]
    face[:, 17] = 1.0
    edge = torch.zeros(2, 9)
    edge[:, 3:6] = face[:, 1:4]
    edge[:, 6:9] = face[:, 4:7]
    return BRepGraph(
        face_cont=face,
        face_surface_type=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_cont=edge,
        edge_type=torch.zeros(2, dtype=torch.long),
        edge_relation=torch.zeros(2, dtype=torch.long),
        labels=torch.tensor([1, 2]),
        sample_id="rotation",
    )


def test_rotation_preserves_labels_distances_and_non_geometric_channels():
    graph = _graph()
    torch.manual_seed(7)
    rotated = rotate_graph_geometry(graph, uv_grid_size=1, edge_u_grid_size=1)

    assert rotated.labels is graph.labels
    assert torch.equal(rotated.face_cont[:, 0], graph.face_cont[:, 0])
    assert torch.equal(rotated.face_cont[:, 7:11], graph.face_cont[:, 7:11])
    assert torch.equal(rotated.face_cont[:, 17], graph.face_cont[:, 17])
    assert torch.allclose(
        torch.cdist(rotated.face_cont[:, 1:4], rotated.face_cont[:, 1:4]),
        torch.cdist(graph.face_cont[:, 1:4], graph.face_cont[:, 1:4]),
        atol=1e-5,
    )
    assert torch.allclose(rotated.face_cont[:, 4:7].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert not torch.equal(rotated.face_cont[:, 1:7], graph.face_cont[:, 1:7])
