from __future__ import annotations

import torch

from blendit.data.graph import (
    BRepGraph,
    GlobalFeatureStats,
    typewise_global_standardize_graph_features,
)


def _graph() -> BRepGraph:
    # uv_grid_size=1 gives 17 face channels: 11 base + xyz/normal grid channels.
    face = torch.arange(34, dtype=torch.float32).reshape(2, 17)
    face[:, 4:7] = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    face[:, 14:17] = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    return BRepGraph(
        face_cont=face,
        face_surface_type=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_cont=torch.tensor([[5.0, 1.2, -1.2], [1.0, -0.2, 1.2]]),
        edge_type=torch.zeros(2, dtype=torch.long),
        edge_relation=torch.zeros(2, dtype=torch.long),
        labels=None,
        sample_id="sample",
    )


def test_typewise_global_standardization_preserves_directional_channels():
    stats = GlobalFeatureStats(
        face_mean=torch.ones(17),
        face_std=torch.full((17,), 2.0),
        edge_mean=torch.tensor([1.0, 99.0, 99.0]),
        edge_std=torch.tensor([2.0, 3.0, 3.0]),
        face_count=2,
        edge_count=2,
        uv_grid_size=1,
    )

    result = typewise_global_standardize_graph_features(
        _graph(), stats, uv_grid_size=1
    )

    assert torch.equal(result.face_cont[:, 0], torch.tensor([-0.5, 8.0]))
    assert torch.equal(
        result.face_cont[:, 4:7],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert torch.equal(
        result.face_cont[:, 14:17],
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        result.edge_cont,
        torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.0, 1.0]]),
    )


def test_typewise_global_standardization_rejects_wrong_grid_size():
    stats = GlobalFeatureStats(
        face_mean=torch.zeros(17),
        face_std=torch.ones(17),
        edge_mean=torch.zeros(3),
        edge_std=torch.ones(3),
        face_count=1,
        edge_count=1,
        uv_grid_size=1,
    )

    try:
        typewise_global_standardize_graph_features(_graph(), stats, uv_grid_size=2)
    except ValueError as exc:
        assert "uv_grid_size" in str(exc)
    else:
        raise AssertionError("Expected incompatible uv_grid_size to fail.")
