from __future__ import annotations

import torch

from blendit.data.graph import (
    BRepGraph,
    GlobalFeatureStats,
    normalize_graph_features,
    typewise_global_standardize_graph_features,
)


def _graph() -> BRepGraph:
    # uv_grid_size=1 gives 18 face channels: 11 base + xyz/normal/mask grid channels.
    face = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    face[:, 4:7] = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    face[:, 14:17] = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    face[:, 17] = torch.tensor([1.0, 0.0])
    return BRepGraph(
        face_cont=face,
        face_surface_type=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_cont=torch.tensor(
            [
                [5.0, 1.2, -1.2, 2.0, 4.0, 6.0, 3.0, 0.0, 0.0],
                [1.0, -0.2, 1.2, 1.0, 3.0, 5.0, 0.0, 0.0, 0.0],
            ]
        ),
        edge_type=torch.zeros(2, dtype=torch.long),
        edge_relation=torch.zeros(2, dtype=torch.long),
        labels=None,
        sample_id="sample",
    )


def test_typewise_global_standardization_preserves_directional_channels():
    stats = GlobalFeatureStats(
        face_mean=torch.ones(18),
        face_std=torch.full((18,), 2.0),
        edge_mean=torch.tensor([1.0, 99.0, 99.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0]),
        edge_std=torch.tensor([2.0, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
        face_count=2,
        edge_count=2,
        uv_grid_size=1,
    )

    result = typewise_global_standardize_graph_features(
        _graph(), stats, uv_grid_size=1, edge_u_grid_size=1
    )

    assert torch.equal(result.face_cont[:, 0], torch.tensor([-0.5, 8.5]))
    assert torch.equal(
        result.face_cont[:, 4:7],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert torch.equal(
        result.face_cont[:, 14:17],
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
    )
    assert torch.equal(result.face_cont[:, 17], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        result.edge_cont[:, :3],
        torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.0, 1.0]]),
    )
    assert torch.equal(
        result.edge_cont[:, 3:6],
        torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]),
    )
    assert torch.equal(
        result.edge_cont[:, 6:9],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_per_graph_standardization_preserves_mask_and_unit_tangents():
    result = normalize_graph_features(
        _graph(), uv_grid_size=1, edge_u_grid_size=1
    )

    assert torch.equal(result.face_cont[:, 17], torch.tensor([1.0, 0.0]))
    assert torch.equal(
        result.edge_cont[:, 6:9],
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_typewise_global_standardization_rejects_wrong_grid_size():
    stats = GlobalFeatureStats(
        face_mean=torch.zeros(18),
        face_std=torch.ones(18),
        edge_mean=torch.zeros(9),
        edge_std=torch.ones(9),
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
