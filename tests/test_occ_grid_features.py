from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from blendit.brep.occ_extractor import OccBRepExtractor
from blendit.config import feature_dims
from blendit.data.dataset import _project_occ_grid_v2_to_legacy
from blendit.data.graph import BRepGraph


class _Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.xyz = (x, y, z)

    def X(self) -> float:
        return self.xyz[0]

    def Y(self) -> float:
        return self.xyz[1]

    def Z(self) -> float:
        return self.xyz[2]


class _Curve:
    def __init__(self, _edge) -> None:
        pass

    def FirstParameter(self) -> float:
        return 0.0

    def LastParameter(self) -> float:
        return 2.0

    def Value(self, parameter: float) -> _Point:
        return _Point(parameter, parameter**2, 0.0)

    def DN(self, _parameter: float, _order: int) -> _Point:
        return _Point(3.0, 4.0, 0.0)


def test_feature_dims_include_face_trim_mask_and_edge_u_grid():
    config = {"brep": {"uv_grid_size": 2, "edge_u_grid_size": 3}}

    assert feature_dims(config) == (11 + 4 * 7, 3 + 3 * 6)


def test_feature_dims_support_legacy_cached_features():
    config = {"brep": {"uv_grid_size": 2, "feature_schema": "legacy"}}

    assert feature_dims(config) == (11 + 4 * 6, 3)


def test_occ_grid_v2_cache_projects_exactly_to_legacy_channels():
    face_base = torch.arange(11, dtype=torch.float32).reshape(1, 11)
    face_grid = torch.arange(28, dtype=torch.float32).reshape(1, 4, 7)
    graph = BRepGraph(
        face_cont=torch.cat([face_base, face_grid.reshape(1, -1)], dim=-1),
        face_surface_type=torch.zeros(1, dtype=torch.long),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_cont=torch.arange(21, dtype=torch.float32).reshape(1, 21),
        edge_type=torch.zeros(1, dtype=torch.long),
        edge_relation=torch.zeros(1, dtype=torch.long),
        labels=None,
        sample_id="sample",
    )

    legacy = _project_occ_grid_v2_to_legacy(graph, uv_grid_size=2)

    assert torch.equal(legacy.face_cont[:, :11], face_base)
    assert torch.equal(legacy.face_cont[:, 11:].reshape(1, 4, 6), face_grid[:, :, :6])
    assert torch.equal(legacy.edge_cont, graph.edge_cont[:, :3])


def test_edge_u_grid_samples_xyz_and_unit_tangent():
    extractor = OccBRepExtractor.__new__(OccBRepExtractor)
    extractor.occ = SimpleNamespace(BRepAdaptor_Curve=_Curve)
    extractor.edge_grid_size = 3
    extractor.precision = 1.0e-6

    grid = np.asarray(extractor._sample_edge_grid(object()), dtype=np.float32).reshape(3, 6)

    np.testing.assert_allclose(
        grid[:, :3],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 4.0, 0.0]],
    )
    np.testing.assert_allclose(grid[:, 3:], [[0.6, 0.8, 0.0]] * 3)


class _Classifier:
    def __init__(self, _face, uv, _precision) -> None:
        self.uv = uv

    def State(self) -> int:
        return 1 if self.uv[0] <= 0.5 else 3


def test_uv_trim_classifier_includes_only_in_or_on_samples():
    extractor = OccBRepExtractor.__new__(OccBRepExtractor)
    extractor.precision = 1.0e-6
    extractor.occ = SimpleNamespace(
        gp_Pnt2d=lambda u, v: (u, v),
        BRepClass_FaceClassifier=_Classifier,
        TopAbs_IN=1,
        TopAbs_ON=2,
    )

    assert extractor._uv_is_on_face(object(), 0.25, 0.5)
    assert not extractor._uv_is_on_face(object(), 0.75, 0.5)
