from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _parse_label_map(raw_mapping: Any) -> dict[int, int] | None:
    if raw_mapping is None:
        return None
    if isinstance(raw_mapping, str):
        stripped = raw_mapping.strip()
        if not stripped:
            return None
        mapping: dict[int, int] = {}
        for item in stripped.split(","):
            raw_id, class_id = item.split(":", 1)
            mapping[int(raw_id.strip())] = int(class_id.strip())
        return mapping
    if isinstance(raw_mapping, dict):
        return {int(raw_id): int(class_id) for raw_id, class_id in raw_mapping.items()}
    raise TypeError(f"Unsupported label map type: {type(raw_mapping).__name__}")


def _remap_labels(
    labels: np.ndarray,
    label_map: dict[int, int] | None,
    default_class: int | None,
    *,
    ignore_index: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if not label_map:
        return labels.astype(np.int64)

    fill_value = ignore_index if default_class is None else int(default_class)
    remapped = np.full(labels.shape, fill_value, dtype=np.int64)
    known = np.zeros(labels.shape, dtype=bool)
    for raw_id, class_id in label_map.items():
        mask = labels == int(raw_id)
        remapped[mask] = int(class_id)
        known |= mask
    if default_class is None and not known.all():
        unknown = sorted(int(v) for v in np.unique(labels[~known]).tolist())
        raise ValueError(f"SEG labels contain unmapped raw ids: {unknown}")
    return remapped


def _occ_imports() -> SimpleNamespace:
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.BRepClass import BRepClass_FaceClassifier
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import (
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_IN,
            TopAbs_ON,
            TopAbs_REVERSED,
        )
        from OCC.Core.TopTools import (
            TopTools_IndexedDataMapOfShapeListOfShape,
            TopTools_IndexedMapOfShape,
        )
        from OCC.Core.TopoDS import topods
        from OCC.Core.gp import gp_Pnt2d
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for STEP feature extraction. "
            "Install the OCC extra with: uv sync --extra occ"
        ) from exc

    try:
        from OCC.Core.TopExp import topexp

        top_exp = topexp
    except ImportError:
        try:
            from OCC.Core.TopExp import TopExp

            top_exp = TopExp
        except ImportError:
            from OCC.Core.TopExp import topexp_MapShapes, topexp_MapShapesAndAncestors

            top_exp = SimpleNamespace(
                MapShapes=topexp_MapShapes,
                MapShapesAndAncestors=topexp_MapShapesAndAncestors,
            )

    try:
        from OCC.Core.BRepGProp import brepgprop

        brepgprop_SurfaceProperties = brepgprop.SurfaceProperties
        brepgprop_LinearProperties = brepgprop.LinearProperties
    except ImportError:
        from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties


    uv_bounds_fn = None
    try:
        from OCC.Core.BRepTools import breptools

        uv_bounds_fn = breptools.UVBounds
    except ImportError:
        try:
            from OCC.Core.BRepTools import BRepTools

            uv_bounds_fn = BRepTools.UVBounds
        except ImportError:
            try:
                from OCC.Core.BRepTools import breptools_UVBounds

                uv_bounds_fn = breptools_UVBounds
            except ImportError:
                uv_bounds_fn = None

    return SimpleNamespace(
        BRepAdaptor_Curve=BRepAdaptor_Curve,
        BRepAdaptor_Surface=BRepAdaptor_Surface,
        BRepClass_FaceClassifier=BRepClass_FaceClassifier,
        BRepLProp_SLProps=BRepLProp_SLProps,
        GProp_GProps=GProp_GProps,
        IFSelect_RetDone=IFSelect_RetDone,
        STEPControl_Reader=STEPControl_Reader,
        TopAbs_EDGE=TopAbs_EDGE,
        TopAbs_FACE=TopAbs_FACE,
        TopAbs_IN=TopAbs_IN,
        TopAbs_ON=TopAbs_ON,
        TopAbs_REVERSED=TopAbs_REVERSED,
        TopExp=top_exp,
        TopTools_IndexedDataMapOfShapeListOfShape=TopTools_IndexedDataMapOfShapeListOfShape,
        TopTools_IndexedMapOfShape=TopTools_IndexedMapOfShape,
        topods=topods,
        gp_Pnt2d=gp_Pnt2d,
        surface_properties=brepgprop_SurfaceProperties,
        linear_properties=brepgprop_LinearProperties,
        uv_bounds=uv_bounds_fn,
    )


class OccBRepExtractor:
    """Extract a face-attributed B-Rep graph from a STEP model with pythonocc-core."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.occ = _occ_imports()
        brep_cfg = config["brep"]
        self.grid_size = int(brep_cfg.get("uv_grid_size", 4))
        self.edge_grid_size = int(brep_cfg.get("edge_u_grid_size", self.grid_size))
        if self.grid_size < 1 or self.edge_grid_size < 1:
            raise ValueError("brep.uv_grid_size and brep.edge_u_grid_size must be positive.")
        self.precision = float(brep_cfg.get("occ_precision", 1.0e-6))
        self.smooth_angle = math.radians(float(brep_cfg.get("smooth_angle_degrees", 5.0)))
        self.surface_vocab = int(brep_cfg.get("surface_type_vocab", 32))
        self.edge_vocab = int(brep_cfg.get("edge_type_vocab", 32))
        self.ignore_index = int(config.get("labels", {}).get("ignore_index", -100))
        labels_cfg = config.get("labels", {})
        self.label_offset = int(labels_cfg.get("value_offset", 0))
        self.label_map = _parse_label_map(labels_cfg.get("raw_to_class_map"))
        default_class = labels_cfg.get("default_class", None)
        self.label_default_class = None if default_class is None else int(default_class)

    def extract(
        self,
        step_path: str | Path,
        seg_path: str | Path | None = None,
        *,
        labels_required: bool = True,
        strict_label_count: bool = True,
    ) -> dict[str, np.ndarray]:
        shape = self._read_step(Path(step_path))
        face_map = self._indexed_faces(shape)
        faces = [self.occ.topods.Face(face_map.FindKey(i)) for i in range(1, self._map_size(face_map) + 1)]
        face_features = [self._face_features(face) for face in faces]

        face_cont = np.stack([item["continuous"] for item in face_features], axis=0).astype(np.float32)
        face_surface_type = np.array([item["surface_type"] for item in face_features], dtype=np.int64)
        face_normals = np.stack([item["normal"] for item in face_features], axis=0).astype(np.float32)

        edge_arrays = self._edge_arrays(shape, face_map, face_normals)
        labels = self._read_labels(seg_path, len(faces), labels_required, strict_label_count)

        arrays: dict[str, np.ndarray] = {
            "face_cont": face_cont,
            "face_surface_type": face_surface_type,
            "edge_index": edge_arrays["edge_index"],
            "edge_cont": edge_arrays["edge_cont"],
            "edge_type": edge_arrays["edge_type"],
            "edge_relation": edge_arrays["edge_relation"],
        }
        if labels is not None:
            arrays["labels"] = labels
        return arrays

    def _read_step(self, path: Path):
        reader = self.occ.STEPControl_Reader()
        status = reader.ReadFile(str(path))
        if status != self.occ.IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {path}")
        reader.TransferRoots()
        return reader.OneShape()

    def _indexed_faces(self, shape):
        face_map = self.occ.TopTools_IndexedMapOfShape()
        self.occ.TopExp.MapShapes(shape, self.occ.TopAbs_FACE, face_map)
        if self._map_size(face_map) == 0:
            raise ValueError("The STEP model contains no faces.")
        return face_map

    def _face_features(self, face) -> dict[str, np.ndarray | int]:
        adaptor = self.occ.BRepAdaptor_Surface(face, True)
        surface_type = self._bounded_category(int(adaptor.GetType()), self.surface_vocab)
        area, centroid = self._face_area_centroid(face)
        umin, umax, vmin, vmax = self._safe_uv_bounds(face)
        uc = 0.5 * (umin + umax)
        vc = 0.5 * (vmin + vmax)
        center_point, center_normal, center_curv = self._sample_surface(adaptor, face, uc, vc)

        grid_values: list[float] = []
        u_values = self._linspace_inside(umin, umax, self.grid_size)
        v_values = self._linspace_inside(vmin, vmax, self.grid_size)
        for u in u_values:
            for v in v_values:
                point, normal, _ = self._sample_surface(adaptor, face, float(u), float(v))
                grid_values.extend(point.tolist())
                grid_values.extend(normal.tolist())
                grid_values.append(float(self._uv_is_on_face(face, float(u), float(v))))

        continuous = np.concatenate(
            [
                np.array([math.log1p(max(area, 0.0))], dtype=np.float32),
                centroid.astype(np.float32),
                center_normal.astype(np.float32),
                center_curv.astype(np.float32),
                np.asarray(grid_values, dtype=np.float32),
            ],
            axis=0,
        )
        return {
            "continuous": continuous,
            "surface_type": surface_type,
            "normal": center_normal.astype(np.float32),
            "center_point": center_point.astype(np.float32),
        }

    def _face_area_centroid(self, face) -> tuple[float, np.ndarray]:
        props = self.occ.GProp_GProps()
        self.occ.surface_properties(face, props)
        area = float(props.Mass())
        center = props.CentreOfMass()
        return area, np.array([center.X(), center.Y(), center.Z()], dtype=np.float32)

    def _safe_uv_bounds(self, face) -> tuple[float, float, float, float]:
        if self.occ.uv_bounds is None:
            return -1.0, 1.0, -1.0, 1.0
        try:
            bounds = self.occ.uv_bounds(face)
            umin, umax, vmin, vmax = [float(x) for x in bounds]
        except Exception:
            return -1.0, 1.0, -1.0, 1.0
        values = [umin, umax, vmin, vmax]
        if any((not math.isfinite(v)) for v in values) or umax <= umin or vmax <= vmin:
            return -1.0, 1.0, -1.0, 1.0
        return umin, umax, vmin, vmax

    def _uv_is_on_face(self, face, u: float, v: float) -> bool:
        """Return whether a UV sample belongs to the trimmed face (boundary included)."""
        try:
            uv = self.occ.gp_Pnt2d(u, v)
            try:
                classifier = self.occ.BRepClass_FaceClassifier(face, uv, self.precision)
            except TypeError:
                classifier = self.occ.BRepClass_FaceClassifier()
                classifier.Perform(face, uv, self.precision)
            state = classifier.State()
            return state == self.occ.TopAbs_IN or state == self.occ.TopAbs_ON
        except Exception:
            # A failed classifier must not silently mark an unverified point valid.
            return False

    def _sample_surface(
        self, adaptor, face, u: float, v: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point = np.zeros(3, dtype=np.float32)
        normal = np.zeros(3, dtype=np.float32)
        curv = np.zeros(4, dtype=np.float32)
        try:
            pnt = adaptor.Value(u, v)
            point = np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32)
            props = self.occ.BRepLProp_SLProps(adaptor, u, v, 2, self.precision)
            if props.IsNormalDefined():
                n = props.Normal()
                normal = np.array([n.X(), n.Y(), n.Z()], dtype=np.float32)
                if face.Orientation() == self.occ.TopAbs_REVERSED:
                    normal *= -1.0
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
            if props.IsCurvatureDefined():
                curv = np.array(
                    [
                        float(props.MinCurvature()),
                        float(props.MaxCurvature()),
                        float(props.MeanCurvature()),
                        float(props.GaussianCurvature()),
                    ],
                    dtype=np.float32,
                )
        except Exception:
            pass
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        return point, normal, curv

    def _edge_arrays(self, shape, face_map, face_normals: np.ndarray) -> dict[str, np.ndarray]:
        edge_face_map = self.occ.TopTools_IndexedDataMapOfShapeListOfShape()
        self.occ.TopExp.MapShapesAndAncestors(
            shape,
            self.occ.TopAbs_EDGE,
            self.occ.TopAbs_FACE,
            edge_face_map,
        )

        directed_edges: list[tuple[int, int]] = []
        edge_cont: list[list[float]] = []
        edge_type: list[int] = []
        edge_relation: list[int] = []

        for edge_idx in range(1, self._map_size(edge_face_map) + 1):
            edge = self.occ.topods.Edge(edge_face_map.FindKey(edge_idx))
            adjacent_faces = edge_face_map.FindFromIndex(edge_idx)
            face_ids: list[int] = []
            for adjacent_shape in self._iter_shape_list(adjacent_faces):
                face = self.occ.topods.Face(adjacent_shape)
                face_id = int(face_map.FindIndex(face)) - 1
                if face_id >= 0:
                    face_ids.append(face_id)
            unique_face_ids = sorted(set(face_ids))
            if len(unique_face_ids) < 2:
                continue

            length = self._edge_length(edge)
            edge_type_id = self._bounded_category(self._edge_type(edge), self.edge_vocab)
            edge_grid = self._sample_edge_grid(edge)
            for i, j in itertools.combinations(unique_face_ids, 2):
                n_i = face_normals[i]
                n_j = face_normals[j]
                dot = float(np.clip(np.dot(n_i, n_j), -1.0, 1.0))
                angle = float(math.acos(dot))
                relation = 1 if angle <= self.smooth_angle else 2
                attrs = [math.log1p(max(length, 0.0)), angle / math.pi, dot, *edge_grid]
                for src, dst in ((i, j), (j, i)):
                    directed_edges.append((src, dst))
                    edge_cont.append(attrs)
                    edge_type.append(edge_type_id)
                    edge_relation.append(relation)

        if not directed_edges:
            return {
                "edge_index": np.empty((2, 0), dtype=np.int64),
                "edge_cont": np.empty((0, 3 + 6 * self.edge_grid_size), dtype=np.float32),
                "edge_type": np.empty((0,), dtype=np.int64),
                "edge_relation": np.empty((0,), dtype=np.int64),
            }
        return {
            "edge_index": np.asarray(directed_edges, dtype=np.int64).T,
            "edge_cont": np.asarray(edge_cont, dtype=np.float32),
            "edge_type": np.asarray(edge_type, dtype=np.int64),
            "edge_relation": np.asarray(edge_relation, dtype=np.int64),
        }

    def _edge_length(self, edge) -> float:
        props = self.occ.GProp_GProps()
        try:
            self.occ.linear_properties(edge, props)
            return float(props.Mass())
        except Exception:
            return 0.0

    def _edge_type(self, edge) -> int:
        try:
            adaptor = self.occ.BRepAdaptor_Curve(edge)
            return int(adaptor.GetType())
        except Exception:
            return 0

    def _sample_edge_grid(self, edge) -> list[float]:
        """Uniformly sample XYZ and a unit tangent along a B-Rep edge."""
        values: list[float] = []
        try:
            adaptor = self.occ.BRepAdaptor_Curve(edge)
            first = float(adaptor.FirstParameter())
            last = float(adaptor.LastParameter())
            if not math.isfinite(first) or not math.isfinite(last) or last < first:
                raise ValueError("Edge has an invalid parameter range.")
            parameters = np.linspace(first, last, self.edge_grid_size, dtype=np.float64)
        except Exception:
            return [0.0] * (6 * self.edge_grid_size)

        for parameter in parameters:
            point = np.zeros(3, dtype=np.float32)
            tangent = np.zeros(3, dtype=np.float32)
            try:
                pnt = adaptor.Value(float(parameter))
                derivative = adaptor.DN(float(parameter), 1)
                point = np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float32)
                tangent = np.array(
                    [derivative.X(), derivative.Y(), derivative.Z()], dtype=np.float32
                )
                tangent_norm = float(np.linalg.norm(tangent))
                if math.isfinite(tangent_norm) and tangent_norm > self.precision:
                    tangent /= tangent_norm
                else:
                    tangent.fill(0.0)
            except Exception:
                pass
            point = np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0)
            tangent = np.nan_to_num(tangent, nan=0.0, posinf=0.0, neginf=0.0)
            values.extend(point.tolist())
            values.extend(tangent.tolist())
        return values

    def _read_labels(
        self,
        seg_path: str | Path | None,
        num_faces: int,
        labels_required: bool,
        strict_label_count: bool,
    ) -> np.ndarray | None:
        if seg_path is None:
            if labels_required:
                raise FileNotFoundError("SEG labels are required but no SEG path was provided.")
            return None

        label_path = Path(seg_path)
        labels: list[int] = []
        if label_path.suffix.lower() == ".json":
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"JSON label must be a list: {label_path}")
            labels = [int(value) - self.label_offset for value in payload]
        else:
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    labels.append(int(line.split()[0]) - self.label_offset)

        if strict_label_count and len(labels) != num_faces:
            raise ValueError(
                f"Label count mismatch for {seg_path}: got {len(labels)} labels, "
                f"but OCC parsed {num_faces} faces."
            )

        result = np.full((num_faces,), self.ignore_index, dtype=np.int64)
        count = min(len(labels), num_faces)
        if count:
            result[:count] = _remap_labels(
                np.asarray(labels[:count], dtype=np.int64),
                self.label_map,
                self.label_default_class,
                ignore_index=self.ignore_index,
            )
        return result

    @staticmethod
    def _linspace_inside(start: float, stop: float, count: int) -> np.ndarray:
        if count <= 1:
            return np.array([0.5 * (start + stop)], dtype=np.float32)
        span = stop - start
        margin = 0.05 * span
        return np.linspace(start + margin, stop - margin, count, dtype=np.float32)

    @staticmethod
    def _map_size(indexed_map) -> int:
        if hasattr(indexed_map, "Extent"):
            return int(indexed_map.Extent())
        return int(indexed_map.Size())

    @staticmethod
    def _iter_shape_list(shape_list):
        try:
            yield from shape_list
        except TypeError:
            if hasattr(shape_list, "Extent") and hasattr(shape_list, "Value"):
                for idx in range(1, int(shape_list.Extent()) + 1):
                    yield shape_list.Value(idx)
                return
            # pythonocc 7.4 exposes TopTools_ListOfShape through an explicit
            # iterator rather than Python's iteration protocol.
            try:
                from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
            except ImportError:
                raise
            iterator = TopTools_ListIteratorOfListOfShape(shape_list)
            while iterator.More():
                yield iterator.Value()
                iterator.Next()

    @staticmethod
    def _bounded_category(value: int, vocab_size: int) -> int:
        if vocab_size <= 1:
            return 0
        return max(0, min(value + 1, vocab_size - 1))
