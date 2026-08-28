from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from brepprediff.brep.occ_extractor import OccBRepExtractor, _occ_imports, _parse_label_map, _remap_labels
from brepprediff.config import feature_dims
from brepprediff.data.dataset import _project_occ_grid_v2_to_legacy
from brepprediff.data.graph import BRepGraph, collate_graphs, load_graph_npz, normalize_graph_features, save_graph_npz
from brepprediff.models import build_segmentation_model, predict_segmentation_probabilities
from brepprediff.inference.step_to_seg import _load_inference_config, write_seg_file
from brepprediff.training.common import load_checkpoint, resolve_device, seed_everything
from brepprediff.training.evaluate import classification_metrics_from_confusion


CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (218, 222, 230),
    1: (255, 79, 163),
    2: (255, 212, 0),
}
CLASS_NAMES = {
    0: "NonTransition",
    1: "VBF",
    2: "EBF",
}
BINARY_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: CLASS_COLORS[0],
    1: CLASS_COLORS[1],
}
BINARY_CLASS_NAMES = {
    0: "NonTransition",
    1: "Transition",
}
INPUT_COLOR = (218, 222, 230)


@dataclass(frozen=True)
class InferenceSample:
    sample_id: str
    source_item: str
    step_path: Path
    seg_path: Path | None
    cache_path: Path | None


@dataclass(frozen=True)
class PlyStats:
    vertices: int
    faces: int
    skipped_faces: int


def _read_split(path: Path) -> list[str]:
    items: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _read_path_list(path: Path) -> list[str]:
    return _read_split(path)


def _sample_id_from_item(item: str) -> str:
    path = Path(item)
    if path.suffix:
        return path.with_suffix("").as_posix()
    return path.as_posix()


def _sample_id_from_step_path(step_path: Path, step_root: Path | None = None) -> str:
    if step_root is not None:
        try:
            return step_path.resolve().relative_to(step_root.resolve()).with_suffix("").as_posix()
        except ValueError:
            pass
    return step_path.with_suffix("").name


def _deduplicate_sample_ids(samples: list[InferenceSample]) -> list[InferenceSample]:
    counts: dict[str, int] = {}
    deduplicated: list[InferenceSample] = []
    for sample in samples:
        count = counts.get(sample.sample_id, 0)
        counts[sample.sample_id] = count + 1
        if count == 0:
            deduplicated.append(sample)
            continue
        digest = hashlib.sha1(str(sample.step_path.resolve()).encode("utf-8")).hexdigest()[:8]
        deduplicated.append(
            InferenceSample(
                sample_id=f"{sample.sample_id}_{digest}",
                source_item=sample.source_item,
                step_path=sample.step_path,
                seg_path=sample.seg_path,
                cache_path=sample.cache_path,
            )
        )
    return deduplicated


def _prefix_sample_ids(samples: list[InferenceSample], prefix: str) -> list[InferenceSample]:
    if not prefix:
        return samples
    return [
        InferenceSample(
            sample_id=f"{prefix}{sample.sample_id}",
            source_item=sample.source_item,
            step_path=sample.step_path,
            seg_path=sample.seg_path,
            cache_path=sample.cache_path,
        )
        for sample in samples
    ]


def _safe_output_stem(sample_id: str) -> str:
    return sample_id.replace("\\", "__").replace("/", "__").replace(" ", "_")


def _resolve_step_path(steps_dir: Path, item: str, extensions: Iterable[str]) -> Path:
    item_path = Path(item)
    candidates: list[Path] = []
    if item_path.is_absolute():
        candidates.append(item_path)
    else:
        candidates.append(steps_dir / item_path)
        if not item_path.suffix:
            candidates.extend(steps_dir / f"{item}{ext}" for ext in extensions)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"STEP file for split item {item!r} was not found under {steps_dir}.")


def _resolve_split_seg_path(segs_dir: Path, steps_dir: Path, step_path: Path) -> Path | None:
    rel = step_path.relative_to(steps_dir)
    candidates = [
        segs_dir / rel.with_suffix(".seg"),
        segs_dir / rel.with_suffix(".json"),
        segs_dir / f"{step_path.stem}.seg",
        segs_dir / f"{step_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _seg_lookup_keys(seg_path: Path, seg_root: Path | None = None) -> set[str]:
    keys = {seg_path.name, seg_path.stem}
    if seg_root is not None:
        try:
            rel = seg_path.resolve().relative_to(seg_root.resolve())
            keys.add(rel.as_posix())
            keys.add(rel.with_suffix("").as_posix())
        except ValueError:
            pass
    return keys


def _build_seg_index(seg_paths: Iterable[Path], seg_root: Path | None = None) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for seg_path in seg_paths:
        if not seg_path.exists() or not seg_path.is_file():
            continue
        for key in _seg_lookup_keys(seg_path, seg_root):
            index.setdefault(key, []).append(seg_path)
    return index


def _resolve_indexed_seg_path(
    step_path: Path,
    *,
    sample_id: str,
    step_root: Path | None,
    seg_root: Path | None,
    seg_index: dict[str, list[Path]],
) -> Path | None:
    candidates: list[Path] = [step_path.with_suffix(".seg")]
    if step_root is not None and seg_root is not None:
        try:
            rel = step_path.resolve().relative_to(step_root.resolve())
            candidates.append(seg_root / rel.with_suffix(".seg"))
        except ValueError:
            pass
    if seg_root is not None:
        candidates.append(seg_root / f"{step_path.stem}.seg")

    for key in {sample_id, step_path.stem, step_path.name, Path(sample_id).name}:
        candidates.extend(seg_index.get(key, []))

    existing = [path for path in candidates if path.exists() and path.is_file()]
    unique = sorted({path.resolve(): path for path in existing}.values())
    if len(unique) > 1:
        formatted = ", ".join(str(path) for path in unique[:5])
        suffix = " ..." if len(unique) > 5 else ""
        raise ValueError(f"STEP file {step_path} matched multiple SEG files: {formatted}{suffix}")
    return unique[0] if unique else None


def _cache_path_for_step(cache_dir: Path, steps_dir: Path, step_path: Path) -> Path:
    rel = step_path.relative_to(steps_dir).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_stem = step_path.stem.replace(" ", "_")
    return cache_dir / f"{safe_stem}_{digest}.npz"


def _cache_path_for_direct_step(cache_dir: Path, step_path: Path, step_root: Path | None) -> Path:
    if step_root is not None:
        try:
            rel = step_path.resolve().relative_to(step_root.resolve()).as_posix()
        except ValueError:
            rel = str(step_path.resolve())
    else:
        rel = str(step_path.resolve())
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_stem = step_path.stem.replace(" ", "_")
    return cache_dir / f"{safe_stem}_{digest}.npz"


def _configured_cache_dirs(data_cfg: dict[str, Any], split: str, override: str | None) -> list[Path]:
    if override:
        return [Path(override)]

    dirs: list[Path] = []
    cache_dirs = data_cfg.get("cache_dirs")
    if isinstance(cache_dirs, dict):
        split_dir = cache_dirs.get(split)
        if split_dir:
            return [Path(split_dir)]
    if data_cfg.get("cache_dir"):
        dirs.append(Path(data_cfg["cache_dir"]))

    unique: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            unique.append(directory)
            seen.add(key)
    return unique


def _cache_lookup_keys(cache_path: Path, cache_dir: Path) -> set[str]:
    keys = {cache_path.name, cache_path.stem}
    try:
        rel = cache_path.relative_to(cache_dir)
        keys.add(rel.as_posix())
        keys.add(rel.with_suffix("").as_posix())
    except ValueError:
        pass
    if "_" in cache_path.stem:
        keys.add(cache_path.stem.rsplit("_", 1)[0])
    return keys


def _build_cache_index(cache_dirs: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for cache_path in sorted(cache_dir.rglob("*.npz")):
            if not cache_path.is_file():
                continue
            for key in _cache_lookup_keys(cache_path, cache_dir):
                index.setdefault(key, []).append(cache_path)
    return index


def _find_cache_path(cache_dirs: list[Path], cache_index: dict[str, list[Path]], item: str) -> Path | None:
    item_path = Path(item)
    stem = item_path.stem if item_path.suffix else item_path.name
    safe_stem = stem.replace(" ", "_")

    candidates: list[Path] = []
    if item_path.is_absolute():
        candidates.append(item_path)
    for cache_dir in cache_dirs:
        candidates.append(cache_dir / item_path)
        candidates.append(cache_dir / f"{safe_stem}.npz")
    for key in {item_path.as_posix(), item_path.with_suffix("").as_posix(), safe_stem}:
        candidates.extend(cache_index.get(key, []))

    existing = [path for path in candidates if path.exists() and path.is_file()]
    unique = sorted({path.resolve(): path for path in existing}.values())
    if len(unique) > 1:
        formatted = ", ".join(str(path) for path in unique[:5])
        suffix = " ..." if len(unique) > 5 else ""
        raise ValueError(f"Split item {item!r} matched multiple cache files: {formatted}{suffix}")
    return unique[0] if unique else None


def _build_samples(
    config: dict[str, Any],
    *,
    split: str,
    split_file: str | None,
    cache_dir: str | None,
) -> list[InferenceSample]:
    data_cfg = config["data"]
    raw_split_path = split_file or data_cfg.get(f"{split}_split")
    if not raw_split_path:
        raise ValueError(f"No split file configured for split={split!r}. Use --split-file or data.{split}_split.")
    split_path = Path(raw_split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    steps_dir = Path(data_cfg["steps_dir"])
    segs_dir = Path(data_cfg["segs_dir"])
    extensions = data_cfg.get("step_extensions", [".step", ".stp"])
    cache_dirs = _configured_cache_dirs(data_cfg, split, cache_dir)
    cache_index = _build_cache_index(cache_dirs)
    write_cache_dir = cache_dirs[0] if cache_dirs else None

    samples: list[InferenceSample] = []
    for item in _read_split(split_path):
        step_path = _resolve_step_path(steps_dir, item, extensions)
        cache_path = _find_cache_path(cache_dirs, cache_index, item)
        if cache_path is None and write_cache_dir is not None:
            cache_path = _cache_path_for_step(write_cache_dir, steps_dir, step_path)
        samples.append(
            InferenceSample(
                sample_id=_sample_id_from_item(item),
                source_item=item,
                step_path=step_path,
                seg_path=_resolve_split_seg_path(segs_dir, steps_dir, step_path),
                cache_path=cache_path,
            )
        )
    return _deduplicate_sample_ids(samples)


def _iter_step_dir(step_dir: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {ext.lower() for ext in extensions}
    return sorted(path for path in step_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts)


def _resolve_direct_step_path(item: str, step_root: Path | None) -> Path:
    path = Path(item)
    if not path.is_absolute() and step_root is not None:
        path = step_root / path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"STEP file does not exist: {path}")
    return path


def _resolve_direct_seg_path(item: str, seg_root: Path | None) -> Path:
    path = Path(item)
    if not path.is_absolute() and seg_root is not None:
        path = seg_root / path
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"SEG file does not exist: {path}")
    return path


def _build_direct_step_samples(
    config: dict[str, Any],
    *,
    steps: list[str],
    step_dirs: list[str],
    step_lists: list[str],
    step_root: str | None,
    segs: list[str],
    seg_dirs: list[str],
    seg_lists: list[str],
    seg_root: str | None,
    cache_dir: str | None,
) -> list[InferenceSample]:
    data_cfg = config["data"]
    extensions = data_cfg.get("step_extensions", [".step", ".stp"])
    root = Path(step_root) if step_root else None
    seg_base = Path(seg_root) if seg_root else root
    cache_root = Path(cache_dir) if cache_dir else None

    step_items: list[str] = list(steps)
    for list_path in step_lists:
        step_items.extend(_read_path_list(Path(list_path)))

    seg_items: list[str] = list(segs)
    for list_path in seg_lists:
        seg_items.extend(_read_path_list(Path(list_path)))

    step_paths: list[Path] = []
    for item in step_items:
        step_paths.append(_resolve_direct_step_path(item, root))
    for raw_dir in step_dirs:
        step_dir = Path(raw_dir)
        if not step_dir.is_absolute() and root is not None:
            step_dir = root / step_dir
        if not step_dir.exists() or not step_dir.is_dir():
            raise NotADirectoryError(f"STEP directory does not exist: {step_dir}")
        step_paths.extend(_iter_step_dir(step_dir, extensions))

    explicit_seg_paths = [_resolve_direct_seg_path(item, seg_base) for item in seg_items]
    seg_dir_paths: list[Path] = []
    for raw_dir in seg_dirs:
        seg_dir = Path(raw_dir)
        if not seg_dir.is_absolute() and seg_base is not None:
            seg_dir = seg_base / seg_dir
        if not seg_dir.exists() or not seg_dir.is_dir():
            raise NotADirectoryError(f"SEG directory does not exist: {seg_dir}")
        seg_dir_paths.extend(
            sorted(
                path
                for path in seg_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".seg", ".json"}
            )
        )

    paired_seg_paths: list[Path | None]
    if explicit_seg_paths and len(explicit_seg_paths) == len(step_paths):
        paired_seg_paths = explicit_seg_paths
    elif explicit_seg_paths and len(step_paths) == 1 and len(explicit_seg_paths) == 1:
        paired_seg_paths = explicit_seg_paths
    else:
        paired_seg_paths = [None] * len(step_paths)

    seg_index = _build_seg_index([*explicit_seg_paths, *seg_dir_paths], seg_base)
    samples = [
        _direct_sample_from_step_path(
            step_path,
            root=root,
            seg_root=seg_base,
            explicit_seg_path=paired_seg_paths[index],
            seg_index=seg_index,
            cache_root=cache_root,
        )
        for index, step_path in enumerate(step_paths)
    ]
    return _deduplicate_sample_ids(samples)


def _direct_sample_from_step_path(
    step_path: Path,
    *,
    root: Path | None,
    seg_root: Path | None,
    explicit_seg_path: Path | None,
    seg_index: dict[str, list[Path]],
    cache_root: Path | None,
) -> InferenceSample:
    sample_id = _sample_id_from_step_path(step_path, root)
    return InferenceSample(
        sample_id=sample_id,
        source_item=str(step_path),
        step_path=step_path,
        seg_path=explicit_seg_path
        or _resolve_indexed_seg_path(
            step_path,
            sample_id=sample_id,
            step_root=root,
            seg_root=seg_root,
            seg_index=seg_index,
        ),
        cache_path=_cache_path_for_direct_step(cache_root, step_path, root) if cache_root else None,
    )


class FinetuneInferenceDataset(Dataset):
    def __init__(self, config: dict[str, Any], samples: list[InferenceSample], *, write_cache: bool) -> None:
        self.config = config
        self.samples = samples
        self.write_cache = write_cache
        self.normalize_per_graph = bool(
            config.get("train", {}).get("normalize_per_graph", True)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BRepGraph:
        sample = self.samples[index]
        if sample.cache_path is not None and sample.cache_path.exists():
            graph = load_graph_npz(sample.cache_path, sample.sample_id)
        else:
            extractor = OccBRepExtractor(self.config)
            arrays = extractor.extract(
                sample.step_path,
                seg_path=None,
                labels_required=False,
                strict_label_count=False,
            )
            if self.write_cache and sample.cache_path is not None:
                save_graph_npz(sample.cache_path, arrays)
            graph = BRepGraph(
                face_cont=torch.as_tensor(arrays["face_cont"], dtype=torch.float32),
                face_surface_type=torch.as_tensor(arrays["face_surface_type"], dtype=torch.long),
                edge_index=torch.as_tensor(arrays["edge_index"], dtype=torch.long),
                edge_cont=torch.as_tensor(arrays["edge_cont"], dtype=torch.float32),
                edge_type=torch.as_tensor(arrays["edge_type"], dtype=torch.long),
                edge_relation=torch.as_tensor(arrays["edge_relation"], dtype=torch.long),
                labels=None,
                sample_id=sample.sample_id,
            )

        uv_grid_size = int(self.config["brep"]["uv_grid_size"])
        feature_schema = str(self.config["brep"].get("feature_schema", "occ_grid_v2"))
        if feature_schema == "legacy":
            graph = _project_occ_grid_v2_to_legacy(graph, uv_grid_size)
        if self.normalize_per_graph:
            if feature_schema == "legacy":
                graph = normalize_graph_features(graph)
            else:
                graph = normalize_graph_features(
                    graph,
                    uv_grid_size=uv_grid_size,
                    edge_u_grid_size=int(
                        self.config["brep"].get("edge_u_grid_size", uv_grid_size)
                    ),
                )
        return graph


def _mesh_occ_imports() -> Any:
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.TopLoc import TopLoc_Location
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for STEP to PLY visualization. "
            "Install the OCC extra with: uv sync --extra occ"
        ) from exc

    try:
        from OCC.Core.BRep import BRep_Tool

        triangulation = BRep_Tool.Triangulation
    except ImportError:
        try:
            from OCC.Core.BRep import breptool

            triangulation = breptool.Triangulation
        except ImportError:
            from OCC.Core.BRep import breptool_Triangulation

            triangulation = breptool_Triangulation

    occ = _occ_imports()
    occ.BRepMesh_IncrementalMesh = BRepMesh_IncrementalMesh
    occ.TopLoc_Location = TopLoc_Location
    occ.triangulation = triangulation
    return occ


def _mesh_shape(occ: Any, shape: Any, linear_deflection: float, angular_deflection: float) -> None:
    try:
        mesh = occ.BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    except TypeError:
        try:
            mesh = occ.BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection)
        except TypeError:
            mesh = occ.BRepMesh_IncrementalMesh(shape, linear_deflection)
    if hasattr(mesh, "Perform"):
        mesh.Perform()


def _triangulation_node(triangulation: Any, nodes: Any, index: int) -> Any:
    if hasattr(triangulation, "Node"):
        return triangulation.Node(index)
    return nodes.Value(index)


def _triangulation_triangle(triangulation: Any, triangles: Any, index: int) -> Any:
    if hasattr(triangulation, "Triangle"):
        return triangulation.Triangle(index)
    return triangles.Value(index)


def _transformed_xyz(point: Any, transform: Any) -> tuple[float, float, float]:
    try:
        transformed = point.Transformed(transform)
    except AttributeError:
        transformed = point
        try:
            transformed.Transform(transform)
        except AttributeError:
            pass
    return float(transformed.X()), float(transformed.Y()), float(transformed.Z())


def _triangle_indices(triangle: Any) -> tuple[int, int, int]:
    values = triangle.Get()
    return int(values[0]), int(values[1]), int(values[2])


def _read_step_shape(occ: Any, step_path: Path) -> Any:
    reader = occ.STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != occ.IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")
    reader.TransferRoots()
    return reader.OneShape()


def _indexed_faces(occ: Any, shape: Any) -> Any:
    face_map = occ.TopTools_IndexedMapOfShape()
    occ.TopExp.MapShapes(shape, occ.TopAbs_FACE, face_map)
    return face_map


def _map_size(indexed_map: Any) -> int:
    if hasattr(indexed_map, "Extent"):
        return int(indexed_map.Extent())
    return int(indexed_map.Size())


def _write_ascii_ply(path: Path, vertices: list[tuple[float, float, float, int, int, int]], faces: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment generated by brepprediff.inference.finetune_visualize\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z, red, green, blue in vertices:
            f.write(f"{x:.9g} {y:.9g} {z:.9g} {red:d} {green:d} {blue:d}\n")
        for i, j, k in faces:
            f.write(f"3 {i:d} {j:d} {k:d}\n")


def write_step_prediction_ply(
    step_path: Path,
    output_path: Path,
    face_classes: np.ndarray,
    *,
    color_map: dict[int, tuple[int, int, int]],
    linear_deflection: float,
    angular_deflection: float,
) -> PlyStats:
    return write_step_prediction_plys(
        step_path,
        [(output_path, face_classes, color_map)],
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )[0]


def write_step_prediction_plys(
    step_path: Path,
    outputs: list[tuple[Path, np.ndarray, dict[int, tuple[int, int, int]]]],
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> list[PlyStats]:
    """Write multiple face-color PLYs while reading and meshing the STEP only once."""
    if not outputs:
        return []
    occ = _mesh_occ_imports()
    shape = _read_step_shape(occ, step_path)
    _mesh_shape(occ, shape, linear_deflection, angular_deflection)
    face_map = _indexed_faces(occ, shape)
    num_faces = _map_size(face_map)
    for _, face_classes, _ in outputs:
        if int(face_classes.shape[0]) != num_faces:
            raise ValueError(
                f"Prediction face count mismatch for {step_path}: "
                f"predictions={face_classes.shape[0]} STEP faces={num_faces}"
            )

    vertices_by_output: list[list[tuple[float, float, float, int, int, int]]] = [
        [] for _ in outputs
    ]
    ply_faces: list[tuple[int, int, int]] = []
    skipped_faces = 0

    for face_index in range(1, num_faces + 1):
        face = occ.topods.Face(face_map.FindKey(face_index))
        location = occ.TopLoc_Location()
        triangulation = occ.triangulation(face, location)
        if triangulation is None:
            skipped_faces += 1
            continue

        transform = location.Transformation()
        nodes = triangulation.Nodes() if hasattr(triangulation, "Nodes") else None
        triangles = triangulation.Triangles() if hasattr(triangulation, "Triangles") else None
        colors = [
            color_map.get(int(face_classes[face_index - 1]), INPUT_COLOR)
            for _, face_classes, color_map in outputs
        ]
        local_to_global: dict[int, int] = {}

        for node_index in range(1, int(triangulation.NbNodes()) + 1):
            point = _triangulation_node(triangulation, nodes, node_index)
            x, y, z = _transformed_xyz(point, transform)
            local_to_global[node_index] = len(vertices_by_output[0])
            for vertices, color in zip(vertices_by_output, colors):
                vertices.append((x, y, z, color[0], color[1], color[2]))

        reverse = face.Orientation() == occ.TopAbs_REVERSED
        for triangle_index in range(1, int(triangulation.NbTriangles()) + 1):
            triangle = _triangulation_triangle(triangulation, triangles, triangle_index)
            i, j, k = _triangle_indices(triangle)
            if reverse:
                j, k = k, j
            ply_faces.append((local_to_global[i], local_to_global[j], local_to_global[k]))

    stats: list[PlyStats] = []
    for (output_path, _, _), vertices in zip(outputs, vertices_by_output):
        _write_ascii_ply(output_path, vertices, ply_faces)
        stats.append(PlyStats(vertices=len(vertices), faces=len(ply_faces), skipped_faces=skipped_faces))
    return stats


def _class_counts(prediction: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_id, name in CLASS_NAMES.items():
        counts[name] = int((prediction == class_id).sum())
    return counts


def _binary_transition_classes(classes: np.ndarray, *, source_num_classes: int = 3) -> np.ndarray:
    values = np.asarray(classes, dtype=np.int64)
    if source_num_classes not in {2, 3}:
        raise ValueError(f"Binary transition display expects a 2- or 3-class model, got {source_num_classes}")
    valid_classes = np.arange(source_num_classes, dtype=np.int64)
    invalid = np.setdiff1d(np.unique(values), valid_classes)
    if invalid.size:
        raise ValueError(
            f"Expected class ids 0..{source_num_classes - 1}, got {invalid.tolist()}"
        )
    if source_num_classes == 2:
        return values.copy()
    return (values != 0).astype(np.int64)


def _binary_class_counts(prediction: np.ndarray) -> dict[str, int]:
    return {
        name: int((prediction == class_id).sum())
        for class_id, name in BINARY_CLASS_NAMES.items()
    }


def _metrics_from_confusion(*, true_positive: int, true_negative: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "faces": int(total),
    }


def _binary_classification_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(f"Binary metric shape mismatch: prediction={prediction.shape}, target={target.shape}")
    for name, values in (("prediction", prediction), ("target", target)):
        invalid = np.setdiff1d(np.unique(values), np.asarray([0, 1], dtype=np.int64))
        if invalid.size:
            raise ValueError(f"Binary {name} contains values other than 0/1: {invalid.tolist()}")
    return _metrics_from_confusion(
        true_positive=int(np.logical_and(prediction == 1, target == 1).sum()),
        true_negative=int(np.logical_and(prediction == 0, target == 0).sum()),
        false_positive=int(np.logical_and(prediction == 1, target == 0).sum()),
        false_negative=int(np.logical_and(prediction == 0, target == 1).sum()),
    )


def _multiclass_classification_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Multiclass metric shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )
    valid = np.logical_and(target >= 0, target < num_classes)
    valid_prediction = prediction[valid]
    valid_target = target[valid]
    invalid_prediction = np.setdiff1d(
        np.unique(valid_prediction), np.arange(num_classes, dtype=np.int64)
    )
    if invalid_prediction.size:
        raise ValueError(
            f"Multiclass prediction contains ids outside 0..{num_classes - 1}: "
            f"{invalid_prediction.tolist()}"
        )
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (valid_target, valid_prediction), 1)
    return classification_metrics_from_confusion(torch.from_numpy(confusion))


def _read_seg_classes(config: dict[str, Any], seg_path: Path, num_faces: int) -> np.ndarray:
    labels_cfg = config.get("labels", {})
    ignore_index = int(labels_cfg.get("ignore_index", -100))
    label_offset = int(labels_cfg.get("value_offset", 0))
    label_map = _parse_label_map(labels_cfg.get("raw_to_class_map"))
    default_class = labels_cfg.get("default_class", None)
    default_class_id = None if default_class is None else int(default_class)

    labels: list[int] = []
    with seg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(int(line.split()[0]) - label_offset)

    if len(labels) != num_faces:
        raise ValueError(
            f"SEG label count mismatch for {seg_path}: got {len(labels)} labels, "
            f"but inference graph has {num_faces} faces."
        )
    return _remap_labels(
        np.asarray(labels, dtype=np.int64),
        label_map,
        default_class_id,
        ignore_index=ignore_index,
    )


def _read_filletrec_json_classes(label_path: Path, num_faces: int) -> np.ndarray:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"FilletRec label must be a JSON list: {label_path}")
    labels = np.asarray(payload, dtype=np.int64)
    if labels.ndim != 1 or int(labels.shape[0]) != num_faces:
        raise ValueError(
            f"FilletRec label count mismatch for {label_path}: "
            f"got shape {labels.shape}, but inference graph has {num_faces} faces."
        )
    invalid = np.setdiff1d(np.unique(labels), np.asarray([0, 1], dtype=np.int64))
    if invalid.size:
        raise ValueError(f"FilletRec label contains values other than 0/1: {invalid.tolist()}")
    return labels


def _read_ground_truth_classes(config: dict[str, Any], label_path: Path, num_faces: int) -> np.ndarray:
    if label_path.suffix.lower() == ".json":
        return _read_filletrec_json_classes(label_path, num_faces)
    return _read_seg_classes(config, label_path, num_faces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a finetuned BRepPreDiff segmentation model and export EBF/VBF highlighted PLY files for the viewer."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config. By default the checkpoint's embedded config is used.",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="Prepare-data YAML. Defaults to data_config in the training YAML.",
    )
    parser.add_argument("--checkpoint", required=True, help="Finetune checkpoint, e.g. runs/finetune/.../checkpoints/last.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-file", default=None, help="Override data.<split>_split.")
    parser.add_argument("--step", action="append", default=[], help="Direct STEP/STP file path. Can be used multiple times.")
    parser.add_argument("--step-dir", action="append", default=[], help="Directory scanned recursively for STEP/STP files.")
    parser.add_argument("--step-list", action="append", default=[], help="Text file with one STEP/STP path per line.")
    parser.add_argument("--step-root", default=None, help="Base directory for relative --step-list or --step paths.")
    parser.add_argument("--seg", action="append", default=[], help="Direct SEG file path for GT highlighting. Can be used multiple times.")
    parser.add_argument("--seg-dir", action="append", default=[], help="Directory scanned recursively for SEG files.")
    parser.add_argument("--seg-list", action="append", default=[], help="Text file with one SEG path per line.")
    parser.add_argument("--seg-root", default=None, help="Base directory for relative --seg-list or --seg paths.")
    parser.add_argument("--cache-dir", default=None, help="Optional feature cache directory. Direct STEP mode only writes cache when this is set.")
    parser.add_argument("--output-dir", default="tools/visualize/results")
    parser.add_argument(
        "--seg-output-dir",
        default=None,
        help="Optional directory for predicted SEG files (NonTransition=0, VBF=6, EBF=4).",
    )
    parser.add_argument("--manifest-name", default="prediction_manifest.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override train.device for inference, e.g. cuda or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Only infer the first N samples.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose semantic PLY already exists.")
    parser.add_argument("--no-write-cache", action="store_true", help="Do not write extracted missing feature cache.")
    parser.add_argument("--no-input-ply", action="store_true", help="Only write *_semantic_pred.ply.")
    parser.add_argument(
        "--binary-transition",
        action="store_true",
        help=(
            "Evaluate binary transition GT labels. Three-class model predictions map VBF/EBF "
            "to transition=1; native two-class predictions are used directly."
        ),
    )
    parser.add_argument("--sample-prefix", default="", help="Prefix output sample ids, e.g. filletrec__.")
    parser.add_argument("--linear-deflection", type=float, default=0.08)
    parser.add_argument("--angular-deflection", type=float, default=0.5)
    parser.add_argument("--override", action="append", default=[], help="Override config, e.g. data.test_split=...")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_inference_config(
        Path(args.checkpoint).expanduser().resolve(),
        args.config,
        args.data_config,
        args.override,
    )
    config["data"]["labels_required"] = False
    config["data"]["strict_label_count"] = False
    if args.device:
        config["train"]["device"] = args.device

    # Match the standard evaluation entry point so CUDA kernels and any model
    # initialization performed before checkpoint loading are reproducible.
    seed_everything(int(config.get("seed", 42)))

    direct_step_input = bool(args.step or args.step_dir or args.step_list)
    if direct_step_input:
        samples = _build_direct_step_samples(
            config,
            steps=args.step,
            step_dirs=args.step_dir,
            step_lists=args.step_list,
            step_root=args.step_root,
            segs=args.seg,
            seg_dirs=args.seg_dir,
            seg_lists=args.seg_list,
            seg_root=args.seg_root,
            cache_dir=args.cache_dir,
        )
    else:
        samples = _build_samples(
            config,
            split=args.split,
            split_file=args.split_file,
            cache_dir=args.cache_dir,
        )
    if args.limit is not None:
        samples = samples[: max(0, int(args.limit))]
    samples = _prefix_sample_ids(samples, args.sample_prefix)
    if not samples:
        raise ValueError("No samples to infer.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_output_dir = Path(args.seg_output_dir) if args.seg_output_dir else None
    if seg_output_dir is not None:
        seg_output_dir.mkdir(parents=True, exist_ok=True)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    batch_size = int(args.batch_size or config["train"].get("batch_size", 1))

    dataset = FinetuneInferenceDataset(config, samples, write_cache=not args.no_write_cache)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_graphs,
    )

    # train.require_cuda is a training safety contract; visualization may still
    # be run on CPU when pythonocc and CUDA are provided by different envs.
    device = resolve_device(config, enforce_cuda_requirement=False)
    face_dim, edge_dim = feature_dims(config)
    model = build_segmentation_model(config, face_dim, edge_dim).to(device)
    checkpoint_epoch = load_checkpoint(args.checkpoint, model=model, optimizer=None, device=device)
    model.eval()
    model_num_classes = int(config["model"]["num_classes"])

    print(f"checkpoint={args.checkpoint} epoch={checkpoint_epoch}")
    print(f"device={device} samples={len(samples)} batch_size={batch_size} output_dir={output_dir}")

    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "config": str(args.config) if args.config else None,
        "input_mode": "step" if direct_step_input else "split",
        "split": None if direct_step_input else args.split,
        "split_file": None if direct_step_input else str(args.split_file or config["data"].get(f"{args.split}_split")),
        "step": args.step,
        "step_dir": args.step_dir,
        "step_list": args.step_list,
        "step_root": args.step_root,
        "seg": args.seg,
        "seg_dir": args.seg_dir,
        "seg_list": args.seg_list,
        "seg_root": args.seg_root,
        "sample_prefix": args.sample_prefix,
        "seg_output_dir": str(seg_output_dir) if seg_output_dir else None,
        "task": "binary_transition" if args.binary_transition else "three_class_transition",
        "positive_class": (
            "Transition" if args.binary_transition and model_num_classes == 2
            else "Transition (VBF or EBF)" if args.binary_transition
            else None
        ),
        "model_num_classes": model_num_classes,
        "colors": {
            "NonTransition": INPUT_COLOR,
            **(
                {"Transition": BINARY_CLASS_COLORS[1]}
                if args.binary_transition
                else {"VBF": CLASS_COLORS[1], "EBF": CLASS_COLORS[2]}
            ),
        },
        "samples": [],
    }
    aggregate_confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    aggregate_multiclass_confusion = np.zeros(
        (model_num_classes, model_num_classes), dtype=np.int64
    )
    evaluated_multiclass_samples = 0

    with torch.no_grad():
        iterator = tqdm(
            dataloader,
            desc=f"infer {args.split}",
            disable=not bool(config.get("run", {}).get("show_progress", True)),
        )
        for batch in iterator:
            batch = batch.to(device)
            probs = predict_segmentation_probabilities(model, batch, config)
            pred = probs.argmax(dim=-1).detach().cpu().numpy()
            confidence = probs.max(dim=-1).values.detach().cpu().numpy()
            graph_ptr = batch.graph_ptr.detach().cpu().numpy()

            for batch_index, sample_id in enumerate(batch.sample_ids):
                sample = sample_by_id[sample_id]
                start = int(graph_ptr[batch_index])
                end = int(graph_ptr[batch_index + 1])
                sample_pred = pred[start:end].astype(np.int64)
                display_pred = (
                    _binary_transition_classes(sample_pred, source_num_classes=model_num_classes)
                    if args.binary_transition
                    else sample_pred
                )
                sample_conf = confidence[start:end]
                stem = _safe_output_stem(sample_id)
                input_ply = output_dir / f"{stem}_instance_pred_rgb.ply"
                semantic_ply = output_dir / f"{stem}_semantic_pred.ply"
                predicted_seg = seg_output_dir / f"{stem}.seg" if seg_output_dir else None

                if args.skip_existing and semantic_ply.exists() and (args.no_input_ply or input_ply.exists()):
                    continue

                input_stats = None
                raw_gt_classes = (
                    _read_ground_truth_classes(config, sample.seg_path, int(sample_pred.shape[0]))
                    if sample.seg_path is not None
                    else None
                )
                gt_classes = (
                    _binary_transition_classes(raw_gt_classes)
                    if args.binary_transition and raw_gt_classes is not None
                    else raw_gt_classes
                )
                color_map = BINARY_CLASS_COLORS if args.binary_transition else CLASS_COLORS
                ply_outputs = []
                if not args.no_input_ply:
                    ply_outputs.append(
                        (input_ply, gt_classes if gt_classes is not None else np.zeros_like(display_pred), color_map)
                    )
                ply_outputs.append((semantic_ply, display_pred, color_map))
                ply_stats = write_step_prediction_plys(
                    sample.step_path,
                    ply_outputs,
                    linear_deflection=float(args.linear_deflection),
                    angular_deflection=float(args.angular_deflection),
                )
                if not args.no_input_ply:
                    input_stats = ply_stats[0]
                semantic_stats = ply_stats[-1]
                if predicted_seg is not None:
                    write_seg_file(predicted_seg, sample_pred)

                binary_metrics = None
                multiclass_metrics = None
                if args.binary_transition and gt_classes is not None:
                    binary_metrics = _binary_classification_metrics(display_pred, gt_classes)
                    for key in aggregate_confusion:
                        aggregate_confusion[key] += int(binary_metrics[key])
                elif gt_classes is not None:
                    multiclass_metrics = _multiclass_classification_metrics(
                        display_pred,
                        gt_classes,
                        num_classes=model_num_classes,
                    )
                    aggregate_multiclass_confusion += np.asarray(
                        multiclass_metrics["confusion_matrix"], dtype=np.int64
                    )
                    evaluated_multiclass_samples += 1

                record = {
                    "sample_id": sample_id,
                    "source_item": sample.source_item,
                    "step_path": str(sample.step_path),
                    "seg_path": str(sample.seg_path) if sample.seg_path else None,
                    "cache_path": str(sample.cache_path) if sample.cache_path else None,
                    "input_ply": input_ply.name if not args.no_input_ply else None,
                    "semantic_ply": semantic_ply.name,
                    "predicted_seg": str(predicted_seg) if predicted_seg else None,
                    "faces": int(sample_pred.shape[0]),
                    "gt_available": sample.seg_path is not None,
                    "gt_class_counts": (
                        _binary_class_counts(gt_classes)
                        if args.binary_transition and gt_classes is not None
                        else _class_counts(gt_classes) if gt_classes is not None else None
                    ),
                    "class_counts": (
                        _binary_class_counts(display_pred)
                        if args.binary_transition
                        else _class_counts(display_pred)
                    ),
                    "model_three_class_counts": (
                        _class_counts(sample_pred)
                        if args.binary_transition and model_num_classes == 3
                        else None
                    ),
                    "binary_metrics": binary_metrics,
                    "metrics": multiclass_metrics,
                    "mean_confidence": float(np.mean(sample_conf)) if sample_conf.size else 0.0,
                    "ply_vertices": semantic_stats.vertices,
                    "ply_triangles": semantic_stats.faces,
                    "skipped_faces": semantic_stats.skipped_faces,
                }
                if input_stats is not None:
                    record["input_ply_triangles"] = input_stats.faces
                manifest["samples"].append(record)

    if args.binary_transition:
        manifest["metrics"] = _metrics_from_confusion(**aggregate_confusion)
        manifest["evaluated_samples"] = sum(
            record["binary_metrics"] is not None for record in manifest["samples"]
        )
    elif evaluated_multiclass_samples:
        detailed_metrics = classification_metrics_from_confusion(
            torch.from_numpy(aggregate_multiclass_confusion)
        )
        manifest["metrics"] = {
            "averaging": "macro",
            "accuracy": detailed_metrics["accuracy"],
            "precision": detailed_metrics["macro_precision"],
            "recall": detailed_metrics["macro_recall"],
            "f1": detailed_metrics["macro_f1"],
            "iou": detailed_metrics["macro_iou"],
            "weighted_f1": detailed_metrics["weighted_f1"],
            "weighted_iou": detailed_metrics["weighted_iou"],
            "faces": detailed_metrics["faces"],
        }
        manifest["detailed_metrics"] = detailed_metrics
        manifest["evaluated_samples"] = evaluated_multiclass_samples

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest={manifest_path} samples={len(manifest['samples'])}")


if __name__ == "__main__":
    main()
