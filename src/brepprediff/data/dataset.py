from __future__ import annotations

from bisect import bisect_right
import copy
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable
import traceback

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from brepprediff.config import feature_dims, load_config
from brepprediff.data.classification import (
    extract_classification_arrays,
    match_classification_label,
    read_class_label,
)
from brepprediff.data.graph import (
    BRepGraph,
    collate_graphs,
    load_graph_npz,
    load_global_feature_stats,
    normalize_graph_features,
    save_graph_npz,
    typewise_global_standardize_graph_features,
)
from brepprediff.data.segmentation import (
    extract_segmentation_arrays,
    match_segmentation_label,
)
from brepprediff.task import CLASSIFICATION, SEGMENTATION, task_type


@dataclass(frozen=True)
class StepSegSample:
    sample_id: str
    step_path: Path
    seg_path: Path | None
    cache_path: Path


@dataclass(frozen=True)
class CachedSample:
    sample_id: str
    cache_path: Path


def _project_occ_grid_v2_to_legacy(graph: BRepGraph, uv_grid_size: int) -> BRepGraph:
    """Drop the v2 trim mask and edge grid while preserving legacy channels."""
    face_base = graph.face_cont[:, :11]
    face_grid = graph.face_cont[:, 11:].reshape(graph.num_faces, uv_grid_size**2, 7)
    legacy_face_grid = face_grid[:, :, :6].reshape(graph.num_faces, uv_grid_size**2 * 6)
    return replace(
        graph,
        face_cont=torch.cat([face_base, legacy_face_grid], dim=-1),
        edge_cont=graph.edge_cont[:, :3],
    )


@dataclass(frozen=True)
class CacheFailure:
    sample_id: str
    step_path: str
    seg_path: str | None
    cache_path: str
    error_type: str
    error: str
    traceback: str


_WORKER_CONFIG: dict[str, Any] | None = None
_WORKER_LABELS_REQUIRED = True
_WORKER_STRICT_LABEL_COUNT = True


class NonFiniteCacheError(ValueError):
    """Raised when an OCC extraction contains NaN/Inf and is not cached."""


def _nonfinite_array_problems(arrays: dict[str, np.ndarray]) -> list[str]:
    problems: list[str] = []
    for key, value in arrays.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.inexact):
            continue
        finite = np.isfinite(array)
        if not finite.all():
            bad_count = int(array.size - int(finite.sum()))
            problems.append(f"{key}: {bad_count} non-finite values")
    return problems


def _save_occ_cache(cache_path: Path, arrays: dict[str, np.ndarray]) -> None:
    problems = _nonfinite_array_problems(arrays)
    if problems:
        # An overwrite must not leave the previous cache in place after the new
        # extraction has been classified as invalid.
        if cache_path.exists():
            cache_path.unlink()
        raise NonFiniteCacheError(
            f"OCC extraction contains NaN/Inf ({'; '.join(problems)}); cache discarded"
        )
    save_graph_npz(cache_path, arrays)


def _init_cache_worker(
    config: dict[str, Any],
    labels_required: bool,
    strict_label_count: bool,
) -> None:
    global _WORKER_CONFIG, _WORKER_LABELS_REQUIRED, _WORKER_STRICT_LABEL_COUNT
    _WORKER_CONFIG = config
    _WORKER_LABELS_REQUIRED = labels_required
    _WORKER_STRICT_LABEL_COUNT = strict_label_count


def _extract_sample_to_cache_worker(sample: StepSegSample) -> str | CacheFailure:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Cache worker was not initialized.")

    try:
        from brepprediff.brep.occ_extractor import OccBRepExtractor

        extractor = OccBRepExtractor(_WORKER_CONFIG)
        arrays = _extract_sample_arrays(
            extractor,
            sample,
            _WORKER_CONFIG,
            labels_required=_WORKER_LABELS_REQUIRED,
            strict_label_count=_WORKER_STRICT_LABEL_COUNT,
        )
        _save_occ_cache(sample.cache_path, arrays)
        return sample.sample_id
    except Exception as exc:
        return _cache_failure(sample, exc)


def _cache_failure(sample: StepSegSample, exc: BaseException) -> CacheFailure:
    return CacheFailure(
        sample_id=sample.sample_id,
        step_path=str(sample.step_path),
        seg_path=str(sample.seg_path) if sample.seg_path is not None else None,
        cache_path=str(sample.cache_path),
        error_type=type(exc).__name__,
        error=str(exc),
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


def _read_split(split_path: Path | None) -> list[str] | None:
    if split_path is None:
        return None
    names: list[str] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def _iter_step_files(steps_dir: Path, extensions: Iterable[str]) -> list[Path]:
    exts = {ext.lower() for ext in extensions}
    files = [p for p in steps_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _cache_path(cache_dir: Path, steps_dir: Path, step_path: Path) -> Path:
    rel = step_path.relative_to(steps_dir).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_stem = step_path.stem.replace(" ", "_")
    return cache_dir / f"{safe_stem}_{digest}.npz"


def _resolve_split_step(steps_dir: Path, item: str, extensions: Iterable[str]) -> Path:
    candidate = steps_dir / item
    if candidate.exists():
        return candidate
    path = Path(item)
    if path.suffix:
        candidate = steps_dir / path
        if candidate.exists():
            return candidate
    for ext in extensions:
        candidate = steps_dir / f"{item}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Split item {item!r} was not found under {steps_dir}.")


def _iter_cache_files(cache_dir: Path) -> list[Path]:
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def _split_cache_index(cache_dir: Path) -> dict[str, list[Path]]:
    """Index hashed cache names once for cache-only migrated datasets."""
    index: dict[str, list[Path]] = {}
    for cache_path in _iter_cache_files(cache_dir):
        source_stem, separator, _digest = cache_path.stem.rpartition("_")
        if separator:
            index.setdefault(source_stem, []).append(cache_path)
    return index


def _resolve_split_cache(
    cache_dir: Path,
    item: str,
    cache_index: dict[str, list[Path]] | None = None,
) -> Path:
    path = Path(item)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([path, cache_dir / path])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    stem = path.stem if path.suffix else path.name
    safe_stem = stem.replace(" ", "_")
    exact = cache_dir / f"{safe_stem}.npz"
    if exact.exists() and exact.is_file():
        return exact

    matches = (
        cache_index.get(safe_stem, [])
        if cache_index is not None
        else sorted(cache_dir.rglob(f"{safe_stem}_*.npz"))
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = ", ".join(str(match) for match in matches[:5])
        suffix = " ..." if len(matches) > 5 else ""
        raise ValueError(f"Split item {item!r} matched multiple cache files: {formatted}{suffix}")
    raise FileNotFoundError(f"Split item {item!r} was not found as a cache file under {cache_dir}.")


def _resolve_split_cache_for_item(
    cache_dir: Path,
    steps_dir: Path,
    item: str,
    extensions: Iterable[str],
    cache_index: dict[str, list[Path]] | None = None,
) -> Path:
    try:
        step_path = _resolve_split_step(steps_dir, item, extensions)
    except FileNotFoundError:
        item_path = Path(item)
        if not item_path.is_absolute() and item_path.suffix.lower() != ".npz":
            relative_candidates = (
                [item_path]
                if item_path.suffix
                else [Path(f"{item}{extension}") for extension in extensions]
            )
            for relative_path in relative_candidates:
                digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
                safe_stem = relative_path.stem.replace(" ", "_")
                expected = cache_dir / f"{safe_stem}_{digest}.npz"
                if expected.is_file():
                    return expected
        return _resolve_split_cache(cache_dir, item, cache_index)

    expected = _cache_path(cache_dir, steps_dir, step_path)
    if expected.exists() and expected.is_file():
        return expected
    raise FileNotFoundError(
        f"Cache for split item {item!r} was not found at the expected path {expected}."
    )


def _cache_sample_id(item: str, cache_path: Path, step_extensions: Iterable[str]) -> str:
    path = Path(item)
    extensions = {ext.lower() for ext in step_extensions}
    if path.suffix.lower() in extensions:
        return path.with_suffix("").as_posix()
    if path.suffix.lower() == ".npz":
        return path.with_suffix("").as_posix()
    if path.suffix:
        return cache_path.stem
    return path.as_posix()


def _match_seg(segs_dir: Path, steps_dir: Path, step_path: Path) -> Path | None:
    return _match_label(segs_dir, steps_dir, step_path, task=SEGMENTATION)


def _match_label(
    labels_dir: Path,
    steps_dir: Path,
    step_path: Path,
    *,
    task: str,
) -> Path | None:
    if task == CLASSIFICATION:
        return match_classification_label(labels_dir, steps_dir, step_path)
    return match_segmentation_label(labels_dir, steps_dir, step_path)


def _configured_labels_dir(data_cfg: dict[str, Any]) -> Path:
    configured = data_cfg.get("labels_dir", data_cfg.get("segs_dir"))
    if configured is None:
        raise ValueError("data.labels_dir (or legacy data.segs_dir) must be configured.")
    return Path(configured)


# Kept as a private compatibility alias for callers from before the data split.
_read_class_label = read_class_label


def _extract_sample_arrays(
    extractor: Any,
    sample: StepSegSample,
    config: dict[str, Any],
    *,
    labels_required: bool,
    strict_label_count: bool,
) -> dict[str, np.ndarray]:
    if task_type(config) == CLASSIFICATION:
        return extract_classification_arrays(
            extractor,
            sample.step_path,
            sample.seg_path,
            config,
            labels_required=labels_required,
        )
    return extract_segmentation_arrays(
        extractor,
        sample.step_path,
        sample.seg_path,
        labels_required=labels_required,
        strict_label_count=strict_label_count,
    )


def _configured_cache_dir(data_cfg: dict[str, Any], split: str) -> Path:
    cache_dirs = data_cfg.get("cache_dirs")
    if isinstance(cache_dirs, dict) and cache_dirs.get(split):
        return Path(cache_dirs[split])
    return Path(data_cfg["cache_dir"])


class StepSegDataset(Dataset):
    def __init__(self, config: dict, split: str = "train", *, source_mode: bool = False) -> None:
        self.config = config
        data_cfg = config["data"]
        self.task = task_type(config)
        self.steps_dir = Path(data_cfg["steps_dir"])
        self.segs_dir = _configured_labels_dir(data_cfg)
        self.cache_dir = _configured_cache_dir(data_cfg, split)
        self.labels_required = bool(data_cfg.get("labels_required", True))
        self.overwrite_cache = bool(data_cfg.get("overwrite_cache", False))
        self.source_mode = bool(source_mode)
        self.strip_labels = bool(data_cfg.get("strip_labels", False))
        train_cfg = config.get("train", {})
        preprocessing_cfg = train_cfg.get("feature_preprocessing", {})
        configured_mode = preprocessing_cfg.get("mode") if isinstance(preprocessing_cfg, dict) else None
        self.feature_preprocessing_mode = str(
            configured_mode
            or ("per_graph" if train_cfg.get("normalize_per_graph", True) else "none")
        )
        if self.feature_preprocessing_mode not in {"per_graph", "none", "typewise_global", "geometric"}:
            raise ValueError(
                "train.feature_preprocessing.mode must be one of: "
                "per_graph, none, typewise_global, geometric."
            )
        self.global_feature_stats = None
        if self.feature_preprocessing_mode == "typewise_global":
            stats_path = preprocessing_cfg.get("stats_path")
            if not stats_path:
                raise ValueError(
                    "train.feature_preprocessing.stats_path is required for typewise_global."
                )
            self.global_feature_stats = load_global_feature_stats(stats_path)
        self.strict_label_count = bool(data_cfg.get("strict_label_count", True))
        self.split = split

        split_key = f"{split}_split"
        split_path = data_cfg.get(split_key)
        split_items = _read_split(Path(split_path) if split_path else None)
        extensions = data_cfg.get("step_extensions", [".step", ".stp"])

        if not self.source_mode:
            if split_items is None:
                cache_files = _iter_cache_files(self.cache_dir)
                samples = [
                    CachedSample(sample_id=cache_path.stem, cache_path=cache_path)
                    for cache_path in cache_files
                ]
            else:
                samples = []
                cache_index = None if self.steps_dir.is_dir() else _split_cache_index(self.cache_dir)
                for item in split_items:
                    cache_path = _resolve_split_cache_for_item(
                        self.cache_dir,
                        self.steps_dir,
                        item,
                        extensions,
                        cache_index,
                    )
                    samples.append(
                        CachedSample(
                            sample_id=_cache_sample_id(item, cache_path, extensions),
                            cache_path=cache_path,
                        )
                    )
            self.samples = samples
            return

        if split_items is None:
            step_files = _iter_step_files(self.steps_dir, extensions)
        else:
            step_files = [_resolve_split_step(self.steps_dir, item, extensions) for item in split_items]

        samples = []
        for step_path in step_files:
            # A source explicitly configured as unlabeled must not opportunistically
            # consume co-located task metadata (for example Fusion Gallery's
            # reconstruction/assembly JSON files) as SEG/CLS labels.
            seg_path = None
            if self.labels_required:
                seg_path = _match_label(
                    self.segs_dir,
                    self.steps_dir,
                    step_path,
                    task=self.task,
                )
            if seg_path is None and self.labels_required:
                expected = "CLS" if self.task == CLASSIFICATION else "SEG/JSON"
                raise FileNotFoundError(f"Missing {expected} label file for {step_path}.")
            rel_id = step_path.relative_to(self.steps_dir).with_suffix("").as_posix()
            samples.append(
                StepSegSample(
                    sample_id=rel_id,
                    step_path=step_path,
                    seg_path=seg_path,
                    cache_path=_cache_path(self.cache_dir, self.steps_dir, step_path),
                )
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_to_cache(self, sample: StepSegSample) -> None:
        from brepprediff.brep.occ_extractor import OccBRepExtractor

        extractor = OccBRepExtractor(self.config)
        arrays = _extract_sample_arrays(
            extractor,
            sample,
            self.config,
            labels_required=self.labels_required,
            strict_label_count=self.strict_label_count,
        )
        _save_occ_cache(sample.cache_path, arrays)

    def _invalid_log_path(self, invalid_log: str | Path | None) -> Path:
        if invalid_log:
            return Path(invalid_log)
        data_cfg = self.config.get("data", {})
        configured = data_cfg.get("invalid_log")
        if configured:
            return Path(configured)
        return self.cache_dir.parent / f"invalid_{self.split}.jsonl"

    @staticmethod
    def _append_failures(path: Path, failures: list[CacheFailure]) -> None:
        if not failures:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure.__dict__, ensure_ascii=False) + "\n")

    def __getitem__(self, index: int) -> BRepGraph:
        sample = self.samples[index]
        if not self.source_mode:
            graph = load_graph_npz(
                sample.cache_path,
                sample.sample_id,
                load_labels=not self.strip_labels,
            )
            return self._preprocess_graph(graph)
        if self.overwrite_cache or not sample.cache_path.exists():
            self._extract_to_cache(sample)
        graph = load_graph_npz(
            sample.cache_path,
            sample.sample_id,
            load_labels=not self.strip_labels,
        )
        return self._preprocess_graph(graph)

    def _preprocess_graph(self, graph: BRepGraph) -> BRepGraph:
        feature_schema = str(self.config["brep"].get("feature_schema", "occ_grid_v2"))
        uv_grid_size = int(self.config["brep"]["uv_grid_size"])
        train_cfg = self.config.get('train', {})
        evaluation_rotation = train_cfg.get('evaluation_rotation_matrix')
        if self.split != 'train' and evaluation_rotation is not None:
            from brepprediff.data.graph import augment_graph_geometry
            if feature_schema == 'legacy':
                raise ValueError('Evaluation rotation requires OCC-grid-v2.')
            graph = augment_graph_geometry(
                graph, uv_grid_size=uv_grid_size,
                edge_u_grid_size=int(self.config['brep'].get('edge_u_grid_size', uv_grid_size)),
                rotation_matrix=torch.tensor(evaluation_rotation),
            )
        if self.split == 'train' and train_cfg.get('stage') == 'finetune' and (
            train_cfg.get('rotation_augmentation', False) or train_cfg.get('uv_augmentation', False)
        ):
            from brepprediff.data.graph import augment_graph_geometry
            if feature_schema == 'legacy':
                raise ValueError('Geometry augmentation requires OCC-grid-v2.')
            rotation_enabled = bool(train_cfg.get('rotation_augmentation', False))
            rotation_probability = float(train_cfg.get('rotation_augmentation_probability', 1.0))
            if not 0.0 <= rotation_probability <= 1.0:
                raise ValueError('rotation_augmentation_probability must be in [0, 1].')
            graph = augment_graph_geometry(
                graph, uv_grid_size=uv_grid_size,
                edge_u_grid_size=int(self.config['brep'].get('edge_u_grid_size', uv_grid_size)),
                rotate=rotation_enabled and torch.rand(()) < rotation_probability,
                reparameterize=bool(train_cfg.get('uv_augmentation', False)),
            )
        if feature_schema == "legacy":
            edge_grid_size = int(self.config["brep"].get("edge_u_grid_size", uv_grid_size))
            v2_dims = (11 + uv_grid_size**2 * 7, 3 + edge_grid_size * 6)
            actual_dims = (int(graph.face_cont.shape[-1]), int(graph.edge_cont.shape[-1]))
            if actual_dims == v2_dims:
                graph = _project_occ_grid_v2_to_legacy(graph, uv_grid_size)
        expected_face_dim, expected_edge_dim = feature_dims(self.config)
        actual_dims = (int(graph.face_cont.shape[-1]), int(graph.edge_cont.shape[-1]))
        if actual_dims != (expected_face_dim, expected_edge_dim):
            raise ValueError(
                f"Cached graph {graph.sample_id!r} uses face/edge dimensions {actual_dims}, "
                f"but the configured OCC feature schema expects "
                f"({expected_face_dim}, {expected_edge_dim}). Rebuild the feature cache."
            )
        if self.feature_preprocessing_mode == "geometric":
            from brepprediff.data.graph import geometric_normalize_graph_features
            if feature_schema == "legacy":
                raise ValueError("Geometric normalization requires OCC-grid-v2.")
            return geometric_normalize_graph_features(
                graph, uv_grid_size=uv_grid_size,
                edge_u_grid_size=int(self.config['brep'].get('edge_u_grid_size', uv_grid_size)),
            )
        if feature_schema == "legacy":
            if self.feature_preprocessing_mode == "per_graph":
                return normalize_graph_features(graph)
            if self.feature_preprocessing_mode == "typewise_global":
                assert self.global_feature_stats is not None
                return typewise_global_standardize_graph_features(
                    graph,
                    self.global_feature_stats,
                    uv_grid_size=int(self.config["brep"]["uv_grid_size"]),
                    legacy=True,
                )
            return graph
        edge_grid_size = int(
            self.config["brep"].get("edge_u_grid_size", self.config["brep"]["uv_grid_size"])
        )
        if self.feature_preprocessing_mode == "per_graph":
            return normalize_graph_features(
                graph,
                uv_grid_size=int(self.config["brep"]["uv_grid_size"]),
                edge_u_grid_size=edge_grid_size,
            )
        if self.feature_preprocessing_mode == "typewise_global":
            assert self.global_feature_stats is not None
            return typewise_global_standardize_graph_features(
                graph,
                self.global_feature_stats,
                uv_grid_size=int(self.config["brep"]["uv_grid_size"]),
                edge_u_grid_size=edge_grid_size,
            )
        return graph

    def _cache_worker_count(self, num_workers: int | None) -> int:
        if num_workers is not None:
            return max(0, int(num_workers))
        data_cfg = self.config.get("data", {})
        return max(0, int(data_cfg.get("cache_num_workers", data_cfg.get("num_workers", 0))))

    def _build_cache_parallel(self, samples: list[StepSegSample], num_workers: int) -> list[CacheFailure]:
        sample_iter = iter(samples)
        max_pending = max(num_workers * 2, 1)
        pending: dict[Future[str], StepSegSample] = {}
        failures: list[CacheFailure] = []

        def submit_next(executor: ProcessPoolExecutor) -> bool:
            try:
                sample = next(sample_iter)
            except StopIteration:
                return False
            future = executor.submit(_extract_sample_to_cache_worker, sample)
            pending[future] = sample
            return True

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_cache_worker,
            initargs=(self.config, self.labels_required, self.strict_label_count),
        ) as executor:
            for _ in range(min(max_pending, len(samples))):
                submit_next(executor)

            with tqdm(total=len(samples), desc=f"cache {self.split} x{num_workers}") as progress:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        sample = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = _cache_failure(sample, exc)
                        if isinstance(result, CacheFailure):
                            failures.append(result)
                            progress.set_postfix(invalid=len(failures))
                        progress.update(1)
                        submit_next(executor)
        return failures

    def build_cache(
        self,
        num_workers: int | None = None,
        *,
        invalid_log: str | Path | None = None,
    ) -> list[CacheFailure]:
        if not self.source_mode:
            raise RuntimeError("build_cache requires StepSegDataset(..., source_mode=True).")

        total_samples = len(self.samples)
        samples = [
            sample
            for sample in self.samples
            if self.overwrite_cache or not sample.cache_path.exists()
        ]
        skipped = total_samples - len(samples)
        mode = "overwrite" if self.overwrite_cache else "resume"
        print(f"cache {self.split}: mode={mode}, pending={len(samples)}, skipped_existing={skipped}")
        if not samples:
            return []

        worker_count = self._cache_worker_count(num_workers)
        failures: list[CacheFailure] = []
        if worker_count <= 1:
            for sample in tqdm(samples, desc=f"cache {self.split}"):
                try:
                    self._extract_to_cache(sample)
                except Exception as exc:
                    failures.append(_cache_failure(sample, exc))
            self._append_failures(self._invalid_log_path(invalid_log), failures)
            return failures

        failures = self._build_cache_parallel(samples, worker_count)
        self._append_failures(self._invalid_log_path(invalid_log), failures)
        return failures


def _source_config_path(config: dict[str, Any], source: dict[str, Any]) -> Path:
    configured = source.get("data_config", source.get("config"))
    if not configured:
        raise ValueError("Each data.sources entry must set data_config.")
    path = Path(configured)
    if path.is_absolute():
        return path
    parent_config = config.get("data_config")
    base_dir = Path(parent_config).parent if parent_config else Path.cwd()
    return base_dir / path


def _mapped_source_splits(source: dict[str, Any], requested_split: str) -> list[str]:
    split_map = source.get("splits")
    if split_map is None:
        return [requested_split]
    if not isinstance(split_map, dict):
        raise TypeError(
            f"data.sources[{source.get('name', '?')}].splits must be a mapping."
        )
    mapped = split_map.get(requested_split)
    if mapped is None:
        return []
    if isinstance(mapped, str):
        return [mapped]
    if isinstance(mapped, list) and all(isinstance(value, str) for value in mapped):
        return mapped
    raise TypeError(
        f"data.sources[{source.get('name', '?')}].splits.{requested_split} "
        "must be a split name or list of split names."
    )


def _runtime_source_config(
    config: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    source_path = _source_config_path(config, source)
    source_config = load_config(source_path)
    if "data" not in source_config or "brep" not in source_config:
        raise ValueError(
            f"Multi-source data config must contain data and brep sections: {source_path}"
        )

    source_config = copy.deepcopy(source_config)
    source_config["data_config"] = str(source_path)
    source_config["seed"] = config.get("seed", 42)
    source_config["train"] = copy.deepcopy(config.get("train", {}))
    source_config["data"]["prepare_on_start"] = False
    source_config["data"]["strip_labels"] = bool(
        config.get("data", {}).get("strip_labels", False)
    )

    data_overrides = source.get("data")
    if data_overrides is not None:
        if not isinstance(data_overrides, dict):
            raise TypeError(
                f"data.sources[{source.get('name', '?')}].data must be a mapping."
            )
        source_config["data"].update(copy.deepcopy(data_overrides))
    return source_config


class MultiSourceDataset(Dataset):
    """Concatenate configured cached datasets while preserving source identity."""

    def __init__(self, config: dict[str, Any], split: str = "train") -> None:
        self.config = config
        self.split = split
        sources = config.get("data", {}).get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("data.sources must be a non-empty list.")

        root_brep = config.get("brep", {})
        self.datasets: list[StepSegDataset] = []
        self.component_names: list[str] = []
        self.component_sizes: list[int] = []
        self.cumulative_sizes: list[int] = []

        total = 0
        seen_names: set[str] = set()
        for source_index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise TypeError(f"data.sources[{source_index}] must be a mapping.")
            name = str(source.get("name", f"source_{source_index}"))
            if name in seen_names:
                raise ValueError(f"Duplicate multi-source dataset name: {name!r}")
            seen_names.add(name)

            source_config = _runtime_source_config(config, source)
            source_brep = source_config["brep"]
            for key in (
                "uv_grid_size",
                "surface_type_vocab",
                "edge_type_vocab",
                "relation_type_vocab",
            ):
                if int(source_brep[key]) != int(root_brep[key]):
                    raise ValueError(
                        f"Source {name!r} has brep.{key}={source_brep[key]}, "
                        f"expected {root_brep[key]}."
                    )
            source_edge_grid = int(
                source_brep.get("edge_u_grid_size", source_brep["uv_grid_size"])
            )
            root_edge_grid = int(
                root_brep.get("edge_u_grid_size", root_brep["uv_grid_size"])
            )
            if source_edge_grid != root_edge_grid:
                raise ValueError(
                    f"Source {name!r} has brep.edge_u_grid_size={source_edge_grid}, "
                    f"expected {root_edge_grid}."
                )

            mapped_splits = _mapped_source_splits(source, split)
            for source_split in mapped_splits:
                dataset = StepSegDataset(source_config, split=source_split)
                component_name = f"{name}/{source_split}"
                self.datasets.append(dataset)
                self.component_names.append(component_name)
                self.component_sizes.append(len(dataset))
                total += len(dataset)
                self.cumulative_sizes.append(total)

        if not self.datasets:
            raise ValueError(f"No multi-source components are configured for split={split!r}.")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> BRepGraph:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_index = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else self.cumulative_sizes[dataset_index - 1]
        graph = self.datasets[dataset_index][index - previous_size]
        return replace(
            graph,
            sample_id=f"{self.component_names[dataset_index]}:{graph.sample_id}",
        )


def build_dataloader(config: dict, split: str, shuffle: bool, distributed: bool = False) -> DataLoader:
    if config.get("data", {}).get("sources"):
        dataset: Dataset = MultiSourceDataset(config, split=split)
    else:
        dataset = StepSegDataset(config, split=split)
    if len(dataset) == 0:
        split_key = f"{split}_split"
        data_cfg = config.get("data", {})
        raise ValueError(
            f"No samples found for split={split!r}. "
            f"Check data.{split_key}={data_cfg.get(split_key)!r}, "
            f"data.cache_dir={data_cfg.get('cache_dir')!r}, "
            f"and data.cache_dirs={data_cfg.get('cache_dirs')!r}."
        )
    configured_seed = config["train"].get("dataloader_seed")
    loader_seed = None
    generator = None
    if configured_seed is not None:
        split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
        loader_seed = int(configured_seed) + split_offset
        generator = torch.Generator()
        generator.manual_seed(loader_seed)
    sampler = (
        DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=loader_seed if loader_seed is not None else 0,
        )
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(config["train"].get("num_workers", 0)),
        collate_fn=collate_graphs,
        pin_memory=False,
        generator=generator,
    )
