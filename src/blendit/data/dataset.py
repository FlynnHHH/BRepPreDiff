from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import traceback

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from blendit.config import apply_overrides, load_config
from blendit.data.graph import (
    BRepGraph,
    collate_graphs,
    load_graph_npz,
    normalize_graph_features,
    save_graph_npz,
)


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


def _init_cache_worker(config: dict[str, Any], labels_required: bool, strict_label_count: bool) -> None:
    global _WORKER_CONFIG, _WORKER_LABELS_REQUIRED, _WORKER_STRICT_LABEL_COUNT
    _WORKER_CONFIG = config
    _WORKER_LABELS_REQUIRED = labels_required
    _WORKER_STRICT_LABEL_COUNT = strict_label_count


def _extract_sample_to_cache_worker(sample: StepSegSample) -> str | CacheFailure:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Cache worker was not initialized.")

    try:
        from blendit.brep.occ_extractor import OccBRepExtractor

        extractor = OccBRepExtractor(_WORKER_CONFIG)
        arrays = extractor.extract(
            sample.step_path,
            sample.seg_path,
            labels_required=_WORKER_LABELS_REQUIRED,
            strict_label_count=_WORKER_STRICT_LABEL_COUNT,
        )
        save_graph_npz(sample.cache_path, arrays)
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


def _resolve_split_cache(cache_dir: Path, item: str) -> Path:
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

    matches = sorted(cache_dir.rglob(f"{safe_stem}_*.npz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        formatted = ", ".join(str(match) for match in matches[:5])
        suffix = " ..." if len(matches) > 5 else ""
        raise ValueError(f"Split item {item!r} matched multiple cache files: {formatted}{suffix}")
    raise FileNotFoundError(f"Split item {item!r} was not found as a cache file under {cache_dir}.")


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
    rel = step_path.relative_to(steps_dir)
    candidates = [
        segs_dir / rel.with_suffix(".seg"),
        segs_dir / f"{step_path.stem}.seg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class StepSegDataset(Dataset):
    def __init__(self, config: dict, split: str = "train") -> None:
        self.config = config
        data_cfg = config["data"]
        root = Path(data_cfg["root"])
        self.steps_dir = root / data_cfg["steps_dir"]
        self.segs_dir = root / data_cfg["segs_dir"]
        self.cache_dir = Path(data_cfg["cache_dir"])
        self.labels_required = bool(data_cfg.get("labels_required", True))
        self.overwrite_cache = bool(data_cfg.get("overwrite_cache", False))
        self.cache_only = bool(data_cfg.get("cache_only", False))
        self.normalize_per_graph = bool(data_cfg.get("normalize_per_graph", True))
        self.strict_label_count = bool(data_cfg.get("strict_label_count", True))
        self.split = split

        split_key = f"{split}_split"
        split_path = data_cfg.get(split_key)
        split_items = _read_split(Path(split_path) if split_path else None)
        extensions = data_cfg.get("step_extensions", [".step", ".stp"])

        if self.cache_only:
            if split == "train" or split_items is None:
                cache_files = _iter_cache_files(self.cache_dir)
                samples = [
                    CachedSample(sample_id=cache_path.stem, cache_path=cache_path)
                    for cache_path in cache_files
                ]
            else:
                samples = []
                for item in split_items:
                    cache_path = _resolve_split_cache(self.cache_dir, item)
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
            seg_path = _match_seg(self.segs_dir, self.steps_dir, step_path)
            if seg_path is None and self.labels_required:
                raise FileNotFoundError(f"Missing SEG file for {step_path}.")
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
        from blendit.brep.occ_extractor import OccBRepExtractor

        extractor = OccBRepExtractor(self.config)
        arrays = extractor.extract(
            sample.step_path,
            sample.seg_path,
            labels_required=self.labels_required,
            strict_label_count=self.strict_label_count,
        )
        save_graph_npz(sample.cache_path, arrays)

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
        if self.cache_only:
            graph = load_graph_npz(sample.cache_path, sample.sample_id)
            if self.normalize_per_graph:
                graph = normalize_graph_features(graph)
            return graph
        if self.overwrite_cache or not sample.cache_path.exists():
            self._extract_to_cache(sample)
        graph = load_graph_npz(sample.cache_path, sample.sample_id)
        if self.normalize_per_graph:
            graph = normalize_graph_features(graph)
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
        if self.cache_only:
            print(f"cache {self.split}: cache_only=true, using {len(self.samples)} existing npz files")
            return []

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


def build_dataloader(config: dict, split: str, shuffle: bool, distributed: bool = False) -> DataLoader:
    dataset = StepSegDataset(config, split=split)
    if len(dataset) == 0:
        split_key = f"{split}_split"
        data_cfg = config.get("data", {})
        raise ValueError(
            f"No samples found for split={split!r}. "
            f"Check data.{split_key}={data_cfg.get(split_key)!r}, "
            f"data.cache_dir={data_cfg.get('cache_dir')!r}, "
            f"and data.cache_only={data_cfg.get('cache_only', False)!r}."
        )
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_graphs,
        pin_memory=False,
    )


def cache_main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache B-Rep graph features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel cache workers.")
    parser.add_argument("--invalid-log", default=None, help="JSONL file for failed cache samples.")
    parser.add_argument(
        "--overwrite-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing cache files. Default follows data.overwrite_cache.",
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.override)
    if args.overwrite_cache is not None:
        config.setdefault("data", {})["overwrite_cache"] = args.overwrite_cache
    dataset = StepSegDataset(config, split=args.split)
    failures = dataset.build_cache(num_workers=args.workers, invalid_log=args.invalid_log)
    if failures:
        print(f"cache completed with {len(failures)} invalid samples; see {dataset._invalid_log_path(args.invalid_log)}")


if __name__ == "__main__":
    cache_main()
