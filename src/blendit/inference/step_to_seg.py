from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from blendit.brep.occ_extractor import OccBRepExtractor
from blendit.config import apply_overrides, feature_dims, load_experiment_config
from blendit.data.graph import BRepGraph, collate_graphs, graph_from_arrays, normalize_graph_features
from blendit.models import build_segmentation_model, predict_segmentation_probabilities
from blendit.training.common import load_checkpoint


STEP_EXTENSIONS = frozenset({".step", ".stp"})
CLASS_NAMES = {
    0: "NonTransition",
    1: "VBF",
    2: "EBF",
}
# The network is trained with compact class ids. SEG files use the original
# ABC/BrepDit labels requested by the downstream application.
CLASS_TO_SEG_LABEL = np.asarray([0, 6, 4], dtype=np.int64)


@dataclass(frozen=True)
class StepJob:
    step_path: Path
    relative_path: Path
    seg_path: Path


def prediction_to_seg_labels(prediction: np.ndarray | Sequence[int]) -> np.ndarray:
    classes = np.asarray(prediction, dtype=np.int64)
    if classes.ndim != 1:
        raise ValueError(f"Expected one class id per face, got shape={classes.shape}.")
    if classes.size and (int(classes.min()) < 0 or int(classes.max()) >= len(CLASS_TO_SEG_LABEL)):
        invalid = sorted(int(value) for value in np.unique(classes).tolist() if value < 0 or value >= 3)
        raise ValueError(f"Prediction contains invalid class ids: {invalid}")
    return CLASS_TO_SEG_LABEL[classes]


def write_seg_file(path: str | Path, prediction: np.ndarray | Sequence[int]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = prediction_to_seg_labels(prediction)
    content = "".join(f"{int(label)}\n" for label in labels)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def discover_step_files(input_path: str | Path, *, recursive: bool = True) -> tuple[Path, list[Path]]:
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    path = path.resolve()

    if path.is_file():
        if path.suffix.lower() not in STEP_EXTENSIONS:
            raise ValueError(f"Input file is not a STEP/STP file: {path}")
        return path.parent, [path]
    if not path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {path}")

    iterator = path.rglob("*") if recursive else path.glob("*")
    files = sorted(
        (candidate for candidate in iterator if candidate.is_file() and candidate.suffix.lower() in STEP_EXTENSIONS),
        key=lambda candidate: candidate.relative_to(path).as_posix().lower(),
    )
    return path, files


def default_output_dir(input_path: str | Path) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return path.parent / f"{path.stem}_seg"
    if path.name:
        return path.parent / f"{path.name}_seg"
    return path / "seg_predictions"


def build_jobs(input_root: Path, step_paths: Sequence[Path], output_dir: str | Path) -> list[StepJob]:
    output_root = Path(output_dir).expanduser().resolve()
    jobs: list[StepJob] = []
    outputs: dict[str, Path] = {}
    for step_path in step_paths:
        relative_path = step_path.relative_to(input_root)
        seg_path = output_root / relative_path.with_suffix(".seg")
        output_key = os.path.normcase(str(seg_path))
        previous = outputs.get(output_key)
        if previous is not None:
            raise ValueError(
                "Two STEP files would produce the same SEG path: "
                f"{previous} and {step_path} -> {seg_path}. Rename one of the input files."
            )
        outputs[output_key] = step_path
        jobs.append(StepJob(step_path=step_path, relative_path=relative_path, seg_path=seg_path))
    return jobs


def _torch_load(path: Path, device: torch.device | str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=device)


def _load_inference_config(
    checkpoint_path: Path,
    config_path: str | Path | None,
    data_config_path: str | Path | None,
    overrides: Sequence[str],
) -> dict[str, Any]:
    if config_path is not None:
        config = load_experiment_config(config_path, data_config_path, list(overrides))
    else:
        checkpoint = _torch_load(checkpoint_path)
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if not isinstance(config, dict):
            raise ValueError(
                "The checkpoint has no embedded config. Supply the training config with --config."
            )
        config = apply_overrides(config, list(overrides))
    config.setdefault("data", {})["labels_required"] = False
    config["data"]["strict_label_count"] = False
    return config


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a CUDA device.")
    return device


def _path_needs_ascii_staging(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return True
    return len(str(path)) >= 240


@contextmanager
def _occ_safe_step_path(path: Path) -> Iterator[Path]:
    """Stage Unicode/long paths because some Windows OCC builds only accept narrow paths."""
    if not _path_needs_ascii_staging(path):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="blendit_step_") as tmp_dir:
        staged_path = Path(tmp_dir) / f"input{path.suffix.lower()}"
        shutil.copy2(path, staged_path)
        yield staged_path


def _extract_graph(extractor: OccBRepExtractor, config: dict[str, Any], job: StepJob) -> BRepGraph:
    with _occ_safe_step_path(job.step_path) as occ_path:
        arrays = extractor.extract(
            occ_path,
            seg_path=None,
            labels_required=False,
            strict_label_count=False,
        )
    graph = graph_from_arrays(arrays, job.relative_path.with_suffix("").as_posix())
    if bool(config.get("train", {}).get("normalize_per_graph", True)):
        graph = normalize_graph_features(graph)
    return graph


def _class_counts(prediction: np.ndarray) -> dict[str, int]:
    return {
        CLASS_NAMES[class_id]: int((prediction == class_id).sum())
        for class_id in sorted(CLASS_NAMES)
    }


def _seg_label_counts(prediction: np.ndarray) -> dict[str, int]:
    labels = prediction_to_seg_labels(prediction)
    return {str(label): int((labels == label).sum()) for label in (0, 4, 6)}


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict_batch(
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    jobs: Sequence[StepJob],
    graphs: Sequence[BRepGraph],
) -> list[dict[str, Any]]:
    batch = collate_graphs(list(graphs)).to(device)
    with torch.inference_mode():
        probabilities = predict_segmentation_probabilities(model, batch, config)
    predictions = probabilities.argmax(dim=-1).cpu().numpy().astype(np.int64)
    confidences = probabilities.max(dim=-1).values.cpu().numpy()
    graph_ptr = batch.graph_ptr.cpu().numpy()

    records: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        start = int(graph_ptr[index])
        end = int(graph_ptr[index + 1])
        prediction = predictions[start:end]
        confidence = confidences[start:end]
        write_seg_file(job.seg_path, prediction)
        records.append(
            {
                "step_path": str(job.step_path),
                "relative_path": job.relative_path.as_posix(),
                "seg_path": str(job.seg_path),
                "faces": int(prediction.size),
                "class_counts": _class_counts(prediction),
                "seg_label_counts": _seg_label_counts(prediction),
                "mean_confidence": float(confidence.mean()) if confidence.size else 0.0,
            }
        )
    return records


def run_inference(args: argparse.Namespace) -> int:
    started_at = time.time()
    input_root, step_paths = discover_step_files(args.input_path, recursive=not args.no_recursive)
    if not step_paths:
        raise ValueError(f"No STEP/STP files were found under: {input_root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(input_root, step_paths, output_dir)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    config = _load_inference_config(
        checkpoint_path,
        args.config,
        args.data_config,
        args.override,
    )
    device = _resolve_device(args.device)
    face_dim, edge_dim = feature_dims(config)
    model = build_segmentation_model(config, face_dim, edge_dim).to(device)
    checkpoint_epoch = load_checkpoint(checkpoint_path, model=model, optimizer=None, device=device)
    model.eval()
    extractor = OccBRepExtractor(config)

    manifest: dict[str, Any] = {
        "input_path": str(Path(args.input_path).expanduser().resolve()),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "device": str(device),
        "label_mapping": {"NonTransition": 0, "VBF": 6, "EBF": 4},
        "total_step_files": len(jobs),
        "samples": [],
        "failures": [],
    }

    batch_size = max(1, int(args.batch_size))
    pending_jobs: list[StepJob] = []
    pending_graphs: list[BRepGraph] = []

    def flush_pending() -> None:
        if not pending_jobs:
            return
        try:
            records = _predict_batch(model, device, config, pending_jobs, pending_graphs)
            manifest["samples"].extend(records)
            for record in records:
                print(f"[OK] {record['relative_path']} -> {record['seg_path']} ({record['faces']} faces)")
        except Exception as exc:
            if len(pending_jobs) == 1:
                job = pending_jobs[0]
                manifest["failures"].append(
                    {
                        "step_path": str(job.step_path),
                        "relative_path": job.relative_path.as_posix(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[FAILED] {job.relative_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise
            else:
                # A malformed graph should not prevent valid files in the same batch
                # from producing outputs.
                for one_job, one_graph in zip(pending_jobs, pending_graphs):
                    try:
                        records = _predict_batch(model, device, config, [one_job], [one_graph])
                        manifest["samples"].extend(records)
                        record = records[0]
                        print(f"[OK] {record['relative_path']} -> {record['seg_path']} ({record['faces']} faces)")
                    except Exception as single_exc:
                        manifest["failures"].append(
                            {
                                "step_path": str(one_job.step_path),
                                "relative_path": one_job.relative_path.as_posix(),
                                "error_type": type(single_exc).__name__,
                                "error": str(single_exc),
                            }
                        )
                        print(
                            f"[FAILED] {one_job.relative_path}: {type(single_exc).__name__}: {single_exc}",
                            file=sys.stderr,
                        )
                        if args.fail_fast:
                            raise
        finally:
            pending_jobs.clear()
            pending_graphs.clear()

    for index, job in enumerate(jobs, start=1):
        if args.skip_existing and job.seg_path.is_file():
            manifest["samples"].append(
                {
                    "step_path": str(job.step_path),
                    "relative_path": job.relative_path.as_posix(),
                    "seg_path": str(job.seg_path),
                    "skipped": True,
                }
            )
            print(f"[SKIP] {job.relative_path} ({index}/{len(jobs)})")
            continue
        try:
            graph = _extract_graph(extractor, config, job)
            pending_jobs.append(job)
            pending_graphs.append(graph)
            if len(pending_jobs) >= batch_size:
                flush_pending()
        except Exception as exc:
            manifest["failures"].append(
                {
                    "step_path": str(job.step_path),
                    "relative_path": job.relative_path.as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {job.relative_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise
    flush_pending()

    manifest["successful"] = len(manifest["samples"])
    manifest["failed"] = len(manifest["failures"])
    manifest["elapsed_seconds"] = round(time.time() - started_at, 3)
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Done: total={len(jobs)} successful={manifest['successful']} failed={manifest['failed']} "
        f"manifest={manifest_path}"
    )
    return 2 if manifest["failures"] else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict face labels for every STEP/STP file in a path and write SEG files "
            "with NonTransition=0, VBF=6 and EBF=4."
        )
    )
    parser.add_argument("input_path", help="A STEP/STP file or a directory containing STEP/STP files.")
    parser.add_argument("--checkpoint", required=True, help="Finetuned Blendit checkpoint.")
    parser.add_argument("--config", default=None, help="Optional YAML config. By default the checkpoint config is used.")
    parser.add_argument(
        "--data-config",
        default=None,
        help="Prepare-data YAML used when --config points to a training-only YAML.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <input>_seg.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device such as cuda:0.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of models per inference batch.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the top level of an input directory.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing SEG files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed STEP file.")
    parser.add_argument("--override", action="append", default=[], help="Override config, for example brep.uv_grid_size=4.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run_inference(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
