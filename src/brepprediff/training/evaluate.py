from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from brepprediff.brep.occ_extractor import OccBRepExtractor
from brepprediff.config import apply_overrides, feature_dims, load_experiment_config
from brepprediff.data import build_dataloader
from brepprediff.data.classification import (
    CLASS_LABEL_EXTENSIONS,
    extract_classification_arrays,
)
from brepprediff.data.graph import (
    BRepGraph,
    collate_graphs,
    graph_from_arrays,
    normalize_graph_features,
)
from brepprediff.data.segmentation import (
    SEGMENTATION_LABEL_EXTENSIONS,
    extract_segmentation_arrays,
)
from brepprediff.models import (
    build_finetune_model,
    finetune_confusion_matrix,
    predict_finetune_probabilities,
)
from brepprediff.training.common import check_finite_batch, load_checkpoint, resolve_device
from brepprediff.task import CLASSIFICATION, task_type
from brepprediff.utils import seed_everything


DEFAULT_CLASS_NAMES = ("NonTransition", "VBF", "EBF")
DEFAULT_BINARY_CLASS_NAMES = ("NonTransition", "Transition")
DEFAULT_STEP_EXTENSIONS = (".step", ".stp")
LABEL_EXTENSIONS = SEGMENTATION_LABEL_EXTENSIONS


@dataclass(frozen=True)
class StepSegSample:
    sample_id: str
    step_path: Path
    seg_path: Path


class StepSegDataset(Dataset):
    """Extract labeled B-Rep graphs directly from paired STEP and SEG files."""

    def __init__(self, config: dict[str, Any], samples: Sequence[StepSegSample]) -> None:
        self.config = config
        self.samples = list(samples)
        self.normalize_per_graph = bool(config.get("train", {}).get("normalize_per_graph", True))
        self._extractor: OccBRepExtractor | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> BRepGraph:
        sample = self.samples[index]
        if self._extractor is None:
            self._extractor = OccBRepExtractor(self.config)
        if task_type(self.config) == CLASSIFICATION:
            arrays = extract_classification_arrays(
                self._extractor,
                sample.step_path,
                sample.seg_path,
                self.config,
                labels_required=True,
            )
        else:
            arrays = extract_segmentation_arrays(
                self._extractor,
                sample.step_path,
                sample.seg_path,
                labels_required=True,
                strict_label_count=True,
            )
        graph = graph_from_arrays(arrays, sample.sample_id)
        if graph.labels is None:
            raise ValueError(f"No labels were loaded from {sample.seg_path}.")
        if self.normalize_per_graph:
            uv_grid_size = int(self.config["brep"]["uv_grid_size"])
            graph = normalize_graph_features(
                graph,
                uv_grid_size=uv_grid_size,
                edge_u_grid_size=int(
                    self.config["brep"].get("edge_u_grid_size", uv_grid_size)
                ),
            )
        return graph


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def classification_metrics_from_confusion(
    confusion: torch.Tensor,
    *,
    class_names: list[str] | tuple[str, ...] | None = None,
    include_transition_binary: bool = True,
) -> dict[str, Any]:
    """Return exact face-level metrics, including per-class and transition metrics."""
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got shape={tuple(confusion.shape)}")

    counts = confusion.detach().to(device="cpu", dtype=torch.int64)
    num_classes = int(counts.shape[0])
    names = list(class_names or ())
    if len(names) != num_classes:
        if num_classes == 2:
            names = list(DEFAULT_BINARY_CLASS_NAMES)
        else:
            names = [
                DEFAULT_CLASS_NAMES[index]
                if index < len(DEFAULT_CLASS_NAMES)
                else f"class_{index}"
                for index in range(num_classes)
            ]

    matrix = [[int(value) for value in row] for row in counts.tolist()]
    support = [sum(row) for row in matrix]
    predicted = [sum(matrix[row][column] for row in range(num_classes)) for column in range(num_classes)]
    total = sum(support)
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        true_positive = matrix[index][index]
        false_positive = predicted[index] - true_positive
        false_negative = support[index] - true_positive
        precision = _safe_ratio(true_positive, predicted[index])
        recall = _safe_ratio(true_positive, support[index])
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "support": support[index],
                "predicted": predicted[index],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
            }
        )

    def macro(metric: str) -> float:
        return _safe_ratio(sum(float(row[metric]) for row in per_class), num_classes)

    def weighted(metric: str) -> float:
        return _safe_ratio(
            sum(float(row[metric]) * int(row["support"]) for row in per_class),
            total,
        )

    result: dict[str, Any] = {
        "faces": total,
        "accuracy": _safe_ratio(sum(matrix[index][index] for index in range(num_classes)), total),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "macro_iou": macro("iou"),
        "weighted_f1": weighted("f1"),
        "weighted_iou": weighted("iou"),
        "confusion_matrix": matrix,
        "per_class": per_class,
    }

    # The visualization reports transition detection with VBF and EBF merged.
    # Record the same view here so the document and webpage remain comparable.
    if include_transition_binary and num_classes >= 2:
        true_negative = matrix[0][0]
        false_positive = sum(matrix[0][1:])
        false_negative = sum(matrix[row][0] for row in range(1, num_classes))
        true_positive = sum(
            matrix[row][column]
            for row in range(1, num_classes)
            for column in range(1, num_classes)
        )
        transition_precision = _safe_ratio(true_positive, true_positive + false_positive)
        transition_recall = _safe_ratio(true_positive, true_positive + false_negative)
        result["transition_binary"] = {
            "accuracy": _safe_ratio(true_positive + true_negative, total),
            "precision": transition_precision,
            "recall": transition_recall,
            "f1": _safe_ratio(
                2.0 * transition_precision * transition_recall,
                transition_precision + transition_recall,
            ),
            "iou": _safe_ratio(true_positive, true_positive + false_positive + false_negative),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "faces": total,
        }
    return result


def _iter_files(path: Path, suffixes: set[str], *, recursive: bool) -> list[Path]:
    iterator = path.rglob("*") if recursive else path.iterdir()
    return sorted(
        candidate.resolve()
        for candidate in iterator
        if candidate.is_file() and candidate.suffix.lower() in suffixes
    )


def _path_key(path: Path) -> str:
    return path.with_suffix("").as_posix().casefold()


def _label_indexes(
    label_paths: Sequence[Path],
    label_root: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    relative_index: dict[str, list[Path]] = {}
    stem_index: dict[str, list[Path]] = {}
    for label_path in label_paths:
        relative_key = _path_key(label_path.relative_to(label_root))
        relative_index.setdefault(relative_key, []).append(label_path)
        stem_index.setdefault(label_path.stem.casefold(), []).append(label_path)
    return relative_index, stem_index


def pair_step_seg_paths(
    step_input: str | Path,
    seg_input: str | Path,
    *,
    step_extensions: Sequence[str] = DEFAULT_STEP_EXTENSIONS,
    recursive: bool = True,
    task: str = "seg",
) -> list[StepSegSample]:
    """Pair a STEP file/directory with a SEG/JSON or CLS file/directory."""
    step_path = Path(step_input).expanduser().resolve()
    seg_path = Path(seg_input).expanduser().resolve()
    if not step_path.exists():
        raise FileNotFoundError(f"STEP path does not exist: {step_path}")
    if not seg_path.exists():
        raise FileNotFoundError(f"SEG path does not exist: {seg_path}")

    normalized_step_extensions = {
        str(extension).lower()
        if str(extension).startswith(".")
        else f".{str(extension).lower()}"
        for extension in step_extensions
    }
    if step_path.is_file():
        if step_path.suffix.lower() not in normalized_step_extensions:
            raise ValueError(f"Not a configured STEP/STP file: {step_path}")
        step_paths = [step_path]
        step_root = step_path.parent
    elif step_path.is_dir():
        step_paths = _iter_files(step_path, normalized_step_extensions, recursive=recursive)
        step_root = step_path
    else:
        raise ValueError(f"STEP path is neither a file nor a directory: {step_path}")
    if not step_paths:
        extensions = ", ".join(sorted(normalized_step_extensions))
        raise ValueError(f"No STEP files with extensions [{extensions}] found under {step_path}.")

    label_extensions = CLASS_LABEL_EXTENSIONS if task == CLASSIFICATION else LABEL_EXTENSIONS
    label_description = "CLS" if task == CLASSIFICATION else "SEG or JSON"
    if seg_path.is_file():
        if seg_path.suffix.lower() not in label_extensions:
            raise ValueError(f"Label file must be {label_description}: {seg_path}")
        if len(step_paths) != 1:
            raise ValueError(
                f"A single label file cannot be paired with {len(step_paths)} STEP files. "
                "Pass a label directory instead."
            )
        relative_step = step_paths[0].relative_to(step_root).with_suffix("")
        return [StepSegSample(relative_step.as_posix(), step_paths[0], seg_path)]
    if not seg_path.is_dir():
        raise ValueError(f"Label path is neither a file nor a directory: {seg_path}")

    label_paths = _iter_files(seg_path, set(label_extensions), recursive=recursive)
    if not label_paths:
        raise ValueError(f"No {label_description} label files found under {seg_path}.")
    relative_index, stem_index = _label_indexes(label_paths, seg_path)

    samples: list[StepSegSample] = []
    missing: list[Path] = []
    for current_step in step_paths:
        relative_step = current_step.relative_to(step_root)
        candidates = relative_index.get(_path_key(relative_step), [])
        if not candidates:
            candidates = stem_index.get(current_step.stem.casefold(), [])
        unique_candidates = sorted(set(candidates))
        if not unique_candidates:
            missing.append(current_step)
            continue
        if len(unique_candidates) > 1:
            matches = ", ".join(str(candidate) for candidate in unique_candidates[:5])
            suffix = " ..." if len(unique_candidates) > 5 else ""
            raise ValueError(f"STEP file {current_step} matched multiple labels: {matches}{suffix}")
        samples.append(
            StepSegSample(
                sample_id=relative_step.with_suffix("").as_posix(),
                step_path=current_step,
                seg_path=unique_candidates[0],
            )
        )

    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        missing_description = "CLS" if task == CLASSIFICATION else "SEG/JSON"
        raise FileNotFoundError(
            f"No matching {missing_description} label was found for {len(missing)} STEP files: "
            f"{examples}{suffix}"
        )
    return samples


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _load_evaluation_config(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    if args.config:
        config = load_experiment_config(args.config, args.data_config, args.override)
    else:
        checkpoint = _torch_load(checkpoint_path)
        checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if not isinstance(checkpoint_config, dict):
            raise ValueError(
                "The checkpoint has no embedded config. Supply the training config with --config."
            )
        config = apply_overrides(copy.deepcopy(checkpoint_config), args.override)
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        config["train"]["num_workers"] = int(args.num_workers)
    if args.device is not None:
        config["train"]["device"] = args.device
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a finetuned checkpoint either on paired STEP/SEG paths or on a cached split."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training YAML. By default the checkpoint's embedded config is used.",
    )
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--step",
        "--step-path",
        dest="step_path",
        default=None,
        help="A STEP/STP file or a directory containing STEP/STP files.",
    )
    parser.add_argument(
        "--seg",
        "--seg-path",
        "--label",
        "--label-path",
        dest="seg_path",
        default=None,
        help="A matching SEG/JSON or CLS file/directory. Layouts are paired by relative path.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recursively scan STEP and SEG directories.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default="evaluation_metrics.json")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    direct_mode = args.step_path is not None or args.seg_path is not None
    if direct_mode and (args.step_path is None or args.seg_path is None):
        raise ValueError("Direct evaluation requires both --step and --seg.")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    config = _load_evaluation_config(args, checkpoint_path)
    configured_task = task_type(config)

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config, enforce_cuda_requirement=False)
    samples: list[StepSegSample] | None = None
    if direct_mode:
        samples = pair_step_seg_paths(
            args.step_path,
            args.seg_path,
            step_extensions=config.get("data", {}).get(
                "step_extensions", DEFAULT_STEP_EXTENSIONS
            ),
            recursive=not args.no_recursive,
            task=configured_task,
        )
        dataset = StepSegDataset(config, samples)
        dataloader = DataLoader(
            dataset,
            batch_size=int(config["train"].get("batch_size", 1)),
            shuffle=False,
            num_workers=int(config["train"].get("num_workers", 0)),
            collate_fn=collate_graphs,
        )
    else:
        dataloader = build_dataloader(config, split=args.split, shuffle=False, distributed=False)
    face_dim, edge_dim = feature_dims(config)
    model = build_finetune_model(config, face_dim, edge_dim).to(device)
    checkpoint_epoch = load_checkpoint(checkpoint_path, model=model, optimizer=None, device=device)
    model.eval()

    num_classes = int(config["model"]["num_classes"])
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        description = "evaluate STEP/SEG" if direct_mode else f"evaluate {args.split}"
        for batch in tqdm(dataloader, desc=description):
            batch = batch.to(device)
            check_finite_batch(batch)
            probabilities = predict_finetune_probabilities(model, batch, config)
            confusion.add_(finetune_confusion_matrix(probabilities, batch, config).cpu())

    configured_names = config.get("labels", {}).get("names")
    metrics = classification_metrics_from_confusion(
        confusion,
        class_names=configured_names,
        include_transition_binary=bool(
            config.get("labels", {}).get("include_transition_binary", True)
            and configured_task != CLASSIFICATION
        ),
    )
    if configured_task == CLASSIFICATION:
        metrics["samples"] = metrics.pop("faces")
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "config": str(args.config) if args.config else "checkpoint_embedded_config",
        "task": configured_task,
        "input_mode": (
            ("step_cls" if configured_task == CLASSIFICATION else "step_seg")
            if direct_mode
            else "cached_split"
        ),
        "step_path": str(Path(args.step_path).expanduser().resolve()) if direct_mode else None,
        "seg_path": str(Path(args.seg_path).expanduser().resolve()) if direct_mode else None,
        "split": None if direct_mode else args.split,
        "split_file": (
            None if direct_mode else str(config["data"].get(f"{args.split}_split"))
        ),
        "cache_dir": (
            None
            if direct_mode
            else str(
                config["data"].get("cache_dirs", {}).get(
                    args.split, config["data"].get("cache_dir")
                )
            )
        ),
        "samples": len(dataloader.dataset),
        "sample_pairs": (
            [
                {
                    "sample_id": sample.sample_id,
                    "step_path": str(sample.step_path),
                    "seg_path": str(sample.seg_path),
                }
                for sample in samples
            ]
            if samples is not None
            else None
        ),
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": int(config["train"].get("num_workers", 0)),
        "device": str(device),
        "metrics": metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
