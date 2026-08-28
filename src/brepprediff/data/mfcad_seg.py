from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm


MFCAD_SPLITS = ("train", "val", "test")
MFCAD_LABEL_MIN = 0
MFCAD_LABEL_MAX = 24
MFCAD_FEATURE_CLASSES = 24
MFCAD_SEGMENTATION_CLASSES = 25
MFCAD_STOCK_LABEL = 24
MFCAD_CLASS_NAMES = (
    "Chamfer",
    "Through hole",
    "Triangular passage",
    "Rectangular passage",
    "6-sided passage",
    "Triangular through slot",
    "Rectangular through slot",
    "Circular through slot",
    "Rectangular through step",
    "2-sided through step",
    "Slanted through step",
    "O-ring",
    "Blind hole",
    "Triangular pocket",
    "Rectangular pocket",
    "6-sided pocket",
    "Circular end pocket",
    "Rectangular blind slot",
    "Vertical circular end blind slot",
    "Horizontal circular end blind slot",
    "Triangular blind step",
    "Circular blind step",
    "Rectangular blind step",
    "Round",
    "Stock",
)


@dataclass(frozen=True)
class SegConversionResult:
    step_path: Path
    seg_path: Path
    labels: tuple[int, ...]
    written: bool


@dataclass(frozen=True)
class SegConversionSummary:
    models: int
    written: int
    unchanged: int
    faces: int
    label_counts: dict[int, int]


def _parse_mfcad_label(name: str, path: Path, face_index: int) -> int:
    value = name.strip()
    try:
        label = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid MFCAD++ face label {value!r} at face {face_index} in {path}"
        ) from exc
    if str(label) != value or not MFCAD_LABEL_MIN <= label <= MFCAD_LABEL_MAX:
        raise ValueError(
            f"MFCAD++ face label {value!r} is outside canonical integer ids "
            f"{MFCAD_LABEL_MIN}..{MFCAD_LABEL_MAX} at face {face_index} in {path}"
        )
    return label


def extract_mfcad_face_labels(step_path: str | Path) -> list[int]:
    """Read labels in the exact OCC face order consumed by BRepPreDiff.

    MFCAD++ stores the class id in each STEP ``ADVANCED_FACE`` name.  Reading
    those entities in textual order is not sufficient: OpenCascade can reorder
    faces while transferring the STEP topology.  The transfer map below links
    every TopoDS face back to its source ADVANCED_FACE entity.
    """

    from OCC.Core.StepShape import StepShape_AdvancedFace

    from brepprediff.brep.occ_extractor import _occ_imports

    path = Path(step_path)
    occ = _occ_imports()
    reader = occ.STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != occ.IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {path}")
    reader.TransferRoots()

    face_map = occ.TopTools_IndexedMapOfShape()
    occ.TopExp.MapShapes(reader.OneShape(), occ.TopAbs_FACE, face_map)
    face_count = face_map.Extent() if hasattr(face_map, "Extent") else face_map.Size()
    if face_count == 0:
        raise ValueError(f"The STEP model contains no faces: {path}")

    transfer_reader = reader.WS().TransferReader()
    labels: list[int] = []
    for face_index in range(1, face_count + 1):
        entity = transfer_reader.EntityFromShapeResult(face_map.FindKey(face_index), 1)
        if entity is None:
            raise ValueError(
                f"Cannot map OCC face {face_index - 1} to a STEP entity in {path}"
            )
        advanced_face = StepShape_AdvancedFace.DownCast(entity)
        if advanced_face is None:
            entity_type = entity.DynamicType().Name()
            raise ValueError(
                f"OCC face {face_index - 1} maps to {entity_type}, not ADVANCED_FACE, in {path}"
            )
        encoded_name = advanced_face.Name()
        if encoded_name is None:
            raise ValueError(f"ADVANCED_FACE {face_index - 1} has no label name in {path}")
        name = (
            encoded_name.ToCString()
            if hasattr(encoded_name, "ToCString")
            else str(encoded_name)
        )
        labels.append(_parse_mfcad_label(name, path, face_index - 1))
    return labels


def _seg_text(labels: Iterable[int]) -> str:
    return "".join(f"{int(label)}\n" for label in labels)


def _lines_text(lines: Iterable[str]) -> str:
    return "".join(f"{line}\n" for line in lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert_mfcad_step(
    step_path: str | Path,
    *,
    step_root: str | Path,
    seg_root: str | Path,
    overwrite: bool = False,
) -> SegConversionResult:
    step_path = Path(step_path)
    step_root = Path(step_root)
    seg_path = Path(seg_root) / step_path.relative_to(step_root).with_suffix(".seg")
    labels = tuple(extract_mfcad_face_labels(step_path))
    expected = _seg_text(labels)

    if seg_path.exists() and not overwrite:
        actual = seg_path.read_text(encoding="utf-8")
        if actual != expected:
            raise FileExistsError(
                f"Existing SEG differs from STEP labels: {seg_path}; "
                "rerun with --overwrite to replace it"
            )
        return SegConversionResult(step_path, seg_path, labels, written=False)

    _write_text_atomic(seg_path, expected)
    return SegConversionResult(step_path, seg_path, labels, written=True)


def _step_files(step_root: Path, splits: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        split_dir = step_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing MFCAD++ STEP split directory: {split_dir}")
        paths.extend(sorted(split_dir.glob("*.step")))
    return paths


def convert_mfcad_dataset(
    step_root: str | Path,
    seg_root: str | Path,
    *,
    splits: Sequence[str] = MFCAD_SPLITS,
    workers: int = 8,
    overwrite: bool = False,
) -> SegConversionSummary:
    step_root = Path(step_root).resolve()
    seg_root = Path(seg_root).resolve()
    step_paths = _step_files(step_root, splits)
    if not step_paths:
        raise FileNotFoundError(f"No STEP files found under {step_root}")

    results: list[SegConversionResult] = []
    worker_count = max(1, int(workers))
    if worker_count == 1:
        for step_path in tqdm(step_paths, desc="MFCAD++ STEP -> SEG"):
            results.append(
                convert_mfcad_step(
                    step_path,
                    step_root=step_root,
                    seg_root=seg_root,
                    overwrite=overwrite,
                )
            )
    else:
        # OpenCascade transfer objects are not thread-safe. Separate worker
        # processes also let STEP parsing scale across CPU cores.
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    convert_mfcad_step,
                    step_path,
                    step_root=step_root,
                    seg_root=seg_root,
                    overwrite=overwrite,
                )
                for step_path in step_paths
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"MFCAD++ STEP -> SEG x{worker_count}",
            ):
                results.append(future.result())

    counts: Counter[int] = Counter()
    for result in results:
        counts.update(result.labels)
    written = sum(result.written for result in results)
    faces = sum(len(result.labels) for result in results)
    return SegConversionSummary(
        models=len(results),
        written=written,
        unchanged=len(results) - written,
        faces=faces,
        label_counts=dict(sorted(counts.items())),
    )


def write_brepprediff_split_files(
    dataset_root: str | Path,
    step_root: str | Path,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = MFCAD_SPLITS,
) -> dict[str, Path]:
    """Write split entries relative to the shared STEP root used by BRepPreDiff."""

    dataset_root = Path(dataset_root).resolve()
    step_root = Path(step_root).resolve()
    output_dir = Path(output_dir).resolve()
    outputs: dict[str, Path] = {}

    for split in splits:
        source = dataset_root / f"{split}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing MFCAD++ split file: {source}")
        model_ids = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"Duplicate model ids in {source}")

        entries: list[str] = []
        listed_paths: set[Path] = set()
        for model_id in model_ids:
            relative_name = Path(model_id)
            if relative_name.is_absolute() or ".." in relative_name.parts:
                raise ValueError(f"Unsafe model id {model_id!r} in {source}")
            if relative_name.suffix.lower() != ".step":
                relative_name = relative_name.with_suffix(".step")
            step_path = step_root / split / relative_name
            if not step_path.is_file():
                raise FileNotFoundError(f"Split entry has no STEP file: {step_path}")
            listed_paths.add(step_path.resolve())
            entries.append(step_path.relative_to(step_root).as_posix())

        actual_paths = {path.resolve() for path in (step_root / split).glob("*.step")}
        if listed_paths != actual_paths:
            missing = len(actual_paths - listed_paths)
            extra = len(listed_paths - actual_paths)
            raise ValueError(
                f"MFCAD++ {split} split does not match its STEP directory: "
                f"unlisted={missing}, missing={extra}"
            )

        output = output_dir / f"{split}.txt"
        _write_text_atomic(output, _lines_text(entries))
        outputs[split] = output

    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-face MFCAD++ labels from STEP ADVANCED_FACE names into SEG files."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="MFCAD++ directory containing split txt files.",
    )
    parser.add_argument(
        "--step-root",
        required=True,
        help="Directory containing train/val/test STEP folders.",
    )
    parser.add_argument(
        "--seg-root",
        required=True,
        help="Output directory for matching SEG folders.",
    )
    parser.add_argument(
        "--brepprediff-splits-dir",
        help="Optional output directory for BRepPreDiff-compatible prefixed split files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=MFCAD_SPLITS,
        default=list(MFCAD_SPLITS),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = convert_mfcad_dataset(
        args.step_root,
        args.seg_root,
        splits=args.splits,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    if args.brepprediff_splits_dir:
        outputs = write_brepprediff_split_files(
            args.dataset_root,
            args.step_root,
            args.brepprediff_splits_dir,
            splits=args.splits,
        )
        for split, output in outputs.items():
            print(f"BRepPreDiff split {split}: {output}")

    histogram = ", ".join(f"{label}:{count}" for label, count in summary.label_counts.items())
    print(
        "MFCAD++ SEG ready: "
        f"models={summary.models} written={summary.written} unchanged={summary.unchanged} "
        f"faces={summary.faces} labels=[{histogram}]"
    )


if __name__ == "__main__":
    main()
