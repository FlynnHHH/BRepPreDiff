#!/usr/bin/env python3
"""Generate and atomically install finetune_test visualization results."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "finetune_test_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite-dir")
    source.add_argument("--run-dir")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-dir", default="tools/visualize/results")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_staging(staging_dir: Path, *, expected_samples: int) -> dict[str, Any]:
    manifest_path = staging_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Visualization manifest was not generated: {manifest_path}")
    manifest = load_json(manifest_path)
    samples = manifest.get("samples", [])
    if len(samples) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} visualization samples, found {len(samples)}"
        )
    if int(manifest.get("evaluated_samples", 0)) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} evaluated samples, "
            f"found {manifest.get('evaluated_samples')}"
        )
    missing: list[str] = []
    for record in samples:
        for key in ("input_ply", "semantic_ply"):
            filename = record.get(key)
            if not filename or Path(filename).name != filename or not (staging_dir / filename).is_file():
                missing.append(str(filename))
    if missing:
        raise FileNotFoundError(
            f"Visualization staging is missing {len(missing)} PLY files; first={missing[:5]}"
        )
    return manifest


def archive_manifest_results(manifest_path: Path, archive_dir: Path) -> int:
    if not manifest_path.exists():
        return 0
    manifest = load_json(manifest_path)
    results_dir = manifest_path.parent
    moved = 0
    for record in manifest.get("samples", []):
        for key in ("input_ply", "semantic_ply"):
            filename = record.get(key)
            if not filename or Path(filename).name != filename:
                continue
            source = results_dir / filename
            if source.is_file():
                source.replace(archive_dir / filename)
                moved += 1
    manifest_path.replace(archive_dir / manifest_path.name)
    return moved


def install_staging(staging_dir: Path, results_dir: Path) -> list[str]:
    installed: list[str] = []
    for source in sorted(staging_dir.iterdir()):
        if not source.is_file():
            continue
        destination = results_dir / source.name
        source.replace(destination)
        installed.append(source.name)
    return installed


def main() -> None:
    args = parse_args()
    results_dir = (ROOT / args.results_dir).resolve()
    if args.run_dir:
        source_dir = (ROOT / args.run_dir).resolve()
        test_evaluation = load_json(source_dir / "test_metrics.json")
        checkpoint = test_evaluation.get("checkpoint") or str(
            source_dir / "checkpoints" / "best.pt"
        )
        checkpoint_epoch = test_evaluation.get("checkpoint_epoch")
        config_path = Path(args.config).resolve() if args.config else source_dir / "config.yaml"
        inference_overrides: list[str] = []
    else:
        source_dir = (ROOT / args.suite_dir).resolve()
        state = load_json(source_dir / "state.json")
        if state.get("status") != "complete":
            raise RuntimeError(f"Suite is not complete: status={state.get('status')!r}")
        baseline = state.get("experiments", {}).get("baseline_default", {})
        checkpoint = baseline.get("best_checkpoint")
        test_evaluation = baseline.get("test_evaluation")
        if not checkpoint or not test_evaluation:
            raise RuntimeError("baseline_default is missing a best checkpoint or test evaluation")
        checkpoint_epoch = baseline.get("best_epoch")
        config_path = Path(args.config or "configs/finetune_diffloss.yaml")
        inference_overrides = [
            "label_diffusion.prediction_type=x_start_epsilon",
            "label_diffusion.x_start_loss_weight=1.0",
            "label_diffusion.epsilon_loss_weight=0.5",
        ]
    expected_samples = int(test_evaluation["samples"])

    checkpoint = Path(checkpoint).resolve()
    config_path = config_path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Visualization config does not exist: {config_path}")

    staging_dir = source_dir / "visualization_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "brepprediff.inference.finetune_visualize",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "test",
        "--output-dir",
        str(staging_dir),
        "--manifest-name",
        MANIFEST_NAME,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    for override in inference_overrides:
        command.extend(("--override", override))
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    log_path = source_dir / "logs" / "finetune_test_visualization.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"Visualization generation failed with returncode={process.returncode}; see {log_path}"
        )

    manifest = validate_staging(staging_dir, expected_samples=expected_samples)
    suite_confusion = test_evaluation["metrics"]["confusion_matrix"]
    visualization_confusion = manifest["detailed_metrics"]["confusion_matrix"]
    if visualization_confusion != suite_confusion:
        raise ValueError(
            "Visualization predictions do not match baseline test evaluation confusion matrix: "
            f"visualization={visualization_confusion} suite={suite_confusion}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = results_dir / "archive" / datetime.now().strftime(
        "%Y%m%d-%H%M%S_before_finetune_test"
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived_files = 0
    for old_name in ("original_testset_manifest.json", MANIFEST_NAME):
        archived_files += archive_manifest_results(results_dir / old_name, archive_dir)
    installed = install_staging(staging_dir, results_dir)

    record = {
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "samples": expected_samples,
        "archived_files": archived_files,
        "archive_dir": str(archive_dir),
        "installed_files": len(installed),
        "metrics": manifest.get("metrics"),
        "detailed_metrics": manifest.get("detailed_metrics"),
    }
    install_record = results_dir / "finetune_test_install.json"
    install_record.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
