#!/usr/bin/env python3
"""Compare matched FabWave MLP and DiffLoss test results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> dict[str, object]:
    result = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    return {
        "run": path,
        "best_epoch": int(result["checkpoint_epoch"]),
        "samples": int(result["samples"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "macro_iou": float(metrics["macro_iou"]),
        "sha256": checkpoint_sha256(path / "checkpoints" / "best.pt"),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-run", type=Path, required=True)
    parser.add_argument("--diffloss-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mlp = load_run(args.mlp_run.resolve())
    diff = load_run(args.diffloss_run.resolve())
    deltas = {
        key: float(diff[key]) - float(mlp[key])
        for key in ("accuracy", "macro_f1", "weighted_f1", "macro_iou")
    }
    winner = "DiffLoss" if deltas["accuracy"] > 0 else "MLP" if deltas["accuracy"] < 0 else "Tie"

    lines = [
        "# FabWave 新 OCC 特征：MLP 与 DiffLoss 对照",
        "",
        "两组实验使用相同的新 OCC 特征缓存、Edge Update Attention encoder 预训练权重、",
        "随机种子、Mean+Max pooling、200-epoch 预算、batch size 64、梯度累积 4，并按验证集",
        "accuracy 选择 best checkpoint。差值定义为 `DiffLoss - MLP`。",
        "",
        "| Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("MLP", mlp), ("DiffLoss", diff)):
        lines.append(
            f"| {name} | {row['best_epoch']} | {row['samples']} | {percent(float(row['accuracy']))} | "
            f"{percent(float(row['macro_f1']))} | {percent(float(row['weighted_f1']))} | "
            f"{percent(float(row['macro_iou']))} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Accuracy (pp) | Macro-F1 (pp) | Weighted-F1 (pp) | mIoU (pp) |",
            "|---|---:|---:|---:|---:|",
            f"| DiffLoss - MLP | {signed_pp(deltas['accuracy'])} | {signed_pp(deltas['macro_f1'])} | "
            f"{signed_pp(deltas['weighted_f1'])} | {signed_pp(deltas['macro_iou'])} |",
            "",
            f"Accuracy winner: **{winner}**.",
            "",
            f"- MLP run: `{mlp['run']}`; best.pt SHA-256 `{mlp['sha256']}`",
            f"- DiffLoss run: `{diff['run']}`; best.pt SHA-256 `{diff['sha256']}`",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

