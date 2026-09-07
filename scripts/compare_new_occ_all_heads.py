#!/usr/bin/env python3
"""Compare matched MLP and DiffLoss runs across all downstream tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_run(path: Path) -> dict[str, object]:
    result = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    return {
        "run": path.resolve(),
        "epoch": int(result["checkpoint_epoch"]),
        "samples": int(result["samples"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "macro_iou": float(metrics["macro_iou"]),
        "sha256": sha256(path / "checkpoints" / "best.pt"),
    }


def pct(value: float) -> str:
    return f"{value * 100.0:.4f}"


def delta(value: float) -> str:
    return f"{value * 100.0:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("TASK", "MLP_RUN", "DIFFLOSS_RUN"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="新 OCC 特征：全部下游任务 MLP 与 DiffLoss 对照")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-description", default="batch size 64、梯度累积 4")
    parser.add_argument(
        "--pretrain-description",
        default="同一 Edge Update Attention 新特征预训练 encoder",
    )
    args = parser.parse_args()

    rows: list[tuple[str, dict[str, object], dict[str, object]]] = []
    for task, mlp_path, diff_path in args.pair:
        rows.append((task, load_run(Path(mlp_path)), load_run(Path(diff_path))))

    lines = [
        f"# {args.title}",
        "",
        f"所有实验使用{args.pretrain_description}、相同数据划分、",
        f"随机种子 42、{args.epochs} epochs、{args.batch_description}，并按 validation accuracy",
        "选择 best checkpoint。分类任务统一使用 Mean+Max pooling。",
        "",
        "| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, mlp, diff in rows:
        for head, row in (("MLP", mlp), ("DiffLoss", diff)):
            lines.append(
                f"| {task} | {head} | {row['epoch']} | {row['samples']} | "
                f"{pct(float(row['accuracy']))} | {pct(float(row['macro_f1']))} | "
                f"{pct(float(row['weighted_f1']))} | {pct(float(row['macro_iou']))} |"
            )

    lines.extend(
        [
            "",
            "差值定义为 `DiffLoss - MLP`，单位为百分点。",
            "",
            "| Task | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for task, mlp, diff in rows:
        differences = {
            key: float(diff[key]) - float(mlp[key])
            for key in ("accuracy", "macro_f1", "weighted_f1", "macro_iou")
        }
        winner = (
            "DiffLoss"
            if differences["accuracy"] > 0
            else "MLP"
            if differences["accuracy"] < 0
            else "Tie"
        )
        lines.append(
            f"| {task} | {delta(differences['accuracy'])} | {delta(differences['macro_f1'])} | "
            f"{delta(differences['weighted_f1'])} | {delta(differences['macro_iou'])} | {winner} |"
        )

    lines.extend(["", "## Run artifacts", ""])
    for task, mlp, diff in rows:
        lines.append(f"- {task} / MLP: `{mlp['run']}`; SHA-256 `{mlp['sha256']}`")
        lines.append(f"- {task} / DiffLoss: `{diff['run']}`; SHA-256 `{diff['sha256']}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
