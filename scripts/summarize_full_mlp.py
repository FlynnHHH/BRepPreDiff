#!/usr/bin/env python3
"""Summarize full-data MLP fine-tuning runs."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("TASK", "RUN_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for task, configured_path in args.run:
        run_dir = Path(configured_path)
        result = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        metrics = result["metrics"]
        rows.append(
            {
                "task": task,
                "run": run_dir.resolve(),
                "epoch": int(result["checkpoint_epoch"]),
                "samples": int(result["samples"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_iou": float(metrics["macro_iou"]),
                "sha256": sha256(run_dir / "checkpoints" / "best.pt"),
            }
        )

    lines = [
        "# Inductive 9-source encoder：全量下游 MLP 200-epoch 结果",
        "",
        "预训练从同配置 epoch 50 checkpoint 续训至 epoch 100；预训练语料仅含九个来源的",
        "train split，不含 FabWave，也不含任何下游 val/test。下游使用全量 train split、",
        "MLP head、随机种子 42、200 epochs，并按 validation accuracy 选择 best checkpoint。",
        "分类任务使用 Mean+Max pooling。预训练和微调均上传至 W&B。",
        "",
        f"Pretrain checkpoint: `{args.pretrain_checkpoint.resolve()}`",
        f"SHA-256: `{sha256(args.pretrain_checkpoint)}`",
        "",
        "| Task | Best epoch | Test samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['epoch']} | {row['samples']} | "
            f"{100 * row['accuracy']:.4f} | {100 * row['macro_f1']:.4f} | "
            f"{100 * row['weighted_f1']:.4f} | {100 * row['macro_iou']:.4f} |"
        )
    lines.extend(["", "## Run artifacts", ""])
    for row in rows:
        lines.append(f"- {row['task']}: `{row['run']}`; SHA-256 `{row['sha256']}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
