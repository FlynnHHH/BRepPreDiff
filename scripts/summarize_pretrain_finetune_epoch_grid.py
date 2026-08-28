#!/usr/bin/env python3
"""Summarize exact-epoch evaluations from the pretrain/finetune epoch grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PRETRAIN_EPOCHS = (20, 50, 100, 150)
FINETUNE_EPOCHS = (10, 20, 50, 100, 150, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def load_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(results_root.glob("*/pretrain_*/finetune_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        pretrain_epoch = int(path.parent.name.removeprefix("pretrain_"))
        finetune_epoch = int(path.stem.removeprefix("finetune_"))
        rows.append(
            {
                "task": path.parents[1].name,
                "pretrain_epoch": pretrain_epoch,
                "finetune_epoch": finetune_epoch,
                "checkpoint_epoch": int(payload["checkpoint_epoch"]),
                "samples": int(payload["samples"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_iou": float(metrics["macro_iou"]),
                "result_path": str(path.resolve()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "task",
        "pretrain_epoch",
        "finetune_epoch",
        "checkpoint_epoch",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_iou",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value: object | None) -> str:
    return "--" if value is None else f"{100.0 * float(value):.4f}"


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {
        (str(row["task"]), int(row["pretrain_epoch"]), int(row["finetune_epoch"])): row
        for row in rows
    }
    tasks = sorted({str(row["task"]) for row in rows})
    lines = [
        "# Pretrain / finetune epoch grid",
        "",
        "Each cell is test accuracy (%) at the exact fine-tuning epoch. The matrix is an",
        "ablation table, not a basis for tuning on the test split; select the final setting",
        "with validation metrics and report its test result once.",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                f"## {task}",
                "",
                "| Pretrain epoch \\ Finetune epoch | "
                + " | ".join(str(epoch) for epoch in FINETUNE_EPOCHS)
                + " |",
                "|---:|" + "---:|" * len(FINETUNE_EPOCHS),
            ]
        )
        for pretrain_epoch in PRETRAIN_EPOCHS:
            values = [
                format_percent(
                    lookup.get((task, pretrain_epoch, finetune_epoch), {}).get("accuracy")
                )
                for finetune_epoch in FINETUNE_EPOCHS
            ]
            lines.append(f"| {pretrain_epoch} | " + " | ".join(values) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_root)
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    print(f"summarized {len(rows)} evaluations")


if __name__ == "__main__":
    main()
