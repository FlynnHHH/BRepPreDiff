#!/usr/bin/env python3
"""Summarize multi-supervision-ratio test results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", required=True)
    args = parser.parse_args()
    rows = []
    for result in sorted(args.results_root.glob("*/r*/test_metrics.json")):
        payload = json.loads(result.read_text())
        audit = json.loads((result.parent / "selection.audit.json").read_text())
        metrics = payload["metrics"]
        rows.append({
            "task": audit["task"],
            "ratio_percent": audit["requested_ratio_percent"],
            "train_samples": audit["selected_samples"],
            "test_samples": payload["samples"],
            "best_epoch": payload["checkpoint_epoch"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_iou": metrics["macro_iou"],
            "result_path": str(result.resolve()),
        })
    if not rows:
        raise ValueError(f"No results below {args.results_root}")
    rows.sort(key=lambda row: (row["task"], float(row["ratio_percent"])))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Multi-supervision-ratio fine-tuning results",
        "",
        "Seed 42; MLP full fine-tuning for 200 epochs; full validation every 5 epochs and",
        "unchanged test sets; best checkpoint selected by",
        "validation accuracy. Ratios count labeled CAD models.",
        "The 100% endpoints reuse matching earlier runs with validation every epoch; all ratios",
        "below 100% are newly run with validation every 5 epochs.",
        f"Encoder checkpoint: `{args.pretrain_checkpoint}`.",
        "",
        "| Benchmark | Ratio | Train CADs | Test CADs | Best epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {float(row['ratio_percent']):g}% | {row['train_samples']} | "
            f"{row['test_samples']} | {row['best_epoch']} | {100*row['accuracy']:.4f} | "
            f"{100*row['macro_f1']:.4f} | {100*row['weighted_f1']:.4f} | {100*row['macro_iou']:.4f} |"
        )
    lines.extend([
        "",
        "## Data-efficiency summary",
        "",
        "Gaps are 3% minus 100% in percentage points; negative values indicate remaining",
        "full-supervision headroom.",
        "",
        "| Benchmark | Accuracy gain 0.1%→0.5% | Accuracy at 3% | Accuracy gap | Macro-F1 at 3% | Macro-F1 gap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    by_task = {
        task: {float(row["ratio_percent"]): row for row in rows if row["task"] == task}
        for task in sorted({row["task"] for row in rows})
    }
    for task, ratio_rows in by_task.items():
        low, half, three, full = (ratio_rows[value] for value in (0.1, 0.5, 3.0, 100.0))
        lines.append(
            f"| {task} | {100*(half['accuracy']-low['accuracy']):+.4f} | "
            f"{100*three['accuracy']:.4f} | {100*(three['accuracy']-full['accuracy']):+.4f} | "
            f"{100*three['macro_f1']:.4f} | {100*(three['macro_f1']-full['macro_f1']):+.4f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")
    print(f"summarized {len(rows)} evaluations to {args.markdown}")


if __name__ == "__main__":
    main()
