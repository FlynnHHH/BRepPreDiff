#!/usr/bin/env python3
"""Summarize few-shot test evaluations as CSV and Markdown."""

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
    args = parser.parse_args()

    rows = []
    result_paths = list(args.results_root.glob("*/*shot/test.json"))
    result_paths.extend(args.results_root.glob("*/diffloss/*shot/test.json"))
    result_paths.extend(args.results_root.glob("*/diffloss_aligned/*shot/test.json"))
    for path in sorted(result_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        audit_path = next(path.parent.glob("*.audit.json"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        variant = path.parent.parent.name
        is_diffloss = variant in {"diffloss", "diffloss_aligned"}
        task = path.parents[2].name if is_diffloss else path.parents[1].name
        head = "MLP"
        if variant == "diffloss_aligned":
            head = "DiffLoss"
        elif variant == "diffloss":
            head = "DiffLoss-xstart-epsilon" if task == "solidletters_cls" else "DiffLoss"
        rows.append(
            {
                "task": task,
                "head": head,
                "shot": int(path.parent.name.removesuffix("shot")),
                "train_samples": int(audit["selected_unique_samples"]),
                "test_samples": int(payload["samples"]),
                "checkpoint_epoch": int(payload["checkpoint_epoch"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "macro_iou": float(metrics["macro_iou"]),
                "result_path": str(path.resolve()),
            }
        )
    rows.sort(key=lambda row: (str(row["task"]), int(row["shot"]), str(row["head"])))
    if not rows:
        raise ValueError(f"No results found under {args.results_root}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 50-epoch encoder few-shot results",
        "",
        "Seed 42; MLP and DiffLoss heads; 100 fine-tuning epochs; full validation split selects the best",
        "checkpoint, which is evaluated once on the unchanged test split.",
        "Classification uses Mean+Max graph pooling. `DiffLoss` uses the cross-task aligned setting:",
        "x_start prediction, epsilon loss weight 0, one DDIM sampling step, and temperature 0.75.",
        "`DiffLoss-xstart-epsilon` is the retained legacy SolidLetters variant (epsilon weight 0.5,",
        "25 DDIM sampling steps, and temperature 1.0) and is excluded from the aligned head comparison.",
        "",
        "| Task | Head | Shot | Unique train CADs | Test CADs | Best epoch | Accuracy (%) | Macro F1 (%) | Macro IoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['head']} | {row['shot']} | {row['train_samples']} | {row['test_samples']} | "
            f"{row['checkpoint_epoch']} | {100 * row['accuracy']:.4f} | "
            f"{100 * row['macro_f1']:.4f} | {100 * row['macro_iou']:.4f} |"
        )
    lookup = {
        (str(row["task"]), int(row["shot"]), str(row["head"])): row
        for row in rows
    }
    comparisons = []
    for task in sorted({str(row["task"]) for row in rows}):
        for shot in sorted({int(row["shot"]) for row in rows if str(row["task"]) == task}):
            mlp = lookup.get((task, shot, "MLP"))
            diffloss = lookup.get((task, shot, "DiffLoss"))
            if mlp is not None and diffloss is not None:
                comparisons.append((task, shot, mlp, diffloss))
    if comparisons:
        lines.extend(
            [
                "",
                "## Head comparison",
                "",
                "Deltas are DiffLoss minus MLP in percentage points; positive values favor DiffLoss.",
                "",
                "| Task | Shot | Accuracy delta | Macro F1 delta | Macro IoU delta |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for task, shot, mlp, diffloss in comparisons:
            lines.append(
                f"| {task} | {shot} | "
                f"{100 * (float(diffloss['accuracy']) - float(mlp['accuracy'])):+.4f} | "
                f"{100 * (float(diffloss['macro_f1']) - float(mlp['macro_f1'])):+.4f} | "
                f"{100 * (float(diffloss['macro_iou']) - float(mlp['macro_iou'])):+.4f} |"
            )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized {len(rows)} evaluations: {args.markdown}")


if __name__ == "__main__":
    main()
