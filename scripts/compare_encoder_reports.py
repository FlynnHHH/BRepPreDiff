#!/usr/bin/env python3
"""Compare downstream run metrics against the first result table in a report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_iou")
DISPLAY = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "macro_iou": "mIoU",
}


def parse_baseline_report(path: Path) -> dict[tuple[str, str], dict[str, float | int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "| Task | Head | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |"
    try:
        start = lines.index(header) + 2
    except ValueError as exc:
        raise ValueError(f"Could not find downstream result table in {path}") from exc

    rows: dict[tuple[str, str], dict[str, float | int]] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 8:
            raise ValueError(f"Unexpected result row in {path}: {line}")
        task, head = cells[0], cells[1]
        rows[(task, head)] = {
            "epoch": int(cells[2]),
            "samples": int(cells[3]),
            "accuracy": float(cells[4]) / 100.0,
            "macro_f1": float(cells[5]) / 100.0,
            "weighted_f1": float(cells[6]) / 100.0,
            "macro_iou": float(cells[7]) / 100.0,
        }
    return rows


def load_run(path: Path) -> dict[str, float | int | str]:
    payload = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {
        "run": str(path.resolve()),
        "epoch": int(payload["checkpoint_epoch"]),
        "samples": int(payload["samples"]),
        **{metric: float(metrics[metric]) for metric in METRICS},
    }


def pct(value: float) -> str:
    return f"{value * 100.0:.4f}"


def delta(value: float) -> str:
    return f"{value * 100.0:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("TASK", "HEAD", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--candidate-name", default="Geometry-only encoder")
    parser.add_argument("--baseline-name", default="Full-loss encoder")
    args = parser.parse_args()

    baseline = parse_baseline_report(args.baseline_report)
    rows = []
    for task, head, run_path in args.run:
        key = (task, head)
        if key not in baseline:
            raise KeyError(f"Missing baseline row for {task} / {head}")
        candidate = load_run(Path(run_path))
        reference = baseline[key]
        if int(candidate["samples"]) != int(reference["samples"]):
            raise ValueError(
                f"Sample mismatch for {task} / {head}: "
                f"candidate={candidate['samples']} baseline={reference['samples']}"
            )
        differences = {
            metric: float(candidate[metric]) - float(reference[metric]) for metric in METRICS
        }
        rows.append((task, head, candidate, reference, differences))

    lines = [
        f"# {args.title}",
        "",
        f"候选模型为 **{args.candidate_name}**，基线为 **{args.baseline_name}**。两侧均使用相同的 9-source（无 FabWave）数据、",
        "随机种子 42、下游划分、head 配置和 200-epoch 预算；分类任务使用 Mean+Max pooling。候选模型从 epoch 100 checkpoint",
        "恢复并训练至 epoch 200，最终 checkpoint 按 epoch 1–200 的最高 validation accuracy 选择。",
        "",
        f"基线来源：`{args.baseline_report.resolve()}`。差值定义为 `候选 - 基线`，单位为百分点。",
        "",
        "| Task | Head | Candidate best | Baseline best | Candidate Acc. | Baseline Acc. | ΔAcc. | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, head, candidate, reference, differences in rows:
        lines.append(
            f"| {task} | {head} | {candidate['epoch']} | {reference['epoch']} | "
            f"{pct(float(candidate['accuracy']))} | {pct(float(reference['accuracy']))} | "
            f"{delta(differences['accuracy'])} | {delta(differences['macro_f1'])} | "
            f"{delta(differences['weighted_f1'])} | {delta(differences['macro_iou'])} |"
        )

    lines.extend(["", "## 汇总", ""])
    for scope in ("Overall", "MLP", "DiffLoss"):
        scoped = rows if scope == "Overall" else [row for row in rows if row[1] == scope]
        if not scoped:
            continue
        wins = sum(row[4]["accuracy"] > 0 for row in scoped)
        ties = sum(row[4]["accuracy"] == 0 for row in scoped)
        losses = len(scoped) - wins - ties
        means = {
            metric: sum(row[4][metric] for row in scoped) / len(scoped) for metric in METRICS
        }
        lines.append(
            f"- {scope}: Accuracy 胜/平/负 = {wins}/{ties}/{losses}；平均 "
            + "，".join(f"Δ{DISPLAY[metric]} {delta(means[metric])}" for metric in METRICS)
        )

    lines.extend(["", "## Candidate run artifacts", ""])
    for task, head, candidate, _, _ in rows:
        lines.append(f"- {task} / {head}: `{candidate['run']}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
