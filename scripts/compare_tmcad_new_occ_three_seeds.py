#!/usr/bin/env python3
"""Summarize TMCAD MLP vs DiffLoss across seeds 42, 43, and 44."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("accuracy", "macro_f1", "weighted_f1", "macro_iou")


def load_run(path: Path) -> dict[str, object]:
    result = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    return {
        "run": path.resolve(),
        "epoch": int(result["checkpoint_epoch"]),
        "samples": int(result["samples"]),
        **{name: float(metrics[name]) for name in METRICS},
    }


def pct(value: float) -> str:
    return f"{value * 100.0:.4f}"


def signed_pp(value: float) -> str:
    return f"{value * 100.0:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("SEED", "MLP_RUN", "DIFFLOSS_RUN"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[tuple[int, dict[str, object], dict[str, object]]] = []
    for seed_text, mlp_path, diff_path in args.pair:
        rows.append((int(seed_text), load_run(Path(mlp_path)), load_run(Path(diff_path))))
    rows.sort(key=lambda row: row[0])

    lines = [
        "# TMCAD 新 OCC 特征：MLP 与 DiffLoss 三随机种子对照",
        "",
        "Seed 42/43/44 使用同一预训练 encoder、数据划分、Mean+Max pooling、200 epochs、",
        "batch size 64、梯度累积 4，并按 validation accuracy 选择 best checkpoint。",
        "DiffLoss 使用 `L_x_start + 0.5 * L_epsilon` 和单步 DDIM。",
        "",
        "| Seed | Head | Best epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for seed, mlp, diff in rows:
        for head, result in (("MLP", mlp), ("DiffLoss", diff)):
            lines.append(
                f"| {seed} | {head} | {result['epoch']} | {pct(float(result['accuracy']))} | "
                f"{pct(float(result['macro_f1']))} | {pct(float(result['weighted_f1']))} | "
                f"{pct(float(result['macro_iou']))} |"
            )

    lines.extend(
        [
            "",
            "差值定义为 `DiffLoss - MLP`，单位为百分点。",
            "",
            "| Seed | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed, mlp, diff in rows:
        lines.append(
            f"| {seed} | "
            + " | ".join(signed_pp(float(diff[name]) - float(mlp[name])) for name in METRICS)
            + " |"
        )

    lines.extend(
        [
            "",
            "下表为三个 seed 的均值 ± 样本标准差。",
            "",
            "| Head | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for head_index, head in ((1, "MLP"), (2, "DiffLoss")):
        cells = []
        for metric in METRICS:
            values = [float(row[head_index][metric]) * 100.0 for row in rows]
            cells.append(f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}")
        lines.append(f"| {head} | " + " | ".join(cells) + " |")

    delta_means = {
        metric: statistics.mean(float(diff[metric]) - float(mlp[metric]) for _, mlp, diff in rows)
        for metric in METRICS
    }
    winner = "DiffLoss" if delta_means["accuracy"] > 0 else "MLP" if delta_means["accuracy"] < 0 else "Tie"
    lines.extend(
        [
            "",
            f"三 seed 平均 Accuracy winner：**{winner}**；平均 `DiffLoss - MLP` 为 "
            f"{signed_pp(delta_means['accuracy'])} pp。",
            "",
            "## Run artifacts",
            "",
        ]
    )
    for seed, mlp, diff in rows:
        lines.append(f"- Seed {seed} / MLP: `{mlp['run']}`")
        lines.append(f"- Seed {seed} / DiffLoss: `{diff['run']}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

