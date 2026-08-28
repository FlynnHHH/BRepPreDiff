#!/usr/bin/env python3
"""Report whether extending segmentation heads from 100 to 200 epochs helps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "edge_update_new_joint" / "finetune"
REPORT_PATH = ROOT / "reports" / "edge_update_segmentation_three_heads_extend200_2026-08-10.md"


@dataclass(frozen=True)
class Pair:
    benchmark: str
    head: str
    source: str
    extension_pattern: str


PAIRS = (
    Pair("BRepPreDiff", "MLP", "20260807-200941_edge_update_brepprediff_seg_20260807-123259", "*_edge_update_extend200_brepprediff_mlp_20260810-seg200-titan"),
    Pair("BRepPreDiff", "DiffLoss x_start", "20260808-163205_edge_update_brepprediff_seg_diffloss_20260808-head-complements-titan", "*_edge_update_extend200_brepprediff_xstart_20260810-seg200-titan"),
    Pair("BRepPreDiff", "DiffLoss x_start+0.5eps", "20260808-183835_edge_update_brepprediff_seg_diffloss_xse_20260808-xse-titan", "*_edge_update_extend200_brepprediff_xse_20260810-seg200-titan"),
    Pair("Fusion360Seg s2.0.0", "MLP", "20260807-200941_edge_update_fusion360seg_20260807-123259", "*_edge_update_extend200_fusion360seg_mlp_20260810-seg200-titan"),
    Pair("Fusion360Seg s2.0.0", "DiffLoss x_start", "20260808-163205_edge_update_fusion360seg_diffloss_20260808-head-complements-titan", "*_edge_update_extend200_fusion360seg_xstart_20260810-seg200-titan"),
    Pair("Fusion360Seg s2.0.0", "DiffLoss x_start+0.5eps", "20260808-183835_edge_update_fusion360seg_diffloss_xse_20260808-xse-titan", "*_edge_update_extend200_fusion360seg_xse_20260810-seg200-titan"),
    Pair("MFCAD++", "MLP", "20260807-200941_edge_update_mfcadpp_seg_20260807-123259", "*_edge_update_extend200_mfcadpp_mlp_20260810-seg200-titan"),
    Pair("MFCAD++", "DiffLoss x_start", "20260808-163205_edge_update_mfcadpp_seg_diffloss_20260808-head-complements-titan", "*_edge_update_extend200_mfcadpp_xstart_20260810-seg200-titan"),
    Pair("MFCAD++", "DiffLoss x_start+0.5eps", "20260808-183835_edge_update_mfcadpp_seg_diffloss_xse_20260808-xse-titan", "*_edge_update_extend200_mfcadpp_xse_20260810-seg200-titan"),
)


def resolve(pattern: str) -> Path:
    path = RUN_ROOT / pattern
    if "*" not in pattern:
        if path.is_dir():
            return path
        raise FileNotFoundError(path)
    matches = sorted(RUN_ROOT.glob(pattern), key=lambda item: item.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(path)
    return matches[-1]


def load_val(path: Path) -> tuple[int, float]:
    checkpoint = torch.load(path / "checkpoints" / "best.pt", map_location="cpu")
    return int(checkpoint["epoch"]), float(checkpoint["metrics"]["acc"])


def load_test(path: Path) -> dict[str, float]:
    result = json.loads((path / "test_metrics.json").read_text(encoding="utf-8"))
    metric_names = ("accuracy", "macro_f1", "weighted_f1", "macro_iou")
    return {name: float(result["metrics"][name]) for name in metric_names}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}"


def delta(value: float) -> str:
    return f"{100.0 * value:+.4f}"


def main() -> None:
    rows: list[dict[str, object]] = []
    for pair in PAIRS:
        early_dir = resolve(pair.source)
        late_dir = resolve(pair.extension_pattern)
        early_epoch, early_val = load_val(early_dir)
        late_epoch, late_val = load_val(late_dir)
        early_test = load_test(early_dir)
        late_test = load_test(late_dir)
        if late_val > early_val:
            chosen_period = "101-200"
            chosen_dir = late_dir
            chosen_epoch = late_epoch
            chosen_test = late_test
        else:
            chosen_period = "1-100"
            chosen_dir = early_dir
            chosen_epoch = early_epoch
            chosen_test = early_test
        rows.append({
            "benchmark": pair.benchmark,
            "head": pair.head,
            "early_epoch": early_epoch,
            "early_val": early_val,
            "late_epoch": late_epoch,
            "late_val": late_val,
            "early_test": early_test,
            "late_test": late_test,
            "chosen_period": chosen_period,
            "chosen_epoch": chosen_epoch,
            "chosen_dir": chosen_dir,
            "chosen_test": chosen_test,
            "late_dir": late_dir,
        })

    lines = [
        "# Edge Update Attention：三种分割 Head 延续至 200 Epochs",
        "",
        f"> 生成日期：{date.today().isoformat()}  ",
        "> 任务：BRepPreDiff、Fusion360Seg、MFCAD++  ",
        "> Head：MLP、DiffLoss x_start、DiffLoss x_start+0.5epsilon  ",
        "> 方法：从各自 epoch-100 `last.pt` 继承模型和 AdamW optimizer，继续训练 epochs 101-200；按 validation accuracy 在完整 epochs 1-200 中重新选择全局 best。",
        "",
        "## 全局 best 是否发生变化",
        "",
        "差值是延长到 200 epochs 后的全局 best test 指标减去原 100-epoch best；若后半程 validation accuracy 未超过原 best，则保留原 checkpoint，差值为零。",
        "",
        "| Benchmark | Head | 1-100 best | 101-200 best | ΔVal Acc (late-old, pp) | Global best period | Global epoch | ΔTest Acc | ΔMacro-F1 | ΔmIoU |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        old = row["early_test"]
        new = row["chosen_test"]
        lines.append(
            f"| {row['benchmark']} | {row['head']} | {row['early_epoch']} | {row['late_epoch']} | "
            f"{delta(float(row['late_val']) - float(row['early_val']))} | {row['chosen_period']} | "
            f"{row['chosen_epoch']} | {delta(new['accuracy'] - old['accuracy'])} | "
            f"{delta(new['macro_f1'] - old['macro_f1'])} | {delta(new['macro_iou'] - old['macro_iou'])} |"
        )

    lines.extend([
        "",
        "## Epochs 101-200 候选本身的测试结果",
        "",
        "| Benchmark | Head | Late epoch | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        metrics = row["late_test"]
        lines.append(
            f"| {row['benchmark']} | {row['head']} | {row['late_epoch']} | "
            f"{pct(metrics['accuracy'])} | {pct(metrics['macro_f1'])} | "
            f"{pct(metrics['weighted_f1'])} | {pct(metrics['macro_iou'])} |"
        )

    lines.extend(["", "## 续训运行与最终 checkpoint", ""])
    for row in rows:
        late_relative = Path(row["late_dir"]).relative_to(ROOT)
        chosen_relative = Path(row["chosen_dir"]).relative_to(ROOT)
        chosen_checkpoint = Path(row["chosen_dir"]) / "checkpoints" / "best.pt"
        lines.append(
            f"- {row['benchmark']} / {row['head']}：续训 `{late_relative}`；"
            f"全局选择 `{chosen_relative}/checkpoints/best.pt`；SHA-256 `{sha256(chosen_checkpoint)}`"
        )

    lines.extend([
        "",
        "说明：旧 checkpoint 未保存 RNG 状态，因此续训会恢复模型和 optimizer，但不会恢复 epoch 100 末尾的随机数流；这不是一次从 epoch 1 不间断运行到 200 的完全等价复现。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
