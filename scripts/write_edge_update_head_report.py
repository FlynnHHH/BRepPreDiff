#!/usr/bin/env python3
"""Write the completed Edge Update Attention MLP/DiffLoss benchmark report."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "edge_update_new_joint" / "finetune"
REPORT_PATH = ROOT / "reports" / "edge_update_mlp_diffloss_all_benchmarks_2026-08-08.md"


@dataclass(frozen=True)
class Experiment:
    benchmark: str
    task: str
    head: str
    run_dir: str
    epochs: int


EXPERIMENTS = (
    Experiment("BRepPreDiff", "3-class face segmentation", "MLP", "20260807-200941_edge_update_brepprediff_seg_20260807-123259", 100),
    Experiment("BRepPreDiff", "3-class face segmentation", "DiffLoss", "*_edge_update_brepprediff_seg_diffloss_20260808-head-complements-titan", 100),
    Experiment("Fusion360Seg s2.0.0", "8-class face segmentation", "MLP", "20260807-200941_edge_update_fusion360seg_20260807-123259", 100),
    Experiment("Fusion360Seg s2.0.0", "8-class face segmentation", "DiffLoss", "*_edge_update_fusion360seg_diffloss_20260808-head-complements-titan", 100),
    Experiment("MFCAD++", "25-class face segmentation", "MLP", "20260807-200941_edge_update_mfcadpp_seg_20260807-123259", 100),
    Experiment("MFCAD++", "25-class face segmentation", "DiffLoss", "*_edge_update_mfcadpp_seg_diffloss_20260808-head-complements-titan", 100),
    Experiment("TMCAD", "10-class graph classification", "MLP", "*_edge_update_tmcad_cls_mlp_20260808-head-complements-titan", 200),
    Experiment("TMCAD", "10-class graph classification", "DiffLoss", "20260807-200941_edge_update_tmcad_cls_20260807-123259", 200),
    Experiment("FabWave min10", "40-class graph classification", "MLP", "*_edge_update_fabwave_cls_mlp_20260808-head-complements-titan", 200),
    Experiment("FabWave min10", "40-class graph classification", "DiffLoss", "20260807-211612_edge_update_fabwave_cls_20260807-123259", 200),
)


def resolve_run(pattern: str) -> Path:
    if "*" not in pattern:
        path = RUN_ROOT / pattern
        if path.is_dir():
            return path
        raise FileNotFoundError(path)
    matches = sorted(RUN_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run matches {RUN_ROOT / pattern}")
    return matches[-1]


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.4f}"


def main() -> None:
    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        run_dir = resolve_run(experiment.run_dir)
        metrics_path = run_dir / "test_metrics.json"
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Incomplete run: {run_dir}")
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        rows.append(
            {
                "benchmark": experiment.benchmark,
                "task": experiment.task,
                "head": experiment.head,
                "epochs": experiment.epochs,
                "run_dir": run_dir,
                "checkpoint_epoch": int(result["checkpoint_epoch"]),
                "samples": int(result["samples"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "macro_iou": float(metrics["macro_iou"]),
                "sha256": checkpoint_sha256(checkpoint_path),
            }
        )

    by_benchmark: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_benchmark.setdefault(str(row["benchmark"]), {})[str(row["head"])] = row

    lines = [
        "# Edge Update Attention：全部 Benchmark 的 MLP 与 DiffLoss 对照",
        "",
        f"> 生成日期：{date.today().isoformat()}  ",
        "> Encoder：Edge Update Attention（4 heads）  ",
        "> 预训练：七来源、全部 split 无标签联合预训练  ",
        "> 随机种子：42；按 validation accuracy 选择 `best.pt`；下表均为正式 test split",
        "",
        "## 1. 实验协议",
        "",
        "每个 benchmark 使用同一份预训练 checkpoint、相同数据划分、优化器设置和有效 batch size。分割任务的两种 head 均训练 100 epochs；分类任务的两种 head 均训练 200 epochs。DiffLoss 使用 bipolar one-hot、cosine 1000-step 训练噪声日程、每 token 4 个噪声样本以及 1-step DDIM 推理。",
        "",
        "该预训练语料包含下游 validation/test 的无标签几何，因此属于 transductive self-supervised 协议，不能视作严格 inductive benchmark。所有配置只有单个随机种子。",
        "",
        "## 2. 完整测试结果",
        "",
        "| Benchmark | Task | Head | Best epoch | Samples | Accuracy (%) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['task']} | {row['head']} | {row['checkpoint_epoch']} | "
            f"{row['samples']:,} | {percent(row['accuracy'])} | {percent(row['macro_precision'])} | "
            f"{percent(row['macro_recall'])} | {percent(row['macro_f1'])} | "
            f"{percent(row['weighted_f1'])} | {percent(row['macro_iou'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. Head 差异",
            "",
            "差值定义为 `DiffLoss - MLP`，单位为百分点；正值表示 DiffLoss 更高。",
            "",
            "| Benchmark | ΔAccuracy | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU | Accuracy winner |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    winners: list[str] = []
    for benchmark, heads in by_benchmark.items():
        mlp = heads["MLP"]
        diff = heads["DiffLoss"]
        delta_acc = float(diff["accuracy"]) - float(mlp["accuracy"])
        if delta_acc > 0:
            winner = "DiffLoss"
        elif delta_acc < 0:
            winner = "MLP"
        else:
            winner = "Tie"
        winners.append(f"- **{benchmark}**：{winner}，Accuracy 差值 {signed_pp(delta_acc)} pp。")
        lines.append(
            f"| {benchmark} | {signed_pp(delta_acc)} | "
            f"{signed_pp(float(diff['macro_f1']) - float(mlp['macro_f1']))} | "
            f"{signed_pp(float(diff['weighted_f1']) - float(mlp['weighted_f1']))} | "
            f"{signed_pp(float(diff['macro_iou']) - float(mlp['macro_iou']))} | {winner} |"
        )

    lines.extend(["", "## 4. 结论", "", *winners, "", "低于 1 pp 的单 seed 差异不应解释为稳定优势，建议对重点组合补做至少 3 个随机种子。", "", "## 5. 运行产物", ""])
    for row in rows:
        relative_run = Path(row["run_dir"]).relative_to(ROOT)
        lines.append(
            f"- {row['benchmark']} / {row['head']}：`{relative_run}`；"
            f"`best.pt` SHA-256 `{row['sha256']}`"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
