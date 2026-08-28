#!/usr/bin/env python3
"""Compare the five joint x_start/epsilon DiffLoss runs to existing heads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "edge_update_new_joint" / "finetune"
REPORT_PATH = ROOT / "reports" / "edge_update_diffloss_xstart_epsilon_all_benchmarks_2026-08-08.md"


@dataclass(frozen=True)
class Experiment:
    benchmark: str
    variant: str
    run_pattern: str


EXPERIMENTS = (
    Experiment("BRepPreDiff", "MLP", "20260807-200941_edge_update_brepprediff_seg_20260807-123259"),
    Experiment("BRepPreDiff", "DiffLoss x_start", "*_edge_update_extend200_brepprediff_xstart_20260810-seg200-titan"),
    Experiment("BRepPreDiff", "DiffLoss x_start+0.5eps", "*_edge_update_brepprediff_seg_diffloss_xse_20260808-xse-titan"),
    Experiment("Fusion360Seg s2.0.0", "MLP", "*_edge_update_extend200_fusion360seg_mlp_20260810-seg200-titan"),
    Experiment("Fusion360Seg s2.0.0", "DiffLoss x_start", "*_edge_update_extend200_fusion360seg_xstart_20260810-seg200-titan"),
    Experiment("Fusion360Seg s2.0.0", "DiffLoss x_start+0.5eps", "*_edge_update_extend200_fusion360seg_xse_20260810-seg200-titan"),
    Experiment("MFCAD++", "MLP", "*_edge_update_extend200_mfcadpp_mlp_20260810-seg200-titan"),
    Experiment("MFCAD++", "DiffLoss x_start", "*_edge_update_extend200_mfcadpp_xstart_20260810-seg200-titan"),
    Experiment("MFCAD++", "DiffLoss x_start+0.5eps", "*_edge_update_extend200_mfcadpp_xse_20260810-seg200-titan"),
    Experiment("TMCAD", "MLP", "20260808-163205_edge_update_tmcad_cls_mlp_20260808-head-complements-titan"),
    Experiment("TMCAD", "DiffLoss x_start", "20260807-200941_edge_update_tmcad_cls_20260807-123259"),
    Experiment("TMCAD", "DiffLoss x_start+0.5eps", "*_edge_update_tmcad_cls_diffloss_xse_20260808-xse-titan"),
    Experiment("FabWave min10", "MLP", "20260808-174658_edge_update_fabwave_cls_mlp_20260808-head-complements-titan"),
    Experiment("FabWave min10", "DiffLoss x_start", "20260807-211612_edge_update_fabwave_cls_20260807-123259"),
    Experiment("FabWave min10", "DiffLoss x_start+0.5eps", "*_edge_update_fabwave_cls_diffloss_xse_20260808-xse-titan"),
)


def resolve(pattern: str) -> Path:
    if "*" not in pattern:
        path = RUN_ROOT / pattern
        if path.is_dir():
            return path
        raise FileNotFoundError(path)
    matches = sorted(RUN_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(RUN_ROOT / pattern)
    return matches[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(value: float) -> str:
    return f"{100.0 * value:.4f}"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.4f}"


def main() -> None:
    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        run_dir = resolve(experiment.run_pattern)
        result = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        metrics = result["metrics"]
        rows.append({
            "benchmark": experiment.benchmark,
            "variant": experiment.variant,
            "run_dir": run_dir,
            "epoch": int(result["checkpoint_epoch"]),
            "samples": int(result["samples"]),
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "macro_iou": float(metrics["macro_iou"]),
            "sha256": sha256(run_dir / "checkpoints" / "best.pt"),
        })

    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["benchmark"]), {})[str(row["variant"])] = row

    lines = [
        "# Edge Update Attention：联合 x_start/epsilon DiffLoss 五项复验",
        "",
        f"> 生成日期：{date.today().isoformat()}  ",
        "> 目标函数：`MSE(x_start) + 0.5 * MSE(epsilon)`（未除以权重和）  ",
        "> 标签：bipolar one-hot；cosine 1000-step；每 token 4 个训练噪声样本；1-step DDIM 测试  ",
        "> Encoder：Edge Update Attention；相同联合预训练 checkpoint；seed 42；按 validation accuracy 选择 `best.pt`",
        "> 训练预算：全部任务 200 epochs；分割任务由 epochs 1-100 与续训 epochs 101-200 的候选共同选择全局 best",
        "",
        "## 完整测试结果",
        "",
        "| Benchmark | Variant | Best epoch | Samples | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark']} | {row['variant']} | {row['epoch']} | {row['samples']:,} | "
            f"{pct(row['accuracy'])} | {pct(row['macro_f1'])} | "
            f"{pct(row['weighted_f1'])} | {pct(row['macro_iou'])} |"
        )

    lines.extend([
        "",
        "## 新 DiffLoss 的差异",
        "",
        "单位为百分点；正值表示联合 x_start/epsilon DiffLoss 更高。",
        "",
        "| Benchmark | ΔAcc vs x_start | ΔMacro-F1 vs x_start | ΔmIoU vs x_start | ΔAcc vs MLP |",
        "|---|---:|---:|---:|---:|",
    ])
    for benchmark, variants in grouped.items():
        new = variants["DiffLoss x_start+0.5eps"]
        old = variants["DiffLoss x_start"]
        mlp = variants["MLP"]
        lines.append(
            f"| {benchmark} | {signed_pp(float(new['accuracy']) - float(old['accuracy']))} | "
            f"{signed_pp(float(new['macro_f1']) - float(old['macro_f1']))} | "
            f"{signed_pp(float(new['macro_iou']) - float(old['macro_iou']))} | "
            f"{signed_pp(float(new['accuracy']) - float(mlp['accuracy']))} |"
        )

    lines.extend(["", "## 运行产物", ""])
    for row in rows:
        relative = Path(row["run_dir"]).relative_to(ROOT)
        lines.append(
            f"- {row['benchmark']} / {row['variant']}：`{relative}`；"
            f"`best.pt` SHA-256 `{row['sha256']}`"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
