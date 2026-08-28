#!/usr/bin/env python3
"""Write the TMCAD/FabWave graph-pooling ablation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "cls_pooling_ablation"
DEFAULT_RUN_TAG = "20260810-pooling-v3"
DEFAULT_REPORT = ROOT / "reports" / "cls_pooling_ablation_2026-08-11.md"
PRETRAIN_CHECKPOINT = (
    ROOT
    / "runs"
    / "pretrain"
    / "20260805-113103_joint_fusion_gallery_mlp_all_unlabeled_20260805-unlabeled-v2"
    / "checkpoints"
    / "last.pt"
)


@dataclass(frozen=True)
class Experiment:
    dataset: str
    dataset_label: str
    method: str
    method_label: str
    gpu: int


EXPERIMENTS = tuple(
    Experiment(dataset, dataset_label, method, method_label, gpu)
    for dataset, dataset_label, gpu_by_method in (
        ("tmcad", "TMCAD（10 类）", {"mean": 1, "mean_max": 2, "mean_std": 3, "residual_attention": 2}),
        ("fabwave", "FabWave min10（40 类）", {"mean": 1, "mean_max": 1, "mean_std": 3, "residual_attention": 3}),
    )
    for method, method_label in (
        ("mean", "Mean（基线）"),
        ("mean_max", "Mean + Max"),
        ("mean_std", "Mean + Std"),
        ("residual_attention", "Residual Attention"),
    )
    for gpu in (gpu_by_method[method],)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run(run_root: Path, experiment: Experiment, run_tag: str) -> Path:
    pattern = f"*_pooling_{experiment.dataset}_{experiment.method}_{run_tag}"
    matches = sorted((run_root / "finetune").glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run matches {run_root / 'finetune' / pattern}")
    return matches[-1]


def parameter_count(run_root: Path, experiment: Experiment, run_tag: str) -> int:
    log_path = run_root / "launch_logs" / run_tag / f"{experiment.dataset}_{experiment.method}.log"
    matches = re.findall(r"model parameters=(\d+)", log_path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(f"Model parameter count not found in {log_path}")
    return int(matches[-1])


def best_validation_accuracy(
    run_root: Path,
    experiment: Experiment,
    run_tag: str,
    checkpoint_epoch: int,
) -> float:
    log_path = run_root / "launch_logs" / run_tag / f"{experiment.dataset}_{experiment.method}.log"
    pattern = rf"epoch={checkpoint_epoch} split=val acc=([0-9.]+)"
    matches = re.findall(pattern, log_path.read_text(encoding="utf-8"))
    if not matches:
        raise ValueError(
            f"Validation accuracy for checkpoint epoch {checkpoint_epoch} not found in {log_path}"
        )
    return float(matches[-1])


def percent(value: float) -> str:
    return f"{100.0 * value:.4f}"


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.4f}"


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        run_dir = resolve_run(run_root, experiment, args.run_tag)
        metrics_path = run_dir / "test_metrics.json"
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        if not metrics_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Incomplete run: {run_dir}")
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = result["metrics"]
        checkpoint_epoch = int(result["checkpoint_epoch"])
        rows.append(
            {
                "dataset": experiment.dataset,
                "dataset_label": experiment.dataset_label,
                "method": experiment.method,
                "method_label": experiment.method_label,
                "gpu": experiment.gpu,
                "parameters": parameter_count(run_root, experiment, args.run_tag),
                "run_dir": run_dir,
                "checkpoint_epoch": checkpoint_epoch,
                "validation_accuracy": best_validation_accuracy(
                    run_root,
                    experiment,
                    args.run_tag,
                    checkpoint_epoch,
                ),
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

    baselines = {
        str(row["dataset"]): row for row in rows if row["method"] == "mean"
    }
    winners = {
        dataset: max(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: float(row["accuracy"]),
        )
        for dataset in baselines
    }
    pretrain_sha = checkpoint_sha256(PRETRAIN_CHECKPOINT)

    lines = [
        "# CLS 面特征聚合改进：TMCAD 与 FabWave 实验报告",
        "",
        f"> 生成日期：{date.today().isoformat()}  ",
        "> 硬件：3 × NVIDIA TITAN RTX（物理 GPU 1、2、3）  ",
        "> 随机种子：42；每组 200 epochs；按 validation accuracy 选择 `best.pt`  ",
        "> 下表均为训练期间未参与优化与选模的正式 test split",
        "",
        "## 1. 改进动机与实现",
        "",
        "当前 CLS 路径把一个 CAD 图内的所有 face embedding 做简单均值，能够稳定表示整体，但会丢失极值、离散程度和少量关键面的信息。本次只改图级 pooling，保持 Encoder、预训练权重、MLP 分类头、数据划分和训练超参数一致。",
        "",
        "- **Mean + Max**：拼接均值与逐通道最大值，用小型残差 MLP 融合；面向少量判别性面被均值稀释的问题。",
        "- **Mean + Std**：拼接均值与逐通道标准差，用残差 MLP 融合；显式保留一个模型内部面表示的异质性。",
        "- **Residual Attention**：学习 face 权重，但不直接替换 mean；输出为 mean 与 attention readout 的门控残差组合，score 零初始化、初始输出严格等价于 mean，以缓解旧版纯 attention pooling 的权重塌缩。",
        "",
        "Mean + Max 和 Mean + Std 的末层也使用零初始化，因此三种改进都从相同的 mean 表示出发。所有方法输出仍为 128 维，可同时用于 MLP 和 DiffLoss 分类头；本次用 MLP 头隔离 pooling 变量。",
        "",
        "## 2. 实验协议",
        "",
        "- Encoder：4-layer B-Rep FFN baseline，hidden dim 128。",
        "- 初始化：同一联合无标签预训练 checkpoint；SHA-256 " + f"`{pretrain_sha}`。",
        "- 优化：AdamW，lr `3e-4`，weight decay `1e-4`，batch size 256，full fine-tuning。",
        "- 数据：TMCAD train/val/test = 8,709/1,090/1,087；FabWave min10 = 3,191/407/391。",
        "- 归一化：per-graph feature normalization；模型 seed 42；独立 DataLoader seed 42，保证各方法每个 epoch 的样本顺序一致。",
        "",
        "联合无标签预训练包含下游 split 的几何，因此属于 transductive self-supervised 协议；结果适合本仓库内公平消融，不应直接当作严格 inductive benchmark。",
        "",
        "## 3. 完整测试结果",
        "",
        "| Dataset | Pooling | GPU | Params | Best epoch | Best val Acc (%) | Samples | Test Acc (%) | ΔTest Acc vs Mean (pp) | Macro-P (%) | Macro-R (%) | Macro-F1 (%) | Weighted-F1 (%) | mIoU (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        baseline = baselines[str(row["dataset"])]
        delta = float(row["accuracy"]) - float(baseline["accuracy"])
        lines.append(
            f"| {row['dataset_label']} | {row['method_label']} | {row['gpu']} | "
            f"{int(row['parameters']):,} | {row['checkpoint_epoch']} | "
            f"{percent(float(row['validation_accuracy']))} | {row['samples']:,} | "
            f"{percent(float(row['accuracy']))} | {signed_pp(delta)} | "
            f"{percent(float(row['macro_precision']))} | {percent(float(row['macro_recall']))} | "
            f"{percent(float(row['macro_f1']))} | {percent(float(row['weighted_f1']))} | "
            f"{percent(float(row['macro_iou']))} |"
        )

    lines.extend(["", "## 4. 结论", ""])
    for dataset, winner in winners.items():
        baseline = baselines[dataset]
        best_accuracy = float(winner["accuracy"])
        tied = [
            row
            for row in rows
            if row["dataset"] == dataset
            and abs(float(row["accuracy"]) - best_accuracy) < 1.0e-12
        ]
        if len(tied) > 1:
            tied_names = "、".join(str(row["method_label"]) for row in tied)
            lines.append(
                f"- **{winner['dataset_label']}**：{tied_names} 的全部 test 指标完全并列，"
                f"Accuracy 均为 {percent(best_accuracy)}%。新增 pooling 没有产生可见收益，"
                "因此应保留参数更少的 Mean。"
            )
            continue
        delta = best_accuracy - float(baseline["accuracy"])
        f1_delta = float(winner["macro_f1"]) - float(baseline["macro_f1"])
        iou_delta = float(winner["macro_iou"]) - float(baseline["macro_iou"])
        parameter_overhead = (
            int(winner["parameters"]) - int(baseline["parameters"])
        ) / int(baseline["parameters"])
        lines.append(
            f"- **{winner['dataset_label']}**：{winner['method_label']} 最优，Test Accuracy "
            f"{percent(best_accuracy)}%，相对 Mean {signed_pp(delta)} pp；Macro-F1 "
            f"{signed_pp(f1_delta)} pp，mIoU {signed_pp(iou_delta)} pp。参数量增加 "
            f"{100.0 * parameter_overhead:.2f}%。"
        )
    lines.extend(
        [
            "",
            "这是单随机种子消融。小于 1 pp 的差异应视为候选信号；若要升级默认配置，建议对领先方法补做至少 3 个 seed，并报告均值和标准差。",
            "",
            "## 5. 运行产物",
            "",
        ]
    )
    for row in rows:
        relative_run = Path(row["run_dir"]).relative_to(ROOT)
        lines.append(
            f"- {row['dataset_label']} / {row['method_label']}：`{relative_run}`；"
            f"`best.pt` SHA-256 `{row['sha256']}`"
        )
    lines.extend(
        [
            "",
            f"复现入口：`RUN_TAG={args.run_tag} scripts/run_cls_pooling_ablation_titan.sh`。",
        ]
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
