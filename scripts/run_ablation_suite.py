#!/usr/bin/env python3
"""Resume a full pretrain/finetune ablation suite and report validation/test metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    name: str
    stage: str
    config: str
    run_name: str
    epochs: int
    factors: dict[str, Any]
    overrides: tuple[str, ...] = ()
    pretrain_source: str | None = None
    initial_run: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run",
        required=True,
        help="Run directory for the already-running configs/pretrain.yaml baseline.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=20)
    parser.add_argument("--suite-dir", default="runs/ablation_suite")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SuiteRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.suite_dir = (ROOT / args.suite_dir).resolve()
        self.log_dir = self.suite_dir / "logs"
        self.state_path = self.suite_dir / "state.json"
        self.report_json_path = self.suite_dir / "results.json"
        self.report_csv_path = self.suite_dir / "results.csv"
        self.report_md_path = self.suite_dir / "RESULTS.md"
        self.evaluation_dir = self.suite_dir / "test_evaluations"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.completed_checkpoints: dict[str, Path] = {}
        self.python = Path(sys.executable).resolve()
        self.torchrun = self.python.with_name("torchrun")
        if not self.torchrun.exists():
            raise FileNotFoundError(f"torchrun not found next to Python interpreter: {self.torchrun}")

    def log(self, message: str) -> None:
        line = f"{now()} | {message}"
        print(line, flush=True)
        with (self.suite_dir / "suite.log").open("a", encoding="utf-8") as stream:
            print(line, file=stream)

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        return {"started_at": now(), "experiments": {}}

    def save_state(self) -> None:
        self.state["updated_at"] = now()
        temporary = self.state_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream, indent=2, ensure_ascii=False, sort_keys=True)
        temporary.replace(self.state_path)

    @staticmethod
    def load_checkpoint(path: Path) -> dict[str, Any] | None:
        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
        except (EOFError, OSError, RuntimeError, ValueError):
            return None
        return checkpoint if isinstance(checkpoint, dict) else None

    def run_dirs(self, experiment: Experiment) -> list[Path]:
        directories: list[Path] = []
        if experiment.initial_run:
            directories.append((ROOT / experiment.initial_run).resolve())
        stage_dir = ROOT / "runs" / experiment.stage
        directories.extend(sorted(stage_dir.glob(f"*_{experiment.run_name}")))
        unique: dict[str, Path] = {}
        for directory in directories:
            if directory.exists():
                unique[str(directory.resolve())] = directory.resolve()
        return list(unique.values())

    def checkpoint_records(self, experiment: Experiment, *, best_only: bool = False) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        pattern = "best.pt" if best_only else "*.pt"
        for run_dir in self.run_dirs(experiment):
            for path in (run_dir / "checkpoints").glob(pattern):
                if not best_only and not (path.name == "last.pt" or path.name.startswith("epoch_")):
                    continue
                checkpoint = self.load_checkpoint(path)
                if checkpoint is None:
                    continue
                records.append(
                    {
                        "path": path.resolve(),
                        "run_dir": run_dir.resolve(),
                        "epoch": int(checkpoint.get("epoch", 0)),
                        "metrics": checkpoint.get("metrics", {}) or {},
                    }
                )
        return records

    def latest_checkpoint(self, experiment: Experiment) -> dict[str, Any] | None:
        records = self.checkpoint_records(experiment)
        if not records:
            return None
        return max(records, key=lambda record: (record["epoch"], record["path"].stat().st_mtime))

    def completion_checkpoint(self, experiment: Experiment) -> dict[str, Any] | None:
        record = self.latest_checkpoint(experiment)
        if record is None or record["epoch"] < experiment.epochs:
            return None
        return record

    def best_checkpoint(self, experiment: Experiment) -> dict[str, Any] | None:
        records = self.checkpoint_records(experiment, best_only=True)
        records = [record for record in records if "acc" in record["metrics"]]
        if not records:
            return None
        return max(records, key=lambda record: float(record["metrics"]["acc"]))

    @staticmethod
    def active_process_lines(experiment: Experiment) -> list[str]:
        module = f"blendit.training.{experiment.stage}"
        result = subprocess.run(
            ["pgrep", "-af", module],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        lines = [line for line in result.stdout.splitlines() if module in line]
        config_token = f"--config {experiment.config}"
        lines = [line for line in lines if config_token in line]
        if experiment.initial_run:
            # The user's initial baseline has no suite run-name override.
            return lines
        run_token = f"run.name={experiment.run_name}"
        return [line for line in lines if run_token in line]

    def wait_for_active_process(self, experiment: Experiment) -> None:
        last_logged_epoch: int | None = None
        while True:
            active = self.active_process_lines(experiment)
            if not active:
                return
            latest = self.latest_checkpoint(experiment)
            epoch = int(latest["epoch"]) if latest else 0
            if epoch != last_logged_epoch:
                self.log(
                    f"{experiment.name}: active processes={len(active)} latest_epoch={epoch}/{experiment.epochs}"
                )
                last_logged_epoch = epoch
            time.sleep(max(1.0, self.args.poll_seconds))

    def command(self, experiment: Experiment, resume: Path | None) -> list[str]:
        gpu_count = len([item for item in self.args.gpus.split(",") if item.strip()])
        command = [
            str(self.torchrun),
            "--standalone",
            f"--nproc_per_node={gpu_count}",
            "-m",
            f"blendit.training.{experiment.stage}",
            "--config",
            experiment.config,
            "--override",
            f"run.name={experiment.run_name}",
        ]
        overrides = [
            *experiment.overrides,
            f"train.epochs={experiment.epochs}",
            f"train.batch_size={self.args.batch_size}",
            f"train.num_workers={self.args.num_workers}",
        ]
        if experiment.pretrain_source:
            pretrain_checkpoint = self.completed_checkpoints[experiment.pretrain_source]
            overrides.append(f"train.pretrain_checkpoint={pretrain_checkpoint}")
        if resume is not None:
            overrides.append(f"train.resume={resume}")
        for override in overrides:
            command.extend(["--override", override])
        return command

    def environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.args.gpus
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = (
            source_path
            if not environment.get("PYTHONPATH")
            else f"{source_path}{os.pathsep}{environment['PYTHONPATH']}"
        )
        return environment

    def launch(self, experiment: Experiment, resume: Path | None) -> int:
        command = self.command(experiment, resume)
        command_text = " ".join(str(item) for item in command)
        self.log(f"{experiment.name}: launch {command_text}")
        record = self.state["experiments"].setdefault(experiment.name, {"attempts": []})
        record["attempts"].append(
            {"started_at": now(), "resume": str(resume) if resume else None, "command": command}
        )
        record["status"] = "running"
        self.save_state()
        if self.args.dry_run:
            return 0

        launcher_log = self.log_dir / f"{experiment.name}.log"
        with launcher_log.open("a", encoding="utf-8") as stream:
            print(f"\n{now()} | {command_text}", file=stream, flush=True)
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=self.environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                start_new_session=True,
            )
        attempt = record["attempts"][-1]
        attempt["finished_at"] = now()
        attempt["returncode"] = process.returncode
        self.save_state()
        return int(process.returncode)

    def run_experiment(self, experiment: Experiment) -> Path:
        self.log(f"{experiment.name}: monitor start")
        attempts_started_here = 0
        while True:
            self.wait_for_active_process(experiment)
            completed = self.completion_checkpoint(experiment)
            if completed is not None:
                checkpoint_path = Path(completed["path"])
                record = self.state["experiments"].setdefault(experiment.name, {"attempts": []})
                record.update(
                    {
                        "status": "complete",
                        "completed_at": now(),
                        "epoch": int(completed["epoch"]),
                        "checkpoint": str(checkpoint_path),
                        "factors": experiment.factors,
                    }
                )
                best = self.best_checkpoint(experiment) if experiment.stage == "finetune" else None
                if best is not None:
                    record["best_checkpoint"] = str(best["path"])
                    record["best_epoch"] = int(best["epoch"])
                    record["best_metrics"] = best["metrics"]
                self.completed_checkpoints[experiment.name] = checkpoint_path
                self.save_state()
                self.write_reports()
                self.log(f"{experiment.name}: complete checkpoint={checkpoint_path}")
                return checkpoint_path

            if self.args.dry_run:
                resume = self.latest_checkpoint(experiment)
                self.launch(experiment, Path(resume["path"]) if resume else None)
                raise SystemExit("dry-run stops after the first pending launch")

            if attempts_started_here >= self.args.max_retries:
                record = self.state["experiments"].setdefault(experiment.name, {"attempts": []})
                record["status"] = "failed"
                record["failed_at"] = now()
                self.save_state()
                raise RuntimeError(
                    f"{experiment.name} did not complete after {self.args.max_retries} automatic attempts"
                )

            latest = self.latest_checkpoint(experiment)
            resume_path = Path(latest["path"]) if latest is not None else None
            returncode = self.launch(experiment, resume_path)
            attempts_started_here += 1
            self.log(f"{experiment.name}: launcher returncode={returncode}; checking completion/resume")

    def evaluate_experiment(self, experiment: Experiment) -> None:
        if experiment.stage != "finetune":
            return
        record = self.state["experiments"].get(experiment.name, {})
        evaluation_path = self.evaluation_dir / f"{experiment.name}.json"
        if record.get("test_evaluation") and evaluation_path.exists():
            return
        best_checkpoint = record.get("best_checkpoint")
        if not best_checkpoint:
            raise RuntimeError(f"{experiment.name} completed without a validation best checkpoint")

        command = [
            str(self.python),
            "-m",
            "blendit.training.evaluate",
            "--config",
            experiment.config,
            "--checkpoint",
            str(best_checkpoint),
            "--split",
            "test",
            "--output",
            str(evaluation_path),
            "--batch-size",
            str(self.args.batch_size),
            "--num-workers",
            str(self.args.num_workers),
        ]
        for override in experiment.overrides:
            command.extend(["--override", override])
        evaluation_log = self.log_dir / f"{experiment.name}.test.log"
        self.log(f"{experiment.name}: test evaluation start checkpoint={best_checkpoint}")
        with evaluation_log.open("a", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=self.environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0 or not evaluation_path.exists():
            raise RuntimeError(
                f"{experiment.name} test evaluation failed with returncode={process.returncode}; "
                f"see {evaluation_log}"
            )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        record["test_evaluation"] = evaluation
        record["test_evaluated_at"] = now()
        self.save_state()
        self.write_reports()
        self.log(
            f"{experiment.name}: test evaluation complete "
            f"macro_f1={evaluation['metrics']['macro_f1']:.6f}"
        )

    def write_reports(self) -> None:
        results: list[dict[str, Any]] = []
        csv_rows: list[dict[str, Any]] = []
        for name, record in self.state.get("experiments", {}).items():
            if record.get("status") != "complete" or "best_metrics" not in record:
                continue
            test_evaluation = record.get("test_evaluation")
            result = {
                "experiment": name,
                "factors": record.get("factors", {}),
                "best_epoch": record.get("best_epoch"),
                "best_checkpoint": record.get("best_checkpoint"),
                "validation_metrics": record["best_metrics"],
                "test_evaluation": test_evaluation,
            }
            results.append(result)

            row = {
                "experiment": name,
                **record.get("factors", {}),
                "best_epoch": record.get("best_epoch"),
                "best_checkpoint": record.get("best_checkpoint"),
                **{f"val_{key}": value for key, value in record["best_metrics"].items()},
            }
            if test_evaluation:
                test_metrics = test_evaluation["metrics"]
                row.update(
                    {
                        f"test_{key}": value
                        for key, value in test_metrics.items()
                        if not isinstance(value, (dict, list))
                    }
                )
                row.update(
                    {
                        f"test_transition_{key}": value
                        for key, value in test_metrics.get("transition_binary", {}).items()
                    }
                )
                for per_class in test_metrics.get("per_class", []):
                    class_id = per_class["class_id"]
                    for key, value in per_class.items():
                        if key not in {"class_id", "class_name"}:
                            row[f"test_class_{class_id}_{key}"] = value
            csv_rows.append(row)

        results.sort(key=lambda result: result["experiment"])
        csv_rows.sort(key=lambda row: row["experiment"])
        with self.report_json_path.open("w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, ensure_ascii=False, sort_keys=True)
        if not results:
            return
        fieldnames: list[str] = []
        for row in csv_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with self.report_csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        self.report_md_path.write_text(self.markdown_report(results), encoding="utf-8")

    def markdown_report(self, results: list[dict[str, Any]]) -> str:
        lines = [
            "# Finetune baseline and ablation results",
            "",
            f"Generated: {now()}",
            "",
            f"- Finetune train split: `data/splits/finetune_train.txt`",
            f"- Finetune validation split: `data/splits/finetune_val.txt`",
            f"- Test split: `data/splits/finetune_test.txt`",
            f"- Batch size per rank: `{self.args.batch_size}`",
            f"- DataLoader workers per rank: `{self.args.num_workers}`",
            "- Best checkpoint selection: validation accuracy",
            "",
            "## Summary",
            "",
            "| Experiment | Role | Ablated factor | Val Macro-F1 | Test accuracy | Test Macro-F1 | Test mIoU | Transition F1 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]

        def metric(value: Any) -> str:
            return "—" if value is None else f"{float(value):.6f}"

        for result in results:
            factors = result["factors"]
            test_metrics = (result.get("test_evaluation") or {}).get("metrics", {})
            transition = test_metrics.get("transition_binary", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(result["experiment"]),
                        str(factors.get("role", "")),
                        str(factors.get("ablated_factor", "")),
                        metric(result["validation_metrics"].get("f1")),
                        metric(test_metrics.get("accuracy")),
                        metric(test_metrics.get("macro_f1")),
                        metric(test_metrics.get("macro_iou")),
                        metric(transition.get("f1")),
                    ]
                )
                + " |"
            )

        for result in results:
            lines.extend(["", f"## {result['experiment']}", ""])
            lines.append(f"- Factors: `{json.dumps(result['factors'], ensure_ascii=False, sort_keys=True)}`")
            lines.append(f"- Best epoch: `{result['best_epoch']}`")
            lines.append(f"- Best checkpoint: `{result['best_checkpoint']}`")
            lines.extend(["", "### Validation metrics", "", "| Metric | Value |", "|---|---:|"])
            for key, value in sorted(result["validation_metrics"].items()):
                lines.append(f"| {key} | {metric(value)} |")

            evaluation = result.get("test_evaluation")
            if not evaluation:
                lines.extend(["", "Test evaluation pending."])
                continue
            test_metrics = evaluation["metrics"]
            lines.extend(["", "### Test metrics", "", "| Metric | Value |", "|---|---:|"])
            for key, value in test_metrics.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"| {key} | {metric(value)} |")
            for key, value in test_metrics.get("transition_binary", {}).items():
                lines.append(f"| transition_{key} | {metric(value)} |")

            lines.extend(
                [
                    "",
                    "### Per-class test metrics",
                    "",
                    "| Class | Support | Predicted | Precision | Recall | F1 | IoU |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in test_metrics.get("per_class", []):
                lines.append(
                    f"| {row['class_id']} {row['class_name']} | {row['support']} | {row['predicted']} | "
                    f"{metric(row['precision'])} | {metric(row['recall'])} | "
                    f"{metric(row['f1'])} | {metric(row['iou'])} |"
                )
            lines.extend(["", "### Test confusion matrix", "", "Rows are ground truth; columns are predictions.", ""])
            matrix = test_metrics.get("confusion_matrix", [])
            if matrix:
                labels = [str(row["class_name"]) for row in test_metrics.get("per_class", [])]
                lines.append("| GT \\ Pred | " + " | ".join(labels) + " |")
                lines.append("|---|" + "---:|" * len(labels))
                for label, row in zip(labels, matrix):
                    lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
        lines.append("")
        return "\n".join(lines)


def experiments(baseline_run: str) -> list[Experiment]:
    baseline_pretrain = Experiment(
        name="baseline_pretrain",
        stage="pretrain",
        config="configs/pretrain.yaml",
        run_name="suite_20260720_baseline_pretrain_resume",
        epochs=150,
        factors={"coarse_label": True},
        initial_run=baseline_run,
    )
    no_coarse_pretrain = Experiment(
        name="ablation_no_coarse_pretrain",
        stage="pretrain",
        config="configs/pretrain_no_coarse.yaml",
        run_name="suite_20260720_ablation_no_coarse_pretrain",
        epochs=150,
        factors={
            "role": "ablation",
            "ablated_factor": "coarse_label_pretraining",
            "coarse_label": False,
        },
    )

    finetunes = [
        Experiment(
            name="baseline_default",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_baseline_default",
            epochs=100,
            factors={
                "role": "baseline",
                "ablated_factor": "none",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "x_start_epsilon",
                "x_start_loss_weight": 1.0,
                "epsilon_loss_weight": 0.5,
                "encoder_freeze": "none",
            },
            overrides=(
                "label_diffusion.prediction_type=x_start_epsilon",
                "label_diffusion.x_start_loss_weight=1.0",
                "label_diffusion.epsilon_loss_weight=0.5",
            ),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_head_mlp_full",
            stage="finetune",
            config="configs/finetune.yaml",
            run_name="suite_20260720_ablation_head_mlp_full",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "finetune_head",
                "coarse_label": True,
                "head": "mlp",
                "prediction_type": "n/a",
                "encoder_freeze": "none",
            },
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_prediction_epsilon_full",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_prediction_epsilon_full",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "prediction_type",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "epsilon",
                "encoder_freeze": "none",
            },
            overrides=("label_diffusion.prediction_type=epsilon",),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_no_coarse_default",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_no_coarse_default",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "coarse_label_pretraining",
                "coarse_label": False,
                "head": "diffloss",
                "prediction_type": "x_start_epsilon",
                "x_start_loss_weight": 1.0,
                "epsilon_loss_weight": 0.5,
                "encoder_freeze": "none",
            },
            overrides=(
                "model.use_coarse_label_head=false",
                "label_diffusion.prediction_type=x_start_epsilon",
                "label_diffusion.x_start_loss_weight=1.0",
                "label_diffusion.epsilon_loss_weight=0.5",
            ),
            pretrain_source="ablation_no_coarse_pretrain",
        ),
        Experiment(
            name="ablation_no_coarse_mlp_full",
            stage="finetune",
            config="configs/finetune.yaml",
            run_name="suite_20260720_ablation_no_coarse_mlp_full",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "coarse_label_pretraining+finetune_head",
                "coarse_label": False,
                "head": "mlp",
                "prediction_type": "n/a",
                "encoder_freeze": "none",
            },
            overrides=("model.use_coarse_label_head=false",),
            pretrain_source="ablation_no_coarse_pretrain",
        ),
        Experiment(
            name="ablation_no_coarse_prediction_epsilon_full",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_no_coarse_prediction_epsilon_full",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "coarse_label_pretraining+prediction_type",
                "coarse_label": False,
                "head": "diffloss",
                "prediction_type": "epsilon",
                "encoder_freeze": "none",
            },
            overrides=(
                "model.use_coarse_label_head=false",
                "label_diffusion.prediction_type=epsilon",
            ),
            pretrain_source="ablation_no_coarse_pretrain",
        ),
        Experiment(
            name="ablation_prediction_x_start_full",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_prediction_x_start_full",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "prediction_type",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "x_start",
                "encoder_freeze": "none",
            },
            overrides=("label_diffusion.prediction_type=x_start",),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_epsilon_weight_1_0",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_epsilon_weight_1_0",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "epsilon_loss_weight",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "x_start_epsilon",
                "x_start_loss_weight": 1.0,
                "epsilon_loss_weight": 1.0,
                "encoder_freeze": "none",
            },
            overrides=(
                "label_diffusion.prediction_type=x_start_epsilon",
                "label_diffusion.x_start_loss_weight=1.0",
                "label_diffusion.epsilon_loss_weight=1.0",
            ),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_epsilon_weight_0_1",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_epsilon_weight_0_1",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "epsilon_loss_weight",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "x_start_epsilon",
                "x_start_loss_weight": 1.0,
                "epsilon_loss_weight": 0.1,
                "encoder_freeze": "none",
            },
            overrides=(
                "label_diffusion.prediction_type=x_start_epsilon",
                "label_diffusion.x_start_loss_weight=1.0",
                "label_diffusion.epsilon_loss_weight=0.1",
            ),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_epsilon_weight_0_25",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_epsilon_weight_0_25",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "epsilon_loss_weight",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "x_start_epsilon",
                "x_start_loss_weight": 1.0,
                "epsilon_loss_weight": 0.25,
                "encoder_freeze": "none",
            },
            overrides=(
                "label_diffusion.prediction_type=x_start_epsilon",
                "label_diffusion.x_start_loss_weight=1.0",
                "label_diffusion.epsilon_loss_weight=0.25",
            ),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_mlp_encoder_all",
            stage="finetune",
            config="configs/finetune.yaml",
            run_name="suite_20260720_ablation_mlp_encoder_all",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "finetune_head+encoder_freeze",
                "coarse_label": True,
                "head": "mlp",
                "prediction_type": "n/a",
                "encoder_freeze": "all",
            },
            overrides=("train.encoder_freeze_mode=all",),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_mlp_encoder_partial",
            stage="finetune",
            config="configs/finetune.yaml",
            run_name="suite_20260720_ablation_mlp_encoder_partial",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "finetune_head+encoder_freeze",
                "coarse_label": True,
                "head": "mlp",
                "prediction_type": "n/a",
                "encoder_freeze": "partial",
                "frozen_layers": 2,
            },
            overrides=("train.encoder_freeze_mode=partial", "train.encoder_frozen_layers=2"),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_epsilon_encoder_all",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_epsilon_encoder_all",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "prediction_type+encoder_freeze",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "epsilon",
                "encoder_freeze": "all",
            },
            overrides=("label_diffusion.prediction_type=epsilon", "train.encoder_freeze_mode=all"),
            pretrain_source="baseline_pretrain",
        ),
        Experiment(
            name="ablation_epsilon_encoder_partial",
            stage="finetune",
            config="configs/finetune_diffloss.yaml",
            run_name="suite_20260720_ablation_epsilon_encoder_partial",
            epochs=100,
            factors={
                "role": "ablation",
                "ablated_factor": "prediction_type+encoder_freeze",
                "coarse_label": True,
                "head": "diffloss",
                "prediction_type": "epsilon",
                "encoder_freeze": "partial",
                "frozen_layers": 2,
            },
            overrides=(
                "label_diffusion.prediction_type=epsilon",
                "train.encoder_freeze_mode=partial",
                "train.encoder_frozen_layers=2",
            ),
            pretrain_source="baseline_pretrain",
        ),
    ]
    return [baseline_pretrain, no_coarse_pretrain, *finetunes]


def main() -> None:
    args = parse_args()
    runner = SuiteRunner(args)
    planned = experiments(args.baseline_run)
    runner.state["plan"] = [asdict(experiment) for experiment in planned]
    runner.save_state()
    runner.log(f"suite start experiments={len(planned)} gpus={args.gpus}")
    for experiment in planned:
        runner.run_experiment(experiment)
        runner.evaluate_experiment(experiment)
    runner.state["status"] = "complete"
    runner.state["completed_at"] = now()
    runner.save_state()
    runner.write_reports()
    runner.log(f"suite complete report={runner.report_md_path}")


if __name__ == "__main__":
    main()
