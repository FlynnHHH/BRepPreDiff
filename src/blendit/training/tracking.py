from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class WandbTracker:
    """Rank-zero W&B tracking for epoch-level training metrics."""

    run: Any | None = None

    @classmethod
    def initialize(
        cls,
        config: dict[str, Any],
        *,
        stage: str,
        run_dir: str | Path,
        logger: Any,
        is_main_process: bool,
    ) -> "WandbTracker":
        wandb_config = config.get("wandb", {})
        if not is_main_process or not bool(wandb_config.get("enabled", True)):
            if is_main_process:
                logger.info("W&B tracking disabled")
            return cls()

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking is enabled, but the 'wandb' package is not installed. "
                "Install the project dependencies or set wandb.enabled=false."
            ) from exc

        run_path = Path(run_dir)
        init_kwargs: dict[str, Any] = {
            "project": str(wandb_config.get("project", "blendit")),
            "name": str(wandb_config.get("name") or run_path.name),
            "job_type": stage,
            # W&B creates its own `wandb/` directory below this run directory.
            "dir": str(run_path),
            "config": config,
        }
        for key in ("entity", "group", "notes", "mode"):
            value = wandb_config.get(key)
            if value is not None:
                init_kwargs[key] = value
        tags = wandb_config.get("tags")
        if tags:
            init_kwargs["tags"] = list(tags)

        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init() did not return a run while W&B tracking is enabled.")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        logger.info(
            "W&B tracking initialized project=%s run=%s url=%s",
            init_kwargs["project"],
            init_kwargs["name"],
            getattr(run, "url", None) or "unavailable",
        )
        return cls(run=run)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
    ) -> None:
        if self.run is None:
            return
        payload: dict[str, float | int] = {"epoch": epoch}
        self._add_metrics(payload, "train", train_metrics)
        if val_metrics is not None:
            self._add_metrics(payload, "val", val_metrics)
        self.run.log(payload)

    @staticmethod
    def _add_metrics(
        payload: dict[str, float | int],
        split: str,
        metrics: dict[str, float],
    ) -> None:
        for key, value in metrics.items():
            payload[f"{split}/{key}"] = float(value)
        if "total" in metrics:
            payload[f"{split}/loss"] = float(metrics["total"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
