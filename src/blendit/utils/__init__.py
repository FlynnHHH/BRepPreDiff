from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_run_dir(output_dir: str | Path, stage: str, name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_{name}" if name else ""
    run_dir = Path(output_dir) / stage / f"{timestamp}{suffix}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def make_logger(log_file: str | Path) -> logging.Logger:
    logger = logging.getLogger("blendit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

__all__ = ["create_run_dir", "make_logger", "seed_everything"]
