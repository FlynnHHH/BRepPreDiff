#!/usr/bin/env python3
"""Fit continuous-feature moments using only a configured training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brepprediff.config import load_experiment_config
from brepprediff.data.dataset import build_dataloader


class RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total: torch.Tensor | None = None
        self.total_square: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.to(dtype=torch.float64)
        batch_total = values.sum(dim=0)
        batch_total_square = values.square().sum(dim=0)
        if self.total is None:
            self.total = batch_total
            self.total_square = batch_total_square
        else:
            self.total += batch_total
            assert self.total_square is not None
            self.total_square += batch_total_square
        self.count += int(values.shape[0])

    def finish(self) -> tuple[list[float], list[float]]:
        if self.count == 0 or self.total is None or self.total_square is None:
            raise ValueError("Cannot compute statistics from zero feature rows.")
        mean = self.total / self.count
        variance = (self.total_square / self.count - mean.square()).clamp_min(0.0)
        std = variance.sqrt().clamp_min(1.0e-6)
        return mean.tolist(), std.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training YAML defining the dataset.")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.data_config)
    config.setdefault("train", {})["normalize_per_graph"] = False
    config["train"]["feature_preprocessing"] = {"mode": "none"}
    dataloader = build_dataloader(config, split="train", shuffle=False)
    dataset = dataloader.dataset

    face_moments = RunningMoments()
    edge_moments = RunningMoments()
    for batch_index, batch in enumerate(dataloader, start=1):
        face_moments.update(batch.face_cont)
        edge_moments.update(batch.edge_cont)
        if batch_index % 25 == 0:
            print(
                f"processed batches={batch_index}/{len(dataloader)} "
                f"faces={face_moments.count} edges={edge_moments.count}",
                flush=True,
            )

    face_mean, face_std = face_moments.finish()
    edge_mean, edge_std = edge_moments.finish()
    payload = {
        "version": 1,
        "source_config": str(Path(args.config)),
        "split": "train",
        "uv_grid_size": int(config["brep"]["uv_grid_size"]),
        "graph_count": len(dataset),
        "face_count": face_moments.count,
        "edge_count": edge_moments.count,
        "face_mean": face_mean,
        "face_std": face_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}: graphs={len(dataset)} faces={face_moments.count} "
        f"edges={edge_moments.count}"
    )


if __name__ == "__main__":
    main()
