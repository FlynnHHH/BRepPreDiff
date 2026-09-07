#!/usr/bin/env python3
"""Select the checkpoint with the best stored validation metric."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def checkpoint_summary(path: Path, metric: str) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("metrics") or {}
    if metric not in metrics:
        raise KeyError(f"Checkpoint {path} does not contain validation metric {metric!r}")
    return {
        "path": path.resolve(),
        "epoch": int(payload["epoch"]),
        "metric": float(metrics[metric]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--metric", default="acc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    candidates = [checkpoint_summary(path, args.metric) for path in args.candidate]
    selected = max(candidates, key=lambda item: (float(item["metric"]), int(item["epoch"])))
    source = Path(selected["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != args.output.resolve():
        shutil.copy2(source, args.output)

    metadata = {
        "selection_metric": args.metric,
        "selected": {**selected, "path": str(selected["path"])},
        "candidates": [{**item, "path": str(item["path"])} for item in candidates],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"selected checkpoint={source} epoch={selected['epoch']} "
        f"{args.metric}={selected['metric']:.8f} output={args.output}"
    )


if __name__ == "__main__":
    main()
