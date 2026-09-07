#!/usr/bin/env python3
"""Recover the official MFCAD++ split entries encoded in cache file hashes."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 41_766, "val": 8_950, "test": 8_949}
CACHE_NAME = re.compile(r"(.+)_([0-9a-f]{10})\.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/features/mfcad"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    recovered: dict[str, list[str]] = {split: [] for split in SPLITS}
    for cache_path in sorted(args.cache_dir.glob("*.npz")):
        match = CACHE_NAME.fullmatch(cache_path.name)
        if match is None:
            raise ValueError(f"Unexpected MFCAD++ cache filename: {cache_path}")
        model_id, encoded_digest = match.groups()
        matching_splits = [
            split
            for split in SPLITS
            if hashlib.sha1(f"{split}/{model_id}.step".encode("utf-8")).hexdigest()[:10]
            == encoded_digest
        ]
        if len(matching_splits) != 1:
            raise ValueError(
                f"Cannot uniquely recover the official split for cache file {cache_path}"
            )
        split = matching_splits[0]
        recovered[split].append(f"{split}/{model_id}.step")

    counts = {split: len(entries) for split, entries in recovered.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Recovered MFCAD++ counts {counts}, expected {EXPECTED_COUNTS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, entries in recovered.items():
        output = args.output_dir / f"mfcad_{split}.txt"
        output.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        print(f"{split}: {len(entries)} -> {output}")


if __name__ == "__main__":
    main()
