from __future__ import annotations

import json
from pathlib import Path

from brepprediff.data.solidletters_cls import prepare_solidletters_classification


def test_prepare_solidletters_uses_prefix_labels_and_preserves_test_split(tmp_path: Path) -> None:
    root = tmp_path / "SolidLetters"
    steps = root / "step"
    steps.mkdir(parents=True)
    train: list[str] = []
    test: list[str] = []
    for letter in ("a", "b"):
        for index in range(10):
            stem = f"{letter}_Font {index}_upper"
            (steps / f"{stem}.step").write_text("STEP", encoding="utf-8")
            (test if index >= 8 else train).append(stem)
    (steps / "b_Unlisted_lower.step").write_text("STEP", encoding="utf-8")
    (root / "train.txt").write_text("\r\n".join(train) + "\r\n", encoding="utf-8")
    (root / "test.txt").write_text("\r\n".join(test) + "\r\n", encoding="utf-8")

    labels = tmp_path / "labels"
    splits = tmp_path / "splits"
    result = prepare_solidletters_classification(
        root,
        labels_root=labels,
        splits_dir=splits,
        validation_ratio=0.25,
        seed=7,
    )

    assert result.model_count == 21
    assert result.written_labels == 21
    assert result.split_counts == {"train": 12, "val": 4, "test": 4}
    assert result.unlisted_models == ("b_Unlisted_lower",)
    assert (labels / "a_Font 0_upper.cls").read_text().split() == ["1"] + ["0"] * 25
    assert (labels / "b_Font 0_upper.cls").read_text().split() == ["0", "1"] + ["0"] * 24
    assert set((splits / "solidletters_test.txt").read_text().splitlines()) == {
        f"{letter}_Font {index}_upper.step"
        for letter in ("a", "b")
        for index in (8, 9)
    }

    class_map = json.loads((splits / "solidletters_class_map.json").read_text())
    assert class_map["num_classes"] == 26
    assert class_map["unlisted_models_excluded"] == ["b_Unlisted_lower"]
