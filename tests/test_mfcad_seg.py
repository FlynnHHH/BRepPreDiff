from __future__ import annotations

from pathlib import Path
import re

import pytest

import blendit.data.mfcad_seg as mfcad_seg
from blendit.data.mfcad_seg import (
    MFCAD_CLASS_NAMES,
    MFCAD_FEATURE_CLASSES,
    MFCAD_SEGMENTATION_CLASSES,
    MFCAD_STOCK_LABEL,
    convert_mfcad_dataset,
    write_blendit_split_files,
)


def _write_step(path: Path, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entities = "\n".join(
        f"#{index} = ADVANCED_FACE('{label}',(),#99,.T.);"
        for index, label in enumerate(labels, start=1)
    )
    path.write_text(f"ISO-10303-21;\nDATA;\n{entities}\nENDSEC;\n", encoding="utf-8")


def test_mfcad_has_24_features_plus_stock_background() -> None:
    assert MFCAD_FEATURE_CLASSES == 24
    assert MFCAD_STOCK_LABEL == 24
    assert MFCAD_SEGMENTATION_CLASSES == len(MFCAD_CLASS_NAMES) == 25
    assert MFCAD_CLASS_NAMES[MFCAD_STOCK_LABEL] == "Stock"


def test_parse_mfcad_face_label_rejects_non_class_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Invalid MFCAD\+\+ face label"):
        mfcad_seg._parse_mfcad_label("stock", tmp_path / "part.step", 0)


def test_parse_mfcad_face_label_rejects_noncanonical_integer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside canonical integer ids"):
        mfcad_seg._parse_mfcad_label("025", tmp_path / "part.step", 0)


def test_convert_dataset_preserves_split_layout_and_validates_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step_root = tmp_path / "step"
    seg_root = tmp_path / "seg"
    _write_step(step_root / "train" / "7.step", [1, 24, 8])

    def labels_from_test_step(path: str | Path) -> list[int]:
        text = Path(path).read_text(encoding="utf-8")
        return [int(value) for value in re.findall(r"ADVANCED_FACE\('([0-9]+)'", text)]

    monkeypatch.setattr(mfcad_seg, "extract_mfcad_face_labels", labels_from_test_step)

    first = convert_mfcad_dataset(step_root, seg_root, splits=["train"], workers=1)
    second = convert_mfcad_dataset(step_root, seg_root, splits=["train"], workers=1)

    assert (seg_root / "train" / "7.seg").read_text(encoding="utf-8") == "1\n24\n8\n"
    assert first.models == first.written == 1
    assert first.faces == 3
    assert first.label_counts == {1: 1, 8: 1, 24: 1}
    assert second.written == 0
    assert second.unchanged == 1


def test_write_blendit_split_files_adds_split_prefix(tmp_path: Path) -> None:
    dataset_root = tmp_path / "MFCAD++"
    step_root = dataset_root / "step"
    _write_step(step_root / "train" / "7.step", [24])
    _write_step(step_root / "train" / "12.step", [0])
    dataset_root.mkdir(exist_ok=True)
    (dataset_root / "train.txt").write_text("12\n7\n", encoding="utf-8")

    outputs = write_blendit_split_files(
        dataset_root,
        step_root,
        dataset_root / "blendit_splits",
        splits=["train"],
    )

    assert outputs["train"].read_text(encoding="utf-8") == "train/12.step\ntrain/7.step\n"
