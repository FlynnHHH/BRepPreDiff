from __future__ import annotations

import json


def test_prepare_tmcad_classification_generates_one_hot_and_stratified_splits(tmp_path):
    from blendit.data.tmcad_cls import prepare_tmcad_classification

    root = tmp_path / "TMCAD"
    for class_name in ("bolt", "bearing"):
        category = root / class_name
        category.mkdir(parents=True)
        for index in range(10):
            (category / f"{index}.stp").write_text("STEP", encoding="utf-8")

    result = prepare_tmcad_classification(root, ratios=(0.6, 0.2, 0.2), seed=7)

    assert result.class_names == ("bearing", "bolt")
    assert result.model_count == 20
    assert result.written_labels == 20
    assert result.split_counts == {"train": 12, "val": 4, "test": 4}
    assert (root / "bearing" / "0.cls").read_text(encoding="utf-8") == "1 0\n"
    assert (root / "bolt" / "0.cls").read_text(encoding="utf-8") == "0 1\n"

    class_map = json.loads((root / "blendit_splits" / "class_map.json").read_text())
    assert class_map["num_classes"] == 2
    assert [row["class_name"] for row in class_map["classes"]] == ["bearing", "bolt"]

    for split, expected_per_class in (("train", 6), ("val", 2), ("test", 2)):
        items = (root / "blendit_splits" / f"{split}.txt").read_text().splitlines()
        assert sum(item.startswith("bearing/") for item in items) == expected_per_class
        assert sum(item.startswith("bolt/") for item in items) == expected_per_class
