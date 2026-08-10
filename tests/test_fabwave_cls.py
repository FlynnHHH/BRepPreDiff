from pathlib import Path

from blendit.data.fabwave_cls import discover_fabwave, prepare_fabwave_classification


def _step(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("STEP", encoding="utf-8")


def test_prepare_fabwave_discovers_second_level_classes(tmp_path: Path) -> None:
    dataset = tmp_path / "FabWave"
    _step(dataset / "CAD_1_15_Classes" / "Bolts" / "STEP" / "a.stp")
    _step(dataset / "CAD16-24" / "Washers" / "nested" / "b.step")
    (dataset / "CAD25-45" / "Empty").mkdir(parents=True)

    categories, empty = discover_fabwave(dataset)
    assert list(categories) == ["Bolts", "Washers"]
    assert empty == ("CAD25-45/Empty",)

    labels = tmp_path / "labels"
    splits = tmp_path / "splits"
    result = prepare_fabwave_classification(
        dataset,
        labels_root=labels,
        splits_dir=splits,
        ratios=(1, 0, 0),
    )

    assert result.class_names == ("Bolts", "Washers")
    assert result.model_count == 2
    assert result.split_counts == {"train": 2, "val": 0, "test": 0}
    assert (labels / "CAD_1_15_Classes/Bolts/STEP/a.cls").read_text() == "1 0\n"
    assert (labels / "CAD16-24/Washers/nested/b.cls").read_text() == "0 1\n"
