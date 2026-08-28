from __future__ import annotations

import numpy as np

from brepprediff.brep.occ_extractor import _parse_label_map, _remap_labels
from brepprediff.brep.occ_extractor import OccBRepExtractor


def test_brepdit_raw_seg_labels_map_to_three_classes():
    raw_labels = np.arange(8, dtype=np.int64)
    mapped = _remap_labels(
        raw_labels,
        _parse_label_map({4: 2, 6: 1}),
        default_class=0,
        ignore_index=-100,
    )
    np.testing.assert_array_equal(mapped, np.array([0, 0, 0, 0, 2, 0, 1, 0], dtype=np.int64))


def test_unmapped_raw_labels_can_be_rejected():
    try:
        _remap_labels(
            np.array([4, 6, 8], dtype=np.int64),
            _parse_label_map("4:2,6:1"),
            default_class=None,
            ignore_index=-100,
        )
    except ValueError as exc:
        assert "8" in str(exc)
    else:
        raise AssertionError("Expected unmapped raw labels to raise ValueError")


def test_filletrec_json_labels_are_read_without_seg_conversion(tmp_path):
    label_path = tmp_path / "42.json"
    label_path.write_text("[0, 1, 1, 0]", encoding="utf-8")
    extractor = OccBRepExtractor.__new__(OccBRepExtractor)
    extractor.label_offset = 0
    extractor.label_map = None
    extractor.label_default_class = None
    extractor.ignore_index = -100

    labels = extractor._read_labels(label_path, 4, True, True)

    np.testing.assert_array_equal(labels, np.array([0, 1, 1, 0], dtype=np.int64))
