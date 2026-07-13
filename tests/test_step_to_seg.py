from pathlib import Path

import numpy as np
import pytest

from blendit.inference.step_to_seg import (
    build_jobs,
    default_output_dir,
    discover_step_files,
    prediction_to_seg_labels,
    write_seg_file,
)


def test_prediction_classes_are_written_as_requested_seg_labels(tmp_path: Path):
    prediction = np.asarray([0, 1, 2, 2, 1, 0], dtype=np.int64)

    assert prediction_to_seg_labels(prediction).tolist() == [0, 6, 4, 4, 6, 0]

    output = tmp_path / "part.seg"
    write_seg_file(output, prediction)
    assert output.read_bytes() == b"0\n6\n4\n4\n6\n0\n"


def test_prediction_rejects_unknown_class_ids():
    with pytest.raises(ValueError, match="invalid class ids"):
        prediction_to_seg_labels([0, 3])


def test_discover_step_files_is_recursive_and_case_insensitive(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.step").touch()
    (nested / "b.STP").touch()
    (nested / "ignore.txt").touch()

    root, paths = discover_step_files(tmp_path)

    assert root == tmp_path.resolve()
    assert [path.relative_to(root).as_posix() for path in paths] == ["a.step", "nested/b.STP"]


def test_jobs_preserve_relative_directories(tmp_path: Path):
    input_root = tmp_path / "steps"
    nested = input_root / "assembly"
    nested.mkdir(parents=True)
    step = nested / "part.step"
    step.touch()
    output = tmp_path / "predicted"

    jobs = build_jobs(input_root.resolve(), [step.resolve()], output)

    assert jobs[0].relative_path == Path("assembly/part.step")
    assert jobs[0].seg_path == (output / "assembly" / "part.seg").resolve()


def test_default_output_is_a_sibling_directory(tmp_path: Path):
    input_dir = tmp_path / "step_models"
    input_dir.mkdir()
    assert default_output_dir(input_dir) == tmp_path.resolve() / "step_models_seg"
