from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).parents[1] / "scripts" / "update_finetune_test_visualization.py"
    spec = importlib.util.spec_from_file_location("blendit_visualization_install", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_result_set(directory: Path, manifest_name: str, sample_id: str) -> None:
    input_name = f"{sample_id}_instance_pred_rgb.ply"
    semantic_name = f"{sample_id}_semantic_pred.ply"
    (directory / input_name).write_text("ply\n", encoding="utf-8")
    (directory / semantic_name).write_text("ply\n", encoding="utf-8")
    (directory / manifest_name).write_text(
        json.dumps(
            {
                "evaluated_samples": 1,
                "samples": [
                    {
                        "sample_id": sample_id,
                        "input_ply": input_name,
                        "semantic_ply": semantic_name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_validate_archive_and_install_visualization_results(tmp_path: Path) -> None:
    module = _load_module()
    staging = tmp_path / "staging"
    results = tmp_path / "results"
    archive = tmp_path / "archive"
    staging.mkdir()
    results.mkdir()
    archive.mkdir()
    _write_result_set(staging, module.MANIFEST_NAME, "new")
    _write_result_set(results, "original_testset_manifest.json", "old")

    manifest = module.validate_staging(staging, expected_samples=1)
    moved = module.archive_manifest_results(results / "original_testset_manifest.json", archive)
    installed = module.install_staging(staging, results)

    assert manifest["samples"][0]["sample_id"] == "new"
    assert moved == 2
    assert (archive / "old_instance_pred_rgb.ply").exists()
    assert (archive / "original_testset_manifest.json").exists()
    assert module.MANIFEST_NAME in installed
    assert (results / module.MANIFEST_NAME).exists()
    assert (results / "new_semantic_pred.ply").exists()
