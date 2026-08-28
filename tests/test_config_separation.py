from __future__ import annotations

from pathlib import Path

import yaml

from blendit.config import load_config, load_experiment_config


DATA_CONFIGS = (
    "data/default.yaml",
    "data/pretrain.yaml",
    "data/finetune.yaml",
    "data/filletrec.yaml",
    "data/mfcad.yaml",
    "data/fusion360seg.yaml",
    "data/pretrain_joint_all_splits.yaml",
)
TRAINING_CONFIGS = (
    "configs/default.yaml",
    "configs/pretrain.yaml",
    "configs/finetune.yaml",
    "configs/finetune_diffloss.yaml",
    "configs/finetune_filletrec_diffloss.yaml",
    "configs/finetune_mfcad_baseline.yaml",
    "configs/finetune_mfcad_mlp.yaml",
    "configs/finetune_fusion360seg_mlp_full.yaml",
    "configs/pretrain_joint_all_splits.yaml",
    "configs/finetune_joint_blendit_mlp.yaml",
    "configs/finetune_joint_blendit_diffloss.yaml",
    "configs/finetune_joint_tmcad_mlp.yaml",
    "configs/finetune_joint_fusion360seg_mlp.yaml",
    "configs/finetune_joint_mfcadpp_mlp.yaml",
)


def test_data_configs_contain_only_prepare_data_sections():
    for path in DATA_CONFIGS:
        assert set(load_config(path)) <= {"data", "brep", "labels"}, path


def test_training_configs_reference_data_without_embedding_prepare_sections():
    for path in TRAINING_CONFIGS:
        config = load_config(path)
        assert "data_config" in config, path
        assert "data" not in config, path
        assert "brep" not in config, path
        assert "raw_to_class_map" not in config.get("labels", {}), path


def test_experiment_config_resolves_relative_data_config_and_applies_overrides(tmp_path: Path):
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "configs"
    data_dir.mkdir()
    config_dir.mkdir()
    data_path = data_dir / "prepare.yaml"
    training_path = config_dir / "train.yaml"
    data_path.write_text(
        yaml.safe_dump({"data": {"cache_dir": "cache"}, "brep": {"uv_grid_size": 4}}),
        encoding="utf-8",
    )
    training_path.write_text(
        yaml.safe_dump({"data_config": "../data/prepare.yaml", "train": {"epochs": 2}}),
        encoding="utf-8",
    )

    config = load_experiment_config(training_path, overrides=["train.epochs=3"])

    assert config["data"]["cache_dir"] == "cache"
    assert config["brep"]["uv_grid_size"] == 4
    assert config["train"]["epochs"] == 3
    assert Path(config["data_config"]) == config_dir / "../data/prepare.yaml"


def test_experiment_config_parses_list_override(tmp_path: Path):
    data_path = tmp_path / "data.yaml"
    training_path = tmp_path / "training.yaml"
    data_path.write_text("data: {}\n", encoding="utf-8")
    training_path.write_text(
        f"data_config: {data_path}\ntrain:\n  class_weights: null\n",
        encoding="utf-8",
    )

    config = load_experiment_config(
        training_path,
        overrides=["train.class_weights=[0.67,1.37,0.96]"],
    )

    assert config["train"]["class_weights"] == [0.67, 1.37, 0.96]


def test_classification_training_configs_default_to_mean_max():
    classification_configs = []
    for path in sorted(Path("configs").glob("*.yaml")):
        training_config = load_config(path)
        configured_data_path = training_config.get("data_config")
        if not configured_data_path:
            continue
        data_path = (path.parent / configured_data_path).resolve()
        if load_config(data_path).get("task") != "cls":
            continue
        classification_configs.append(path)
        assert training_config["model"].get("graph_pooling") == "mean_max", path

    assert classification_configs


def test_default_training_configs_use_edge_update_attention():
    for path in ("configs/default.yaml", "configs/pretrain.yaml", "configs/finetune.yaml"):
        model_config = load_config(path)["model"]
        assert model_config["encoder_type"] == "edge_update_attention", path
        assert model_config["num_heads"] == 4, path
