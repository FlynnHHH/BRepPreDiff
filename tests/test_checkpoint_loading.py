from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_load_pretrain_checkpoint_for_finetune_loads_encoder_only(tmp_path):
    import torch

    from brepprediff.config import feature_dims, load_experiment_config
    from brepprediff.models import DiffusionPretrainModel, SegmentationModel
    from brepprediff.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    config = load_experiment_config("configs/default.yaml")
    face_dim, edge_dim = feature_dims(config)

    pretrain = DiffusionPretrainModel(config, face_dim, edge_dim)
    ckpt_path = tmp_path / "pretrain.pt"
    save_checkpoint(ckpt_path, model=pretrain, optimizer=None, epoch=3, config=config)

    segmenter = SegmentationModel(config, face_dim, edge_dim)
    initial_head = segmenter.seg_head.net[0].weight.detach().clone()
    load_info = load_pretrain_checkpoint_for_finetune(ckpt_path, model=segmenter)

    assert any(key.startswith("encoder.") for key in load_info.loaded_keys)
    assert not any(key.startswith("seg_head.") for key in load_info.loaded_keys)
    assert torch.equal(segmenter.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(segmenter.seg_head.net[0].weight, initial_head)


def test_load_pretrain_checkpoint_for_diffusion_finetune_loads_encoder_only(tmp_path):
    import torch

    from brepprediff.config import feature_dims, load_experiment_config
    from brepprediff.models import DiffusionPretrainModel, DiffusionSegmentationModel, build_segmentation_model
    from brepprediff.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    pretrain_config = load_experiment_config("configs/default.yaml")
    finetune_config = load_experiment_config("configs/finetune_diffloss.yaml")
    face_dim, edge_dim = feature_dims(pretrain_config)
    pretrain = DiffusionPretrainModel(pretrain_config, face_dim, edge_dim)
    ckpt_path = tmp_path / "pretrain.pt"
    save_checkpoint(ckpt_path, model=pretrain, optimizer=None, epoch=3, config=pretrain_config)

    segmenter = build_segmentation_model(finetune_config, face_dim, edge_dim)
    assert isinstance(segmenter, DiffusionSegmentationModel)
    initial_head = segmenter.diffusion_head.condition_projection.weight.detach().clone()
    load_info = load_pretrain_checkpoint_for_finetune(ckpt_path, model=segmenter)

    assert any(key.startswith("encoder.") for key in load_info.loaded_keys)
    assert not any(key.startswith("diffusion_head.") for key in load_info.loaded_keys)
    assert torch.equal(segmenter.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(segmenter.diffusion_head.condition_projection.weight, initial_head)


def test_classification_mlp_loads_pretrained_encoder_only(tmp_path):
    import torch

    from brepprediff.config import feature_dims, load_experiment_config
    from brepprediff.models import DiffusionPretrainModel, build_classification_model
    from brepprediff.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    pretrain_config = load_experiment_config("configs/pretrain.yaml")
    finetune_config = load_experiment_config("configs/finetune_tmcad_mlp.yaml")
    face_dim, edge_dim = feature_dims(pretrain_config)
    pretrain = DiffusionPretrainModel(pretrain_config, face_dim, edge_dim)
    checkpoint = tmp_path / "pretrain.pt"
    save_checkpoint(checkpoint, model=pretrain, optimizer=None, epoch=1, config=pretrain_config)

    classifier = build_classification_model(finetune_config, face_dim, edge_dim)
    initial_head = classifier.cls_head.net[0].weight.detach().clone()
    load_info = load_pretrain_checkpoint_for_finetune(checkpoint, model=classifier)

    assert any(key.startswith("encoder.") for key in load_info.loaded_keys)
    assert not any(key.startswith("cls_head.") for key in load_info.loaded_keys)
    assert torch.equal(classifier.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(classifier.cls_head.net[0].weight, initial_head)


def test_load_checkpoint_migrates_legacy_classification_head(tmp_path):
    import torch

    from brepprediff.config import feature_dims, load_experiment_config
    from brepprediff.models import build_classification_model
    from brepprediff.training.common import load_checkpoint

    config = load_experiment_config("configs/finetune_tmcad_mlp.yaml")
    face_dim, edge_dim = feature_dims(config)
    source = build_classification_model(config, face_dim, edge_dim)
    legacy_state = {
        (f"seg_head.{key.removeprefix('cls_head.')}" if key.startswith("cls_head.") else key): value
        for key, value in source.state_dict().items()
    }
    checkpoint = tmp_path / "legacy_cls.pt"
    torch.save({"epoch": 7, "model": legacy_state}, checkpoint)

    restored = build_classification_model(config, face_dim, edge_dim)
    epoch = load_checkpoint(checkpoint, model=restored)

    assert epoch == 7
    assert torch.equal(restored.cls_head.net[0].weight, source.cls_head.net[0].weight)
