from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_load_pretrain_checkpoint_for_finetune_maps_encoder_and_head(tmp_path):
    import torch

    from blendit.config import feature_dims, load_experiment_config
    from blendit.models import DiffusionPretrainModel, SegmentationModel
    from blendit.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    config = load_experiment_config("configs/default.yaml")
    face_dim, edge_dim = feature_dims(config)

    pretrain = DiffusionPretrainModel(config, face_dim, edge_dim)
    ckpt_path = tmp_path / "pretrain.pt"
    save_checkpoint(ckpt_path, model=pretrain, optimizer=None, epoch=3, config=config)

    segmenter = SegmentationModel(config, face_dim, edge_dim)
    load_info = load_pretrain_checkpoint_for_finetune(ckpt_path, model=segmenter)

    assert any(key.startswith("encoder.") for key in load_info.loaded_keys)
    assert any(key.startswith("seg_head.") for key in load_info.loaded_keys)
    assert torch.equal(segmenter.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(segmenter.seg_head.net[0].weight, pretrain.coarse_label_head.net[0].weight)


def test_load_pretrain_checkpoint_for_diffusion_finetune_loads_encoder_only(tmp_path):
    import torch

    from blendit.config import feature_dims, load_experiment_config
    from blendit.models import DiffusionPretrainModel, DiffusionSegmentationModel, build_segmentation_model
    from blendit.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

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
    assert any("coarse_label_head." in key for key in load_info.skipped_keys)
    assert torch.equal(segmenter.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(segmenter.diffusion_head.condition_projection.weight, initial_head)


def test_load_no_coarse_pretrain_checkpoint_for_mlp_loads_encoder_only(tmp_path):
    import torch

    from blendit.config import feature_dims, load_experiment_config
    from blendit.models import DiffusionPretrainModel, SegmentationModel
    from blendit.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    pretrain_config = load_experiment_config("configs/pretrain_no_coarse.yaml")
    finetune_config = load_experiment_config("configs/finetune.yaml")
    face_dim, edge_dim = feature_dims(pretrain_config)
    pretrain = DiffusionPretrainModel(pretrain_config, face_dim, edge_dim)
    assert pretrain.coarse_label_head is None
    ckpt_path = tmp_path / "pretrain_no_coarse.pt"
    save_checkpoint(ckpt_path, model=pretrain, optimizer=None, epoch=100, config=pretrain_config)

    segmenter = SegmentationModel(finetune_config, face_dim, edge_dim)
    initial_head = segmenter.seg_head.net[0].weight.detach().clone()
    load_info = load_pretrain_checkpoint_for_finetune(ckpt_path, model=segmenter)

    assert any(key.startswith("encoder.") for key in load_info.loaded_keys)
    assert not any(key.startswith("seg_head.") for key in load_info.loaded_keys)
    assert torch.equal(segmenter.encoder.face_cont_proj.weight, pretrain.encoder.face_cont_proj.weight)
    assert torch.equal(segmenter.seg_head.net[0].weight, initial_head)
