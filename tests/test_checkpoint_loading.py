from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_load_pretrain_checkpoint_for_finetune_maps_encoder_and_head(tmp_path):
    import torch

    from blendit.config import feature_dims, load_config
    from blendit.models import DiffusionPretrainModel, SegmentationModel
    from blendit.training.common import load_pretrain_checkpoint_for_finetune, save_checkpoint

    config = load_config("configs/default.yaml")
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
