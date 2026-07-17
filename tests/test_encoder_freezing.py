from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


@pytest.mark.parametrize(
    ("mode", "expected_frozen_layers"),
    [("none", 0), ("all", 4), ("partial", 2)],
)
def test_encoder_freeze_strategies(mode, expected_frozen_layers):
    from blendit.config import feature_dims, load_experiment_config
    from blendit.models import SegmentationModel
    from blendit.training.common import configure_encoder_finetuning, set_frozen_encoder_eval

    config = load_experiment_config("configs/finetune.yaml")
    config["train"]["encoder_freeze_mode"] = mode
    config["train"]["encoder_frozen_layers"] = 2
    face_dim, edge_dim = feature_dims(config)
    model = SegmentationModel(config, face_dim, edge_dim)
    result = configure_encoder_finetuning(model, config)

    assert result.mode == mode
    assert result.frozen_layers == expected_frozen_layers
    assert all(parameter.requires_grad for parameter in model.seg_head.parameters())
    if mode == "none":
        assert all(parameter.requires_grad for parameter in model.encoder.parameters())
        assert result.frozen_parameters == 0
    elif mode == "all":
        assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
        assert result.frozen_parameters > 0
    else:
        assert not any(parameter.requires_grad for parameter in model.encoder.face_cont_proj.parameters())
        assert not any(parameter.requires_grad for parameter in model.encoder.layers[0].parameters())
        assert not any(parameter.requires_grad for parameter in model.encoder.layers[1].parameters())
        assert all(parameter.requires_grad for parameter in model.encoder.layers[2].parameters())
        assert all(parameter.requires_grad for parameter in model.encoder.layers[3].parameters())

    model.train()
    set_frozen_encoder_eval(model, config)
    if mode == "all":
        assert not model.encoder.training
    elif mode == "partial":
        assert not model.encoder.layers[0].training
        assert model.encoder.layers[2].training


def test_partial_encoder_freeze_rejects_invalid_layer_count():
    from blendit.config import feature_dims, load_experiment_config
    from blendit.models import SegmentationModel
    from blendit.training.common import configure_encoder_finetuning

    config = load_experiment_config("configs/finetune.yaml")
    config["train"]["encoder_freeze_mode"] = "partial"
    config["train"]["encoder_frozen_layers"] = int(config["model"]["num_layers"])
    face_dim, edge_dim = feature_dims(config)
    model = SegmentationModel(config, face_dim, edge_dim)
    with pytest.raises(ValueError, match="encoder_frozen_layers"):
        configure_encoder_finetuning(model, config)
