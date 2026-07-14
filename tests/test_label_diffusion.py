from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def _small_diffusion_config():
    from blendit.config import load_config

    config = load_config("configs/finetune_diffloss.yaml")
    config["model"]["hidden_dim"] = 32
    config["model"]["num_layers"] = 2
    config["label_diffusion"]["head_width"] = 32
    config["label_diffusion"]["head_depth"] = 2
    config["label_diffusion"]["train_timesteps"] = 20
    config["label_diffusion"]["sampling_steps"] = 5
    config["label_diffusion"]["noise_samples_per_token"] = 2
    config["train"]["class_weights"] = None
    return config


def test_bipolar_one_hot_encoding():
    import torch

    from blendit.models import bipolar_one_hot

    encoded = bipolar_one_hot(torch.tensor([0, 1, 2]), num_classes=3)
    expected = torch.tensor(
        [
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    assert torch.equal(encoded, expected)


def test_diffusion_head_forward_backward_and_sampling():
    import torch

    from blendit.config import feature_dims
    from blendit.data.graph import collate_graphs
    from blendit.models import (
        DiffusionSegmentationModel,
        build_segmentation_model,
        compute_label_diffusion_loss,
        predict_segmentation_probabilities,
        prepare_label_diffusion_training_batch,
    )
    from blendit.training.smoke import synthetic_graph

    config = _small_diffusion_config()
    face_dim, edge_dim = feature_dims(config)
    batch = collate_graphs(
        [
            synthetic_graph("a", 4, face_dim, edge_dim, 3),
            synthetic_graph("b", 5, face_dim, edge_dim, 3),
        ]
    )
    model = build_segmentation_model(config, face_dim, edge_dim)
    assert isinstance(model, DiffusionSegmentationModel)

    prepared = prepare_label_diffusion_training_batch(model, batch, config)
    assert prepared.x_start.shape == (18, 3)
    prediction = model(batch, prepared.x_t, prepared.timesteps, prepared.face_indices)
    assert prediction.shape == prepared.noise.shape
    loss, metrics = compute_label_diffusion_loss(prediction, prepared, model)
    assert torch.isfinite(loss)
    assert metrics["total"] > 0.0
    loss.backward()
    assert model.encoder.face_cont_proj.weight.grad is not None

    model.eval()
    with torch.inference_mode():
        first = predict_segmentation_probabilities(model, batch, config)
        second = predict_segmentation_probabilities(model, batch, config)
    assert first.shape == (9, 3)
    assert torch.allclose(first.sum(dim=-1), torch.ones(9), atol=1.0e-6)
    assert torch.equal(first, second)


def test_label_diffusion_ignores_unlabelled_faces():
    import torch

    from blendit.config import feature_dims
    from blendit.data.graph import collate_graphs
    from blendit.models import DiffusionSegmentationModel, build_segmentation_model, prepare_label_diffusion_training_batch
    from blendit.training.smoke import synthetic_graph

    config = _small_diffusion_config()
    face_dim, edge_dim = feature_dims(config)
    graph = synthetic_graph("ignored", 4, face_dim, edge_dim, 3)
    graph.labels[1] = int(config["train"]["ignore_index"])
    batch = collate_graphs([graph])
    model = build_segmentation_model(config, face_dim, edge_dim)
    assert isinstance(model, DiffusionSegmentationModel)

    prepared = prepare_label_diffusion_training_batch(model, batch, config)
    assert prepared.labels.numel() == 3 * int(config["label_diffusion"]["noise_samples_per_token"])
    assert not (prepared.face_indices == 1).any()
