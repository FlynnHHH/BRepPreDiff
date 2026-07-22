from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


def _classification_config(head: str):
    from blendit.config import load_experiment_config

    config = load_experiment_config(f"configs/finetune_tmcad_{head}.yaml")
    config["model"]["hidden_dim"] = 32
    config["model"]["num_layers"] = 2
    if head == "diffloss":
        config["label_diffusion"]["head_width"] = 32
        config["label_diffusion"]["head_depth"] = 2
        config["label_diffusion"]["train_timesteps"] = 20
        config["label_diffusion"]["sampling_steps"] = 4
        config["label_diffusion"]["noise_samples_per_token"] = 2
    return config


def _classification_batch(config):
    import torch

    from blendit.config import feature_dims
    from blendit.data.graph import collate_graphs
    from blendit.training.smoke import synthetic_graph

    face_dim, edge_dim = feature_dims(config)
    graphs = [
        synthetic_graph("bearing/a", 4, face_dim, edge_dim, 10),
        synthetic_graph("gear/b", 6, face_dim, edge_dim, 10),
    ]
    graphs[0].labels = torch.tensor([0], dtype=torch.long)
    graphs[1].labels = torch.tensor([5], dtype=torch.long)
    return collate_graphs(graphs), face_dim, edge_dim


def test_read_class_label_validates_one_hot(tmp_path):
    from blendit.data.classification import read_class_label

    path = tmp_path / "part.cls"
    path.write_text("0 0 1 0\n", encoding="utf-8")
    config = {"labels": {"num_classes": 4}}

    assert np.array_equal(read_class_label(path, config), np.asarray([2], dtype=np.int64))

    path.write_text("0 1 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_class_label(path, config)


def test_pair_step_cls_paths_uses_relative_layout(tmp_path):
    from blendit.training.evaluate import pair_step_seg_paths

    step_root = tmp_path / "steps"
    label_root = tmp_path / "labels"
    (step_root / "gear").mkdir(parents=True)
    (label_root / "gear").mkdir(parents=True)
    step_path = step_root / "gear" / "part.stp"
    cls_path = label_root / "gear" / "part.cls"
    step_path.write_text("STEP", encoding="utf-8")
    cls_path.write_text("0 1\n", encoding="utf-8")

    samples = pair_step_seg_paths(step_root, label_root, task="cls")

    assert len(samples) == 1
    assert samples[0].sample_id == "gear/part"
    assert samples[0].step_path == step_path.resolve()
    assert samples[0].seg_path == cls_path.resolve()


def test_mlp_classification_pools_faces_to_one_logit_per_graph():
    import torch

    from blendit.models import (
        ClassificationModel,
        build_classification_model,
        compute_classification_loss,
    )

    config = _classification_config("mlp")
    batch, face_dim, edge_dim = _classification_batch(config)
    model = build_classification_model(config, face_dim, edge_dim)
    assert isinstance(model, ClassificationModel)

    logits = model(batch)
    assert logits.shape == (2, 10)
    loss, metrics = compute_classification_loss(logits, batch, config)
    assert torch.isfinite(loss)
    assert "dice" not in metrics
    loss.backward()


def test_task_specific_builders_reject_the_other_task():
    from blendit.models import build_classification_model, build_segmentation_model

    classification_config = _classification_config("mlp")
    _, face_dim, edge_dim = _classification_batch(classification_config)
    with pytest.raises(ValueError, match="Segmentation model received task='cls'"):
        build_segmentation_model(classification_config, face_dim, edge_dim)

    segmentation_config = dict(classification_config)
    segmentation_config["task"] = "seg"
    with pytest.raises(ValueError, match="Classification model received task='seg'"):
        build_classification_model(segmentation_config, face_dim, edge_dim)


def test_diffloss_classification_trains_and_samples_graph_tokens():
    import torch

    from blendit.models import (
        DiffusionClassificationModel,
        build_classification_model,
        compute_classification_label_diffusion_loss,
        predict_classification_probabilities,
        prepare_label_diffusion_training_batch,
    )

    config = _classification_config("diffloss")
    batch, face_dim, edge_dim = _classification_batch(config)
    model = build_classification_model(config, face_dim, edge_dim)
    assert isinstance(model, DiffusionClassificationModel)

    prepared = prepare_label_diffusion_training_batch(model, batch, config)
    assert prepared.labels.shape == (4,)
    prediction = model(batch, prepared.x_t, prepared.timesteps, prepared.face_indices)
    loss, metrics = compute_classification_label_diffusion_loss(
        prediction,
        prepared,
        model,
    )
    assert torch.isfinite(loss)
    assert "dice" not in metrics
    loss.backward()

    model.eval()
    with torch.inference_mode():
        probabilities = predict_classification_probabilities(model, batch, config)
    assert probabilities.shape == (2, 10)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2), atol=1.0e-6)
