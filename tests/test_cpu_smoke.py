from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")


def test_cpu_smoke_forward_backward():
    from blendit.config import feature_dims, load_experiment_config
    from blendit.training.smoke import synthetic_graph
    from blendit.data.graph import collate_graphs
    from blendit.models import DiffusionPretrainModel, DiffusionSchedule, SegmentationModel
    from blendit.models.diffusion import compute_pretrain_loss
    from blendit.models.segmentation import compute_segmentation_loss

    import torch

    config = load_experiment_config("configs/default.yaml")
    face_dim, edge_dim = feature_dims(config)
    num_classes = int(config["model"]["num_classes"])
    batch = collate_graphs(
        [
            synthetic_graph("a", 4, face_dim, edge_dim, num_classes),
            synthetic_graph("b", 5, face_dim, edge_dim, num_classes),
        ]
    )

    schedule = DiffusionSchedule(
        int(config["diffusion"]["timesteps"]),
        float(config["diffusion"]["beta_start"]),
        float(config["diffusion"]["beta_end"]),
    )
    graph_t = torch.randint(0, schedule.timesteps, (batch.graph_ptr.numel() - 1,), dtype=torch.long)
    face_t = graph_t[batch.batch_index]
    edge_t = graph_t[batch.edge_batch_index]
    face_noisy, face_noise = schedule.q_sample(batch.face_cont, face_t)
    edge_noisy, edge_noise = schedule.q_sample(batch.edge_cont, edge_t)

    pretrain = DiffusionPretrainModel(config, face_dim, edge_dim)
    outputs = pretrain(batch, face_noisy, edge_noisy, face_t)
    loss, metrics = compute_pretrain_loss(
        outputs,
        batch,
        face_noise=face_noise,
        edge_noise=edge_noise,
        config=config,
    )
    assert metrics["total"] > 0
    loss.backward()

    segmenter = SegmentationModel(config, face_dim, edge_dim)
    logits = segmenter(batch)
    seg_loss, seg_metrics = compute_segmentation_loss(logits, batch, config)
    assert seg_metrics["total"] > 0
    seg_loss.backward()


def test_pretrain_without_coarse_label_head_removes_module_and_loss():
    from blendit.config import feature_dims, load_experiment_config
    from blendit.data.graph import collate_graphs
    from blendit.models import DiffusionPretrainModel, DiffusionSchedule, compute_pretrain_loss
    from blendit.training.smoke import synthetic_graph

    import torch

    config = load_experiment_config("configs/default.yaml")
    config["model"]["use_coarse_label_head"] = False
    config["diffusion"]["coarse_label_loss_weight"] = 0.0
    face_dim, edge_dim = feature_dims(config)
    batch = collate_graphs([synthetic_graph("no-coarse", 4, face_dim, edge_dim, 3)])
    schedule = DiffusionSchedule(10, 1.0e-4, 0.02)
    face_t = torch.zeros(batch.face_cont.shape[0], dtype=torch.long)
    edge_t = torch.zeros(batch.edge_cont.shape[0], dtype=torch.long)
    face_noisy, face_noise = schedule.q_sample(batch.face_cont, face_t)
    edge_noisy, edge_noise = schedule.q_sample(batch.edge_cont, edge_t)

    model = DiffusionPretrainModel(config, face_dim, edge_dim)
    assert model.coarse_label_head is None
    assert not any(name.startswith("coarse_label_head.") for name, _ in model.named_parameters())
    outputs = model(batch, face_noisy, edge_noisy, face_t)
    assert "coarse_label_logits" not in outputs
    loss, metrics = compute_pretrain_loss(
        outputs,
        batch,
        face_noise=face_noise,
        edge_noise=edge_noise,
        config=config,
    )
    assert torch.isfinite(loss)
    assert "coarse_label" not in metrics
