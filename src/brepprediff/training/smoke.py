from __future__ import annotations

import argparse

import torch

from brepprediff.config import feature_dims, load_experiment_config
from brepprediff.data.graph import BRepGraph, collate_graphs
from brepprediff.models import (
    DiffusionPretrainModel,
    DiffusionSchedule,
    LabelDiffusionModel,
    build_finetune_model,
    compute_finetune_label_diffusion_loss,
    compute_finetune_loss,
    prepare_label_diffusion_training_batch,
)
from brepprediff.models.diffusion import compute_pretrain_loss
from brepprediff.task import CLASSIFICATION, task_type
from brepprediff.utils import seed_everything


def synthetic_graph(sample_id: str, num_faces: int, face_dim: int, edge_dim: int, num_classes: int) -> BRepGraph:
    src = torch.arange(0, num_faces - 1, dtype=torch.long)
    dst = torch.arange(1, num_faces, dtype=torch.long)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    num_edges = edge_index.shape[1]
    return BRepGraph(
        face_cont=torch.randn(num_faces, face_dim),
        face_surface_type=torch.randint(1, 6, (num_faces,), dtype=torch.long),
        edge_index=edge_index,
        edge_cont=torch.randn(num_edges, edge_dim),
        edge_type=torch.randint(1, 4, (num_edges,), dtype=torch.long),
        edge_relation=torch.randint(1, 3, (num_edges,), dtype=torch.long),
        labels=torch.randint(0, num_classes, (num_faces,), dtype=torch.long),
        sample_id=sample_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke test without STEP/OCC data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    config = load_experiment_config(args.config, args.data_config, args.override)
    seed_everything(int(config.get("seed", 42)))

    device = torch.device("cpu")
    face_dim, edge_dim = feature_dims(config)
    num_classes = int(config["model"]["num_classes"])
    graphs = [
        synthetic_graph("a", 6, face_dim, edge_dim, num_classes),
        synthetic_graph("b", 5, face_dim, edge_dim, num_classes),
    ]
    batch = collate_graphs(graphs).to(device)

    schedule = DiffusionSchedule(
        int(config["diffusion"]["timesteps"]),
        float(config["diffusion"]["beta_start"]),
        float(config["diffusion"]["beta_end"]),
    )
    pretrain_model = DiffusionPretrainModel(config, face_dim, edge_dim).to(device)
    graph_t = torch.randint(0, schedule.timesteps, (batch.graph_ptr.numel() - 1,), dtype=torch.long)
    face_t = graph_t[batch.batch_index]
    edge_t = graph_t[batch.edge_batch_index]
    face_noisy, face_noise = schedule.q_sample(batch.face_cont, face_t)
    edge_noisy, edge_noise = schedule.q_sample(batch.edge_cont, edge_t)
    pretrain_outputs = pretrain_model(batch, face_noisy, edge_noisy, face_t)
    pretrain_loss, pretrain_metrics = compute_pretrain_loss(
        pretrain_outputs,
        batch,
        face_noise=face_noise,
        edge_noise=edge_noise,
        config=config,
    )
    pretrain_loss.backward()

    finetune_batch = batch
    if task_type(config) == CLASSIFICATION:
        for class_id, graph in enumerate(graphs):
            graph.labels = torch.tensor([class_id % num_classes], dtype=torch.long)
        finetune_batch = collate_graphs(graphs).to(device)

    finetune_model = build_finetune_model(config, face_dim, edge_dim).to(device)
    if isinstance(finetune_model, LabelDiffusionModel):
        prepared = prepare_label_diffusion_training_batch(finetune_model, finetune_batch, config)
        prediction = finetune_model(
            finetune_batch,
            prepared.x_t,
            prepared.timesteps,
            prepared.token_indices,
        )
        seg_loss, seg_metrics = compute_finetune_label_diffusion_loss(
            prediction,
            prepared,
            finetune_model,
            config,
        )
    else:
        logits = finetune_model(finetune_batch)
        seg_loss, seg_metrics = compute_finetune_loss(logits, finetune_batch, config)
    seg_loss.backward()

    print("pretrain", pretrain_metrics)
    print("finetune", seg_metrics)


if __name__ == "__main__":
    main()
