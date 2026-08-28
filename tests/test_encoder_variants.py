from __future__ import annotations

import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)


def test_ffn_encoder_type_alias_selects_message_passing():
    from blendit.config import feature_dims, load_experiment_config
    from blendit.models.encoder import BRepGraphEncoder, GraphMessageLayer

    config = load_experiment_config(
        "configs/pretrain.yaml",
        overrides=["model.encoder_type=ffn"],
    )
    face_dim, edge_dim = feature_dims(config)
    model = BRepGraphEncoder.from_config(config, face_dim, edge_dim)

    assert model.encoder_type == "message_passing"
    assert all(isinstance(layer, GraphMessageLayer) for layer in model.layers)


def test_edge_update_attention_mean_max_classification_forward_backward():
    import torch

    from blendit.config import feature_dims, load_experiment_config
    from blendit.data.graph import collate_graphs
    from blendit.models import ClassificationModel
    from blendit.training.smoke import synthetic_graph

    config = load_experiment_config("configs/finetune_tmcad_mlp.yaml")
    config["model"].update(
        hidden_dim=32,
        num_layers=2,
        encoder_type="edge_update_attention",
        num_heads=4,
        graph_pooling="mean_max",
    )
    face_dim, edge_dim = feature_dims(config)
    graphs = [
        synthetic_graph("a", 4, face_dim, edge_dim, 10),
        synthetic_graph("b", 6, face_dim, edge_dim, 10),
    ]
    graphs[0].labels = torch.tensor([0])
    graphs[1].labels = torch.tensor([1])
    batch = collate_graphs(graphs)
    model = ClassificationModel(config, face_dim, edge_dim)

    logits = model(batch)

    assert logits.shape == (2, 10)
    logits.square().mean().backward()
    assert model.encoder.layers[0].edge_update is not None
    assert model.encoder.layers[0].query.weight.grad is not None
    assert model.graph_pool.residual[-1].weight.grad is not None


def test_edge_update_attention_changes_edge_embeddings():
    import torch

    from blendit.config import feature_dims, load_experiment_config
    from blendit.data.graph import collate_graphs
    from blendit.models.encoder import BRepGraphEncoder
    from blendit.training.smoke import synthetic_graph

    config = load_experiment_config("configs/finetune_tmcad_mlp.yaml")
    config["model"].update(
        hidden_dim=32,
        num_layers=2,
        encoder_type="edge_update_attention",
        num_heads=4,
    )
    face_dim, edge_dim = feature_dims(config)
    batch = collate_graphs([synthetic_graph("a", 5, face_dim, edge_dim, 10)])
    model = BRepGraphEncoder.from_config(config, face_dim, edge_dim)
    initial_edges = model.embed_edges(
        batch.edge_cont,
        batch.edge_type,
        batch.edge_relation,
    )

    _, final_edges = model(
        batch.face_cont,
        batch.face_surface_type,
        batch.edge_index,
        batch.edge_cont,
        batch.edge_type,
        batch.edge_relation,
        graph_ptr=batch.graph_ptr,
    )

    assert not torch.allclose(initial_edges, final_edges)
