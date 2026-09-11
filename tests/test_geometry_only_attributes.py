"""Geometry-only models must be invariant to every discrete attribute."""
import copy

import pytest
import torch

from brepprediff.config import feature_dims, load_experiment_config
from brepprediff.data.graph import collate_graphs
from brepprediff.models import DiffusionPretrainModel, build_finetune_model
from brepprediff.models.diffusion import compute_pretrain_loss
from brepprediff.training.smoke import synthetic_graph


def setup(config_path):
    cfg = load_experiment_config(config_path)
    cfg['model'].update(hidden_dim=32, num_layers=2, dropout=0.0,
                        encoder_type='edge_update_attention', num_heads=4,
                        use_discrete_attributes=False)
    fd, ed = feature_dims(cfg)
    batch = collate_graphs([synthetic_graph('a', 5, fd, ed, 3)])
    changed = copy.deepcopy(batch)
    # Out-of-range sentinels also detect accidental cross-entropy target reads.
    changed.face_surface_type.fill_(999)
    changed.edge_type.fill_(888)
    changed.edge_relation.fill_(777)
    return cfg, fd, ed, batch, changed


def test_pretraining_ignores_discrete_targets_and_inputs():
    cfg, fd, ed, batch, changed = setup('configs/pretrain.yaml')
    cfg['diffusion'].update(categorical_loss_weight=0.0, relation_loss_weight=0.0)
    model = DiffusionPretrainModel(cfg, fd, ed).eval()
    t = torch.zeros(batch.face_cont.shape[0], dtype=torch.long)
    a = model(batch, batch.face_cont, batch.edge_cont, t)
    b = model(changed, batch.face_cont, batch.edge_cont, t)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    kwargs = dict(face_noise=torch.zeros_like(batch.face_cont),
                  edge_noise=torch.zeros_like(batch.edge_cont), config=cfg)
    loss, metrics = compute_pretrain_loss(a, batch, **kwargs)
    other_loss, other_metrics = compute_pretrain_loss(b, changed, **kwargs)
    torch.testing.assert_close(loss, other_loss, rtol=0, atol=0)
    assert metrics == other_metrics
    loss.backward()
    for emb in [model.encoder.surface_emb, model.encoder.edge_type_emb, model.encoder.edge_relation_emb]:
        assert emb.weight.grad is None
    assert model.encoder.face_cont_proj.weight.grad.abs().sum() > 0
    assert model.encoder.edge_cont_proj.weight.grad.abs().sum() > 0
    cfg['diffusion']['categorical_loss_weight'] = 0.5
    with pytest.raises(ValueError, match='zero categorical'):
        compute_pretrain_loss(a, batch, **kwargs)


@pytest.mark.parametrize('path', [
    'configs/finetune_joint_brepprediff_mlp.yaml',
    'configs/finetune_joint_brepprediff_diffloss_200.yaml',
    'configs/finetune_joint_tmcad_mlp.yaml',
    'configs/finetune_joint_tmcad_diffloss_200.yaml',
])
def test_all_downstream_encoders_ignore_discrete_inputs(path):
    cfg, fd, ed, batch, changed = setup(path)
    model = build_finetune_model(cfg, fd, ed).eval()
    a = model.encode_faces(batch)
    b = model.encode_faces(changed)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    assert model.encoder.surface_emb.weight.grad is None
    assert model.encoder.edge_type_emb.weight.grad is None
    assert model.encoder.edge_relation_emb.weight.grad is None


def test_default_keeps_discrete_attributes_and_identical_initialization():
    cfg, fd, ed, batch, changed = setup('configs/pretrain.yaml')
    cfg['model'].pop('use_discrete_attributes')
    torch.manual_seed(42)
    default = DiffusionPretrainModel(cfg, fd, ed).eval()
    cfg['model']['use_discrete_attributes'] = False
    torch.manual_seed(42)
    geometry = DiffusionPretrainModel(cfg, fd, ed).eval()
    for key, weight in default.state_dict().items():
        torch.testing.assert_close(weight, geometry.state_dict()[key], rtol=0, atol=0)
    t = torch.zeros(batch.face_cont.shape[0], dtype=torch.long)
    a = default(batch, batch.face_cont, batch.edge_cont, t)['face_noise']
    b = default(changed, batch.face_cont, batch.edge_cont, t)['face_noise']
    assert not torch.equal(a, b)
