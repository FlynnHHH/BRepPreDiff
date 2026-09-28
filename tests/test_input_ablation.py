"""Ensure omitted inputs cannot influence encoder outputs or receive gradients."""
import pytest
import torch
from brepprediff.models.encoder import BRepGraphEncoder


def encoder(mode='none'):
    torch.manual_seed(42)
    return BRepGraphEncoder(39, 15, hidden_dim=16, num_layers=2, dropout=0,
                            surface_type_vocab=8, edge_type_vocab=8,
                            relation_type_vocab=4, encoder_type='edge_update_attention',
                            input_ablation=mode).eval()


def inputs():
    torch.manual_seed(4)
    return [torch.randn(4, 39), torch.tensor([0, 1, 2, 3]),
            torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]]),
            torch.randn(5, 15), torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 0])]


@pytest.mark.parametrize('mode', ['face_geometry', 'edge_geometry', 'uv_grid', 'face_edge_grid'])
def test_masked_information_cannot_change_outputs(mode):
    model = encoder(mode)
    x = inputs(); changed = [t.clone() for t in x]
    if mode == 'face_geometry':
        changed[0][:, :11] += 100
        changed[1] = (changed[1] + 2) % 8
    elif mode == 'edge_geometry':
        changed[3] += 100
        changed[4] = (changed[4] + 2) % 8
        changed[5] = (changed[5] + 2) % 4
    else:
        changed[0][:, 11:] += 100
        if mode == 'face_edge_grid': changed[3][:, 3:] += 100
    for a, b in zip(model(*x), model(*changed)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # The same perturbation must affect the complete-input control.
    base = encoder()
    assert not torch.allclose(base(*x)[0], base(*changed)[0])
    x[0].requires_grad_(); x[3].requires_grad_()
    nodes, edges = model(*x)
    (nodes[:, 0].sum() + edges[:, 0].sum()).backward()
    if mode == 'edge_geometry':
        assert x[3].grad is None or x[3].grad.count_nonzero() == 0
        assert model.edge_type_emb.weight.grad is None
        assert model.edge_relation_emb.weight.grad is None
        assert x[0].grad.abs().sum() > 0
    else:
        omitted = slice(0, 11) if mode == 'face_geometry' else slice(11, None)
        kept = slice(11, None) if mode == 'face_geometry' else slice(0, 11)
        assert x[0].grad[:, omitted].count_nonzero() == 0
        assert x[0].grad[:, kept].abs().sum() > 0
        if mode == 'face_geometry': assert model.surface_emb.weight.grad is None
        if mode == 'face_edge_grid':
            assert x[3].grad[:, 3:].count_nonzero() == 0
            assert x[3].grad[:, :3].abs().sum() > 0
            assert model.surface_emb.weight.grad.abs().sum() > 0
            assert model.edge_type_emb.weight.grad.abs().sum() > 0
            assert model.edge_relation_emb.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['none', 'face_geometry', 'edge_geometry', 'uv_grid', 'face_edge_grid'])
def test_same_initialization_checkpoint_shapes_and_empty_edges(mode):
    base, model = encoder(), encoder(mode)
    assert base.state_dict().keys() == model.state_dict().keys()
    for k, v in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
    x = inputs(); x[2] = x[2][:, :0]
    for i in (3, 4, 5): x[i] = x[i][:0]
    nodes, edges = model(*x)
    assert torch.isfinite(nodes).all() and edges.shape == (0, 16)


def test_config_roundtrip_and_invalid_mode():
    cfg = dict(model=dict(hidden_dim=16, num_layers=2, dropout=0, input_ablation='uv_grid'),
               brep=dict(surface_type_vocab=8, edge_type_vocab=8, relation_type_vocab=4))
    assert BRepGraphEncoder.from_config(cfg, 39, 15).input_ablation == 'uv_grid'
    cfg['model'].pop('input_ablation')
    assert BRepGraphEncoder.from_config(cfg, 39, 15).input_ablation == 'none'
    with pytest.raises(ValueError, match='input_ablation'):
        encoder('typo')
