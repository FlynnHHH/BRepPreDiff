from types import SimpleNamespace

import pytest
import torch

from brepprediff.config import feature_dims, load_experiment_config
from brepprediff.models.finetune import build_finetune_model, predict_finetune_probabilities
from brepprediff.training.common import configure_encoder_finetuning, set_frozen_encoder_eval


@pytest.mark.parametrize('config_path,task', [
    ('configs/finetune_joint_fusion360seg_mlp.yaml', 'seg'),
    ('configs/finetune_joint_tmcad_mlp.yaml', 'cls'),
])
def test_probe_only_linear_parameters_update_and_checkpoint_roundtrip(config_path, task, tmp_path):
    cfg = load_experiment_config(config_path)
    cfg['model'].update(finetune_head='linear', graph_pooling='mean_max')
    cfg['train']['encoder_freeze_mode'] = 'all'
    model = build_finetune_model(cfg, *feature_dims(cfg))
    configure_encoder_finetuning(model, cfg)
    model.train(); set_frozen_encoder_eval(model, cfg)
    assert not model.encoder.training
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert [n for n, _ in trainable] == ([f'{task}_head.weight', f'{task}_head.bias'])
    head = getattr(model, f'{task}_head')
    assert type(head) is torch.nn.Linear
    width = cfg['model']['hidden_dim']
    faces = torch.randn(5, width)
    batch = SimpleNamespace(sample_ids=['a', 'b'], batch_index=torch.tensor([0, 0, 1, 1, 1]))
    model.encode_faces = lambda _: faces
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    logits = model(batch)
    torch.testing.assert_close(predict_finetune_probabilities(model, batch, cfg), logits.softmax(-1))
    if task == 'cls':
        pooled = torch.stack([torch.cat((faces[:2].mean(0), faces[:2].max(0).values)),
                              torch.cat((faces[2:].mean(0), faces[2:].max(0).values))])
        torch.testing.assert_close(logits, head(pooled))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=0.01)
    initial_head = head.weight.clone()
    logits.square().mean().backward(); optimizer.step()
    assert not torch.equal(initial_head, head.weight)
    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, before[key])
    path = tmp_path / 'probe.pt'
    torch.save(model.state_dict(), path)
    restored = build_finetune_model(cfg, *feature_dims(cfg))
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.encode_faces = lambda _: faces
    torch.testing.assert_close(restored(batch), model(batch))
