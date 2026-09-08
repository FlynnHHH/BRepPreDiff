import pytest
import torch


def test_operation_auxiliary_head_gradients_and_ignored_labels():
    from brepprediff.models.segmentation import SegmentationModel, compute_segmentation_loss
    config = load_experiment_config('configs/finetune_joint_fusion360seg_mlp.yaml')
    config['model'].update(hidden_dim=16, num_layers=2, operation_class_map=[0,0,1,1,2,3,4,4])
    dims = feature_dims(config)
    batch = collate_graphs([synthetic_graph('operation', 8, *dims, 8)])
    batch.labels = torch.tensor([0,1,2,3,4,5,6,-100])
    model = SegmentationModel(config, *dims)
    output = model(batch, return_aux=True)
    loss, metrics = compute_segmentation_loss(output, batch, config)
    loss.backward()
    assert torch.isfinite(loss) and metrics['operation'] > 0
    assert model.operation_head.net[-1].weight.grad.abs().sum() > 0
    assert model.eval()(batch).shape == (8,8)

from brepprediff.config import feature_dims, load_experiment_config
from brepprediff.data.graph import collate_graphs
from brepprediff.models.diffusion import DiffusionPretrainModel, compute_pretrain_loss
from brepprediff.models.encoder import BRepGraphEncoder
from brepprediff.training.smoke import synthetic_graph


@pytest.mark.parametrize('variant', ['grid_encoder', 'multiscale_context', 'masked'])
def test_experiment_forward_backward_and_batch_isolation(variant):
    config = load_experiment_config('configs/pretrain.yaml')
    config['model'].update(hidden_dim=16, num_layers=2, dropout=0.0)
    if variant == 'masked':
        config['diffusion']['attribute_mask_ratio'] = 1.0
    else:
        config['model'][variant] = True
    dims = feature_dims(config)
    graphs = [synthetic_graph('a', 4, *dims, 8), synthetic_graph('b', 6, *dims, 8)]
    batch = collate_graphs(graphs)
    model = DiffusionPretrainModel(config, *dims)
    outputs = model(batch, batch.face_cont, batch.edge_cont, torch.zeros(10, dtype=torch.long))
    loss, _ = compute_pretrain_loss(outputs, batch, face_noise=torch.zeros_like(batch.face_cont),
                                    edge_noise=torch.zeros_like(batch.edge_cont), config=config)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.face_cont_proj.weight.grad is not None
    if variant == 'masked':
        assert outputs['face_type_mask'].all() and outputs['edge_type_mask'].all()
    encoder = model.encoder.eval()
    def encode(b):
        return encoder(b.face_cont, b.face_surface_type, b.edge_index, b.edge_cont,
                       b.edge_type, b.edge_relation, graph_ptr=b.graph_ptr)[0]
    with torch.no_grad():
        assert torch.allclose(encode(batch)[:4], encode(collate_graphs(graphs[:1])), atol=2e-5)


def test_masked_categories_do_not_affect_encoder():
    config = load_experiment_config('configs/pretrain.yaml')
    config['model'].update(hidden_dim=16, num_layers=2, dropout=0.0)
    dims = feature_dims(config)
    batch = collate_graphs([synthetic_graph('a', 4, *dims, 8)])
    encoder = BRepGraphEncoder.from_config(config, *dims).eval()
    def run(offset):
        return encoder(batch.face_cont, (batch.face_surface_type + offset) % 32,
                       batch.edge_index, batch.edge_cont, (batch.edge_type + offset) % 32,
                       (batch.edge_relation + offset) % 4, graph_ptr=batch.graph_ptr,
                       face_type_mask=torch.ones(4, dtype=torch.bool),
                       edge_type_mask=torch.ones(batch.edge_cont.shape[0], dtype=torch.bool))[0]
    assert torch.allclose(run(0), run(1), atol=2e-5)


def test_geometric_normalization_translation_scale_and_no_mutation():
    from dataclasses import replace
    from brepprediff.data.graph import geometric_normalize_graph_features
    graph = synthetic_graph('a', 4, 711, 63, 8)
    graph.face_cont[:, 0] = torch.rand(4)
    graph.edge_cont[:, 0] = torch.rand(graph.num_edges)
    graph.face_cont[:, 11:].reshape(4, 100, 7)[:, :, 6] = 1
    original = graph.face_cont.clone()
    face, edge = graph.face_cont.clone(), graph.edge_cont.clone()
    factor, shift = 3.0, torch.tensor([2., -4., 1.])
    face[:, 1:4] = face[:, 1:4] * factor + shift
    face[:, 11:].reshape(4, 100, 7)[:, :, :3] = face[:, 11:].reshape(4, 100, 7)[:, :, :3] * factor + shift
    face[:, 0] = torch.log1p(torch.expm1(face[:, 0]) * factor**2)
    face[:, 7:10] /= factor
    face[:, 10] /= factor**2
    edge[:, 0] = torch.log1p(torch.expm1(edge[:, 0]) * factor)
    eg = edge[:, 3:].reshape(-1, 10, 6)
    eg[:, :, :3] = eg[:, :, :3] * factor + shift
    a = geometric_normalize_graph_features(graph, uv_grid_size=10, edge_u_grid_size=10)
    b = geometric_normalize_graph_features(replace(graph, face_cont=face, edge_cont=edge), uv_grid_size=10, edge_u_grid_size=10)
    assert torch.equal(graph.face_cont, original)
    assert torch.allclose(a.face_cont, b.face_cont, atol=2e-5)
    assert torch.allclose(a.edge_cont, b.edge_cont, atol=2e-5)


@pytest.mark.parametrize('refine', [False, True])
def test_boundary_head_supervision_and_inference(refine):
    from brepprediff.models.segmentation import SegmentationModel, compute_segmentation_loss
    config = load_experiment_config('configs/finetune_joint_fusion360seg_mlp.yaml')
    config['model'].update(hidden_dim=16, num_layers=2, dropout=0., boundary_aux=True,
                           boundary_refine=refine)
    dims = feature_dims(config)
    batch = collate_graphs([synthetic_graph('a', 4, *dims, 8)])
    model = SegmentationModel(config, *dims)
    output = model(batch, return_aux=True)
    loss, metrics = compute_segmentation_loss(output, batch, config)
    loss.backward()
    assert torch.isfinite(loss) and metrics['boundary'] > 0
    assert model.boundary_head.net[-1].weight.grad.abs().sum() > 0
    assert model.eval()(batch).shape == (4, 8)


def test_structured_diffusion_complete_graph_training_and_sampling():
    from brepprediff.models.segmentation import DiffusionSegmentationModel
    from brepprediff.models.downstream import prepare_label_diffusion_training_batch, label_diffusion_objective
    config = load_experiment_config('configs/finetune_joint_fusion360seg_diffloss_200.yaml')
    config['model'].update(hidden_dim=16, num_layers=2, dropout=0.)
    config['label_diffusion'].update(structured=True, noise_samples_per_token=1)
    dims = feature_dims(config)
    batch = collate_graphs([synthetic_graph('a', 4, *dims, 8), synthetic_graph('b', 6, *dims, 8)])
    model = DiffusionSegmentationModel(config, *dims)
    prepared = prepare_label_diffusion_training_batch(model, batch, config)
    assert prepared.timesteps[:4].unique().numel() == 1
    assert prepared.timesteps[4:].unique().numel() == 1
    output = model(batch, prepared.x_t, prepared.timesteps, prepared.token_indices)
    loss, _, _ = label_diffusion_objective(output, prepared, model)
    loss.backward()
    assert model.diffusion_head.layers[0].query.weight.grad is not None
    with torch.no_grad():
        result = model.eval().sample(batch, initial_noise=torch.zeros(10, 8), steps=3)
    assert result.shape == (10, 8) and torch.isfinite(result).all()
    batch.labels[0] = -100
    with pytest.raises(ValueError, match='complete labels'):
        prepare_label_diffusion_training_batch(model, batch, config)


def test_rigid_augmentation_preserves_distances_labels_and_masks():
    from brepprediff.data.graph import augment_graph_geometry, geometric_normalize_graph_features
    graph = synthetic_graph('a', 4, 711, 63, 8)
    original = graph.face_cont.clone()
    rotated = augment_graph_geometry(graph, uv_grid_size=10, edge_u_grid_size=10, rotate=True)
    assert torch.equal(graph.face_cont, original)
    assert rotated.labels is graph.labels
    assert torch.allclose(torch.cdist(graph.face_cont[:, 1:4], graph.face_cont[:, 1:4]),
                          torch.cdist(rotated.face_cont[:, 1:4], rotated.face_cont[:, 1:4]), atol=1e-5)
    old_grid = graph.face_cont[:, 11:].reshape(4, 100, 7)
    new_grid = rotated.face_cont[:, 11:].reshape(4, 100, 7)
    assert torch.equal(old_grid[:, :, 6], new_grid[:, :, 6])
    assert torch.allclose(old_grid[:, :, 3:6].norm(dim=-1), new_grid[:, :, 3:6].norm(dim=-1), atol=1e-5)
    reparam = augment_graph_geometry(graph, uv_grid_size=10, edge_u_grid_size=10, reparameterize=True)
    assert torch.equal(reparam.face_cont[:, :11], graph.face_cont[:, :11])
    assert torch.equal(reparam.face_cont[:, 11:].reshape(4, 100, 7).sort(dim=1).values,
                       old_grid.sort(dim=1).values)
    a = geometric_normalize_graph_features(graph, uv_grid_size=10, edge_u_grid_size=10)
    b = geometric_normalize_graph_features(rotated, uv_grid_size=10, edge_u_grid_size=10)
    assert torch.allclose(torch.cdist(a.face_cont[:, 1:4], a.face_cont[:, 1:4]),
                          torch.cdist(b.face_cont[:, 1:4], b.face_cont[:, 1:4]), atol=1e-5)


def test_fixed_rotation_is_deterministic_and_validated():
    from brepprediff.data.graph import augment_graph_geometry
    graph = synthetic_graph('a', 4, 711, 63, 8)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    first = augment_graph_geometry(graph, uv_grid_size=10, edge_u_grid_size=10,
                                   rotation_matrix=rotation)
    second = augment_graph_geometry(graph, uv_grid_size=10, edge_u_grid_size=10,
                                    rotation_matrix=rotation.tolist())
    assert torch.equal(first.face_cont, second.face_cont)
    assert torch.equal(first.edge_cont, second.edge_cont)
    with pytest.raises(ValueError, match='3x3'):
        augment_graph_geometry(graph, uv_grid_size=10, edge_u_grid_size=10,
                               rotation_matrix=torch.eye(4))


def test_rotation_probability_range_is_enforced():
    from brepprediff.data.dataset import StepSegDataset
    config = load_experiment_config('configs/finetune_joint_fusion360seg_mlp.yaml')
    config['train'].update(stage='finetune', rotation_augmentation=True,
                           rotation_augmentation_probability=1.5)
    dataset = StepSegDataset.__new__(StepSegDataset)
    dataset.config = config
    dataset.split = 'train'
    dataset.feature_preprocessing_mode = 'per_graph'
    dataset.global_feature_stats = None
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        dataset._preprocess_graph(synthetic_graph('a', 4, 711, 63, 8))
