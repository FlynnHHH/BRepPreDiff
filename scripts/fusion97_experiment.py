"""Reproducible Fusion360Seg-only 50/100 experiments; validation-only search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/nvme03/hhfeng/fusion360segmentationdataset/s2.0.0')
VARIANTS = {
    'baseline': [],
    'masked': ['diffusion.attribute_mask_ratio=0.5'],
    'wide': ['model.hidden_dim=256'],
    'grid': ['model.grid_encoder=true'],
    'context': ['model.multiscale_context=true'],
    'geometric': ['train.feature_preprocessing.mode=geometric'],
    'geometric_grid': ['train.feature_preprocessing.mode=geometric', 'model.grid_encoder=true'],
    'boundary': ['model.boundary_aux=true'],
    'boundary_refine': ['model.boundary_aux=true', 'model.boundary_refine=true'],
    'deep': ['model.num_layers=8'],
    'graph_diffusion': ['label_diffusion.structured=true', 'label_diffusion.noise_samples_per_token=1',
                        'label_diffusion.sampling_steps=5'],
    'rotation': ['train.rotation_augmentation=true'],
    'uv_augment': ['train.uv_augmentation=true'],
    'localedge': ['data.cache_dir=cache/features/fusion360seg_localedge_v1', 'brep.local_edge_normals=true'],
    'cosine': ['train.lr_scheduler=cosine', 'train.min_lr=1e-6'],
    'context_cosine': ['model.multiscale_context=true', 'train.lr_scheduler=cosine', 'train.min_lr=1e-6'],
    'context_rotation': ['model.multiscale_context=true', 'train.rotation_augmentation=true'],
    'context_rotation_seed43': ['model.multiscale_context=true', 'train.rotation_augmentation=true',
                                'seed=43', 'train.dataloader_seed=43'],
    'context_rotation_mix': ['model.multiscale_context=true', 'train.rotation_augmentation=true',
                             'train.rotation_augmentation_probability=0.5'],
    'context_rotation_operation': ['model.multiscale_context=true', 'train.rotation_augmentation=true',
                                   'model.operation_class_map=[0,0,1,1,2,3,4,4]',
                                   'train.operation_loss_weight=0.2'],
    'context_ce': ['model.multiscale_context=true', 'train.dice_loss_weight=0.0'],
    'context_operation': ['model.multiscale_context=true',
                          'model.operation_class_map=[0,0,1,1,2,3,4,4]', 'train.operation_loss_weight=0.2'],
    'wide_context': ['model.multiscale_context=true', 'model.hidden_dim=256'],
}


def audit():
    splits = {}
    for split in ('train', 'val', 'test'):
        items = (ROOT / f'data/splits/fusion360seg_{split}.txt').read_text().splitlines()
        ids = [Path(x.strip()).stem for x in items if x.strip()]
        assert len(ids) == len(set(ids)), f'duplicates in {split}'
        splits[split] = set(ids)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not splits[a] & splits[b], f'overlap {a}/{b}'
    official = json.loads((RAW / 'train_test.json').read_text())
    assert splits['test'] <= set(official['test']), 'non-official test samples'
    assert splits['train'] | splits['val'] <= set(official['train']), 'train/val outside official train'
    cache_ids = {p.stem.rsplit('_', 1)[0] for p in (ROOT / 'cache/features/fusion360seg').glob('*.npz')}
    for name, ids in splits.items():
        assert ids <= cache_ids, f'missing {name} caches: {len(ids - cache_ids)}'
    return {k: len(v) for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--gpu', type=int, choices=(4,), default=4,
                        help='Current campaign is restricted to GPU 4 by user instruction.')
    parser.add_argument('--root', default='runs/fusion97/round1')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--after', action='append', default=[], help='Wait for completed experiment directories.')
    parser.add_argument('--reuse-pretrain-from', default=None, help='Reuse a completed baseline experiment encoder.')
    args = parser.parse_args()
    counts = audit()
    print(json.dumps(counts), flush=True)
    if args.audit_only:
        return
    directory = (ROOT / args.root / args.variant).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'result.json').exists():
        raise RuntimeError('Experiment already completed; use a new root.')
    snapshot = directory / 'source'
    if snapshot.exists():
        raise RuntimeError('Existing source snapshot; inspect interrupted experiment before resuming.')
    shutil.copytree(ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__, snapshot / Path(__file__).name)
    source_hashes = {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(snapshot.rglob('*.py'))}
    (directory / 'manifest.json').write_text(json.dumps({
        'variant': args.variant, 'gpu': args.gpu, 'splits': counts,
        'source_hashes': source_hashes, 'overrides': VARIANTS[args.variant],
        'pretrain_epochs': 50, 'finetune_epochs': 100,
        'selection': 'validation accuracy', 'test_accessed': False,
        'split_sha256': {split: hashlib.sha256((ROOT / f'data/splits/fusion360seg_{split}.txt').read_bytes()).hexdigest()
                         for split in ('train', 'val', 'test')},
    }, indent=2))
    for dependency in args.after:
        result = ROOT / dependency / 'result.json'
        print(f'Waiting for {result}', flush=True)
        while not result.exists():
            time.sleep(10)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(snapshot / 'src'),
               WANDB_MODE='disabled', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    common = [f'run.output_dir={directory}', f'run.name={args.variant}',
              'run.show_progress=false', 'run.save_every_epochs=50', 'wandb.enabled=false',
              'model.encoder_type=edge_update_attention', 'model.hidden_dim=128',
              'model.num_layers=4', 'model.num_heads=4', 'seed=42',
              'train.num_workers=4', 'train.dataloader_seed=42',
              f'data.steps_dir={RAW / "breps/step"}', f'data.segs_dir={RAW / "breps/seg"}',
              'data.prepare_on_start=false', 'train.resume=null'] + VARIANTS[args.variant]

    def run(stage, config, extra):
        command = [sys.executable, '-m', f'brepprediff.training.{stage}', '--config', str(ROOT / config),
                   '--data-config', str(ROOT / 'data/fusion360seg.yaml')]
        for value in common + extra:
            command += ['--override', value]
        (directory / f'{stage}_command.json').write_text(json.dumps(command, indent=2))
        print(f'{time.ctime()} starting {stage} {args.variant} GPU {args.gpu}', flush=True)
        with (directory / f'{stage}.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        runs = list((directory / stage).iterdir())
        assert len(runs) == 1, runs
        return runs[0]

    if args.reuse_pretrain_from:
        if args.variant not in {'boundary', 'boundary_refine', 'graph_diffusion', 'rotation', 'uv_augment', 'cosine',
                               'context_cosine', 'context_rotation', 'context_rotation_seed43',
                               'context_rotation_mix',
                               'context_rotation_operation',
                               'context_ce', 'context_operation'}:
            raise ValueError('Pretraining reuse is restricted to downstream-only changes.')
        import torch
        source_result = json.loads((ROOT / args.reuse_pretrain_from / 'result.json').read_text())
        expected_source = 'context' if args.variant.startswith('context_') else 'baseline'
        if source_result['variant'] != expected_source:
            raise ValueError(f'Expected {expected_source} encoder.')
        pre = Path(source_result['pretrain'])
        checkpoint = pre / 'checkpoints/last.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = payload['config']
        assert payload['epoch'] == 50 and cfg['train']['epochs'] == 50
        assert cfg['data']['train_split'] == 'data/splits/fusion360seg_train.txt'
        assert cfg['data'].get('val_split') is None and cfg['data'].get('test_split') is None
        assert cfg['data']['strip_labels'] and not cfg['data'].get('sources')
        (directory / 'pretrain_reuse.json').write_text(json.dumps({
            'checkpoint': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'source_result': str(ROOT / args.reuse_pretrain_from / 'result.json'),
        }, indent=2))
        del payload
    else:
        pre = run('pretrain', 'configs/pretrain.yaml', [
            'data.strip_labels=true', 'data.val_split=null', 'data.test_split=null',
            'train.epochs=50', 'train.batch_size=128', 'train.lr=1e-4', 'train.lr_scheduler=constant'])
    ft_config = ('configs/finetune_joint_fusion360seg_diffloss_200.yaml'
                 if args.variant == 'graph_diffusion' else 'configs/finetune_joint_fusion360seg_mlp.yaml')
    ft = run('finetune', ft_config, [
        'data.strip_labels=false', 'train.epochs=100', 'train.batch_size=256',
        'train.lr=3e-4', 'train.selection_metric=acc',
        f'train.pretrain_checkpoint={pre / "checkpoints/last.pt"}'])
    output = directory / 'validation.json'
    command = [sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint',
               str(ft / 'checkpoints/best.pt'), '--split', 'val', '--output', str(output), '--num-workers', '4']
    with (directory / 'evaluate.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    (directory / 'result.json').write_text(json.dumps({
        'variant': args.variant, 'pretrain': str(pre), 'finetune': str(ft),
        'validation': json.loads(output.read_text()), 'test_accessed': False,
    }, indent=2))
    print(f'{time.ctime()} finished {args.variant}', flush=True)


if __name__ == '__main__':
    main()
