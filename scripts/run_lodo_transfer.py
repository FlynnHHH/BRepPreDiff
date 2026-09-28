#!/usr/bin/env python3
"""Target-excluded pretraining, full fine-tuning and strict linear probes on GPUs 1/2."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
SUITE = ROOT / 'runs/lodo_transfer_pre100_ft200_seed42_20260918'
TARGETS = {'fusion360seg': ('fusion360seg', 'fusion360seg_s2_0_1'),
           'blend': ('brepprediff_seg', 'brepprediff'), 'tmcad': ('tmcad_cls', 'tmcad')}
KEYS = ('face_cont', 'face_surface_type', 'edge_index', 'edge_cont', 'edge_type', 'edge_relation')


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def saved(stage, suffix):
    paths = list((ROOT / 'runs/inductive9_rotate_mix' / stage).glob(f'*{suffix}'))
    assert len(paths) == 1, paths
    return yaml.safe_load((paths[0] / 'config.yaml').read_text())


def fingerprint(path):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for key in KEYS:
            a = np.ascontiguousarray(arrays[key])
            h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.dtype.str.encode()); h.update(a.tobytes())
    return str(path), h.hexdigest()


def prepare_probes():
    for sub in ('configs', 'splits', 'logs', 'results', 'state'):
        (SUITE / sub).mkdir(parents=True, exist_ok=True)
    if (SUITE / 'probe_manifest.json').exists():
        return
    jobs = []
    for target, (task, _) in TARGETS.items():
        cfg = saved('finetune', f'_full_{task}_mlp_ft200_seed42_{BASE}')
        checkpoint = cfg['train']['pretrain_checkpoint']
        assert Path(checkpoint).is_file()
        name = f'{target}_probe'
        cfg['run'].update(name=name, output_dir=str(SUITE), show_progress=False)
        cfg['train'].update(epochs=200, resume=None, encoder_freeze_mode='all')
        cfg['model']['finetune_head'] = 'linear'
        if target == 'tmcad': cfg['model']['graph_pooling'] = 'mean_max'
        cfg['wandb'] = {'enabled': False, 'project': 'brepprediff'}
        path = SUITE / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(dict(target=target, mode='probe', name=name, config=str(path), pretrain_checkpoint=checkpoint))
    shutil.copytree(ROOT / 'src/brepprediff', SUITE / 'source/brepprediff', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    dependencies = list((SUITE / 'source').rglob('*.py')) + [Path(j['config']) for j in jobs]
    dependencies += [Path(__file__).resolve()] + list((ROOT / 'data').glob('*.yaml')) + list((ROOT / 'data/splits').glob('*.txt'))
    dependencies += sorted({Path(j['pretrain_checkpoint']) for j in jobs})
    dump(SUITE / 'probe_provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies})
    dump(SUITE / 'probe_manifest.json', jobs)


def prepare():
    from brepprediff.data.dataset import StepSegDataset, _runtime_source_config, _read_split
    for sub in ('configs', 'splits', 'logs', 'results', 'state'):
        (SUITE / sub).mkdir(parents=True, exist_ok=True)
    if (SUITE / 'manifest.json').exists():
        return
    prepare_probes()
    pre = saved('pretrain', BASE)
    sources, source_items = {}, {}
    for src in pre['data']['sources']:
        cfg = _runtime_source_config(pre, src)
        samples = StepSegDataset(cfg, 'train').samples
        items = _read_split(Path(cfg['data']['train_split']))
        assert len(samples) == len(items)
        sources[src['name']] = (src, samples)
        source_items[src['name']] = {str(sample.cache_path): item for sample, item in zip(samples, items)}
    fine = {target: saved('finetune', f'_full_{task}_mlp_ft200_seed42_{BASE}')
            for target, (task, _) in TARGETS.items()}
    heldout = {target: [sample for split in ('train', 'val', 'test')
                       for sample in StepSegDataset(cfg, split).samples] for target, cfg in fine.items()}
    paths = sorted({sample.cache_path.resolve() for _, samples in sources.values() for sample in samples}
                   | {sample.cache_path.resolve() for samples in heldout.values() for sample in samples})
    print(f'Hashing {len(paths)} geometry caches (labels excluded)', flush=True)
    hashes = {}
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i, (path, digest) in enumerate(pool.map(fingerprint, paths, chunksize=128), 1):
            hashes[path] = digest
            if i % 20000 == 0: print(f'Hashed {i}/{len(paths)}', flush=True)
    dump(SUITE / 'geometry_hashes.json', hashes)
    jobs, audit = [], {}
    def write_job(target, mode, cfg):
        cfg = copy.deepcopy(cfg)
        name = f'{target}_{mode}'
        cfg['run'].update(name=name, output_dir=str(SUITE), show_progress=False)
        cfg['train']['resume'] = None
        cfg['wandb'] = {'enabled': False, 'project': 'brepprediff'}
        path = SUITE / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(dict(target=target, mode=mode, name=name, config=str(path)))
    for target, (_, excluded) in TARGETS.items():
        forbidden = {hashes[str(s.cache_path.resolve())] for s in heldout[target]}
        cfg = copy.deepcopy(pre)
        cfg['data']['sources'] = []
        rows = []
        for name, (src, samples) in sources.items():
            kept = [s for s in samples if name != excluded and hashes[str(s.cache_path.resolve())] not in forbidden]
            split = SUITE / 'splits' / f'{target}_without_target_{name}.txt'
            split.write_text(''.join(source_items[name][str(s.cache_path)] + '\n' for s in kept))
            rows.append(dict(source=name, before=len(samples), kept=len(kept), removed=len(samples)-len(kept)))
            if kept:
                entry = copy.deepcopy(src)
                entry['data_config'] = str((ROOT / 'data' / src['data_config']).resolve())
                entry.setdefault('data', {})['train_split'] = str(split)
                cfg['data']['sources'].append(entry)
                assert not ({hashes[str(s.cache_path.resolve())] for s in kept} & forbidden)
        cfg['train'].pop('pretrain_checkpoint', None)
        assert cfg['train']['epochs'] == 100
        audit[target] = dict(excluded_source=excluded, target_samples_all_splits=len(heldout[target]),
                             target_unique_geometry=len(forbidden), remaining_exact_overlap=0, sources=rows)
        write_job(target, 'pretrain', cfg)
        for mode in ('finetune',):
            cfg = copy.deepcopy(fine[target])
            cfg['train'].update(epochs=200, pretrain_checkpoint=None,
                                encoder_freeze_mode='none' if mode == 'finetune' else 'all')
            cfg['model']['finetune_head'] = 'mlp' if mode == 'finetune' else 'linear'
            if target == 'tmcad': cfg['model']['graph_pooling'] = 'mean_max'
            write_job(target, mode, cfg)
    dump(SUITE / 'exclusion_audit.json', audit)
    dependencies = list((SUITE / 'source').rglob('*.py')) + list((SUITE / 'configs').glob('*.yaml')) + list((SUITE / 'splits').glob('*.txt'))
    dependencies += [Path(__file__).resolve()] + list((ROOT / 'data').glob('*.yaml')) + list((ROOT / 'data/splits').glob('*.txt'))
    dump(SUITE / 'provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies})
    dump(SUITE / 'manifest.json', jobs)
    print(json.dumps(audit, indent=2), flush=True)


def execute(job, gpu):
    state = SUITE / 'state' / f"{job['name']}.json"
    if state.exists() and json.loads(state.read_text()).get('status') == 'completed': return
    cfg = yaml.safe_load(Path(job['config']).read_text())
    stage = 'pretrain' if job['mode'] == 'pretrain' else 'finetune'
    overrides = []
    if job['mode'] == 'finetune':
        marker = SUITE / f"{job['target']}_pretrain_checkpoint"
        dump(state, dict(job, status='waiting_for_pretrain', gpu=gpu))
        while not marker.exists():
            upstream = SUITE / 'state' / f"{job['target']}_pretrain.json"
            if upstream.exists() and json.loads(upstream.read_text()).get('status') == 'failed':
                raise RuntimeError(f'Upstream pretraining failed: {upstream}')
            time.sleep(20)
        checkpoint = marker.read_text().strip()
        assert Path(checkpoint).is_file()
        overrides += ['--override', f'train.pretrain_checkpoint={checkpoint}']
    previous = sorted((SUITE / stage).glob(f"*_{job['name']}"))
    if previous and (previous[-1] / 'checkpoints/last.pt').exists():
        overrides += ['--override', f'train.resume={previous[-1]}/checkpoints/last.pt']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SUITE / 'source'),
               PYTHONUNBUFFERED='1', WANDB_MODE='disabled', OMP_NUM_THREADS='4')
    lockdir = ROOT / 'runs/rotate_mix_gpu_locks'; lockdir.mkdir(exist_ok=True)
    dump(state, dict(job, status='waiting_for_gpu', gpu=gpu))
    with (lockdir / f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            p = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            if int(p.stdout.strip()) >= 50000: break
            time.sleep(20)
        dump(state, dict(job, status='running', gpu=gpu, pid=os.getpid()))
        with (SUITE / 'logs' / f"{job['name']}.log").open('a') as log:
            subprocess.run([sys.executable, '-m', f'brepprediff.training.{stage}', '--config', job['config'], *overrides], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            run = sorted((SUITE / stage).glob(f"*_{job['name']}"))[-1]
            if stage == 'pretrain':
                checkpoint = run / 'checkpoints/last.pt'
                assert checkpoint.exists()
                (SUITE / f"{job['target']}_pretrain_checkpoint").write_text(str(checkpoint))
            else:
                subprocess.run([sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint', str(run / 'checkpoints/best.pt'), '--split', 'test', '--output', str(SUITE / 'results' / f"{job['name']}.json"), '--batch-size', str(cfg['train']['batch_size']), '--num-workers', '8', '--device', 'cuda'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    dump(state, dict(job, status='completed', gpu=gpu, run=str(run)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--prepare-probes', action='store_true')
    p.add_argument('--worker', choices=['finetune', 'probe'])
    args = p.parse_args()
    os.chdir(ROOT)
    if args.prepare_probes:
        prepare_probes(); return
    if args.prepare_only:
        prepare(); return
    if not args.worker: p.error('use --prepare-only or --worker')
    gpu = 1 if args.worker == 'finetune' else 2
    lock = (SUITE / f'worker_{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    provenance = 'provenance.json' if gpu == 1 else 'probe_provenance.json'
    for path, digest in json.loads((SUITE / provenance).read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    (SUITE / f'worker_{gpu}.pid').write_text(str(os.getpid()))
    jobs = json.loads((SUITE / ('manifest.json' if gpu == 1 else 'probe_manifest.json')).read_text())
    modes = ('pretrain', 'finetune') if gpu == 1 else ('probe',)
    for target in TARGETS:
        for mode in modes:
            job = next(j for j in jobs if j['target'] == target and j['mode'] == mode)
            try:
                execute(job, gpu)
            except Exception as exc:
                dump(SUITE / 'state' / f"{job['name']}.json", dict(job, status='failed', error=str(exc)))
                raise


if __name__ == '__main__': main()
