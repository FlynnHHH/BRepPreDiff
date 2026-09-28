#!/usr/bin/env python3
"""Joint face/edge sampling input ablation on idle GPU 0."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
TAG = 'face_edge_grid_ablation_pre100_ft200_seed42_20260924'
SUITE = ROOT / 'runs' / TAG
REPORT = ROOT / 'reports' / f'{TAG}.md'
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
VARIANTS = ('face_edge_grid',)
TARGETS = dict(fusion360seg='fusion360seg', blend='brepprediff_seg', tmcad='tmcad_cls')
GPU = 0


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def baseline(stage, suffix):
    paths = list((ROOT / 'runs/inductive9_rotate_mix' / stage).glob(f'*{suffix}'))
    assert len(paths) == 1, paths
    return paths[0]


def prepare():
    for sub in ('configs', 'logs', 'results', 'state'):
        (SUITE / sub).mkdir(parents=True, exist_ok=True)
    if (SUITE / 'manifest.json').exists():
        return json.loads((SUITE / 'manifest.json').read_text())
    jobs = []
    pre = baseline('pretrain', BASE)
    downstream = {target: baseline('finetune', f'_full_{task}_mlp_ft200_seed42_{BASE}') for target, task in TARGETS.items()}
    for variant in VARIANTS:
        for target, source in [('pretrain', pre), *downstream.items()]:
            stage = 'pretrain' if target == 'pretrain' else 'finetune'
            cfg = yaml.safe_load((source / 'config.yaml').read_text())
            original = copy.deepcopy(cfg)
            name = f'no_{variant}_{target}'
            cfg['run'].update(name=name, output_dir=str(SUITE), show_progress=False)
            cfg['model']['input_ablation'] = variant
            cfg['train']['resume'] = None
            assert cfg['seed'] == 42 and cfg['train']['epochs'] == (100 if stage == 'pretrain' else 200)
            if stage == 'pretrain':
                assert len(cfg['data']['sources']) == 9
                assert cfg['data']['strip_labels']
                assert all(s['splits']['train'] == ['train'] for s in cfg['data']['sources'])
            else:
                cfg['train'].update(pretrain_checkpoint=None, encoder_freeze_mode='none')
                assert cfg['model']['finetune_head'] == 'mlp'
                if target == 'tmcad': cfg['model']['graph_pooling'] = 'mean_max'
            # Check all differences against the explicitly permitted protocol changes.
            check = copy.deepcopy(cfg)
            check['run'] = original['run']
            check['model'].pop('input_ablation')
            for key in ('resume', 'pretrain_checkpoint', 'encoder_freeze_mode'):
                if key in original['train']: check['train'][key] = original['train'][key]
                else: check['train'].pop(key, None)
            if target == 'tmcad':
                if 'graph_pooling' in original['model']: check['model']['graph_pooling'] = original['model']['graph_pooling']
                else: check['model'].pop('graph_pooling', None)
            assert check == original, name
            path = SUITE / 'configs' / f'{name}.yaml'
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            jobs.append(dict(name=name, variant=variant, target=target, stage=stage, config=str(path), baseline=str(source)))
    shutil.copytree(ROOT / 'src/brepprediff', SUITE / 'source/brepprediff', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    paths = list((SUITE / 'source').rglob('*.py')) + list((SUITE / 'configs').glob('*.yaml'))
    paths += [Path(__file__).resolve()] + list((ROOT / 'data').rglob('*.yaml')) + list((ROOT / 'data/splits').glob('*.txt'))
    paths += [Path(j['baseline']) / 'config.yaml' for j in jobs]
    dump(SUITE / 'provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    dump(SUITE / 'manifest.json', jobs)
    return jobs


def report(jobs):
    lines = ['# 输入消融：全语料预训练 → Fusion360Seg / Blend / TMCAD', '',
             'GPU 0 串行；Inductive9 全部九源 train 划分，去标签；预训练 100 epochs，MLP 全量微调 200 epochs，seed 42，50% SO(3)，TMCAD mean_max。其余协议复制完整输入基线。',
             '单组消融：预训练、微调、测试同时屏蔽面 UV-grid 和边 U-grid 输入。',
             'no_face_edge_grid：face_cont[:, 11:] 的 700 维面采样与 edge_cont[:, 3:] 的 60 维边采样均置零。',
             '保留 11 维面基础几何、3 维边基础几何、面类型、边类型/关系与拓扑。旧 uv_grid 配置保持仅屏蔽面采样的原含义，以便复现旧结果。',
             '屏蔽发生在编码器入口（归一化、旋转、扩散加噪之后），保留原参数形状和初始化；被屏蔽通道的预训练重建/噪声/类别监督仍保留。这是输入信息消融。',
             '使用各组预训练 last.pt；下游按 validation accuracy 选 best.pt 后测试。完整输入对照复用现有基线，单 seed。', '',
             '| 变体 | 任务 | 状态 | Acc % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    rows = []
    for target, task in TARGETS.items():
        m = json.loads((baseline('finetune', f'_full_{task}_mlp_ft200_seed42_{BASE}') / 'test_metrics.json').read_text())['metrics']
        lines.append(f"| full_input | {target} | 复用基线 | {100*m['accuracy']:.4f} | 0 | {100*m['macro_f1']:.4f} | {100*m['macro_iou']:.4f} | 0 |")
    for job in jobs:
        state = SUITE / 'state' / f"{job['name']}.json"
        status = json.loads(state.read_text())['status'] if state.exists() else 'pending'
        values = ['—'] * 5
        result = SUITE / 'results' / f"{job['name']}.json"
        if result.exists():
            new = json.loads(result.read_text())
            old = json.loads((Path(job['baseline']) / 'test_metrics.json').read_text())
            assert new['split'] == old['split'] == 'test' and new['samples'] == old['samples']
            m, b = new['metrics'], old['metrics']
            values = [f"{100*m['accuracy']:.4f}", f"{100*(m['accuracy']-b['accuracy']):+.4f}", f"{100*m['macro_f1']:.4f}", f"{100*m['macro_iou']:.4f}", f"{100*(m['macro_iou']-b['macro_iou']):+.4f}"]
            rows.append(dict(variant=job['variant'], target=job['target'], accuracy=m['accuracy'], macro_f1=m['macro_f1'], macro_iou=m['macro_iou'], delta_acc_pp=100*(m['accuracy']-b['accuracy']), samples=new['samples'], best_epoch=new['checkpoint_epoch']))
        lines.append('| ' + ' | '.join([job['variant'], job['target'], status, *values]) + ' |')
    lines += ['', f'配置、源码快照、校验哈希、日志和状态：`{SUITE}`。']
    tmp = REPORT.with_suffix('.tmp'); tmp.write_text('\n'.join(lines) + '\n'); tmp.replace(REPORT)
    if rows:
        with (SUITE / 'results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(job, jobs):
    state = SUITE / 'state' / f"{job['name']}.json"
    if state.exists() and json.loads(state.read_text())['status'] == 'completed': return
    def status(value, **extra):
        dump(state, dict(job, status=value, gpu=GPU, **extra)); report(jobs)
    status('waiting_for_gpu')
    locks = ROOT / 'runs/rotate_mix_gpu_locks'; locks.mkdir(exist_ok=True)
    with (locks / f'gpu{GPU}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            query = subprocess.run(['nvidia-smi', '-i', str(GPU), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
            if int(query.stdout.strip()) >= 80000: break
            time.sleep(20)
        cfg = yaml.safe_load(Path(job['config']).read_text())
        overrides = []
        if job['stage'] == 'finetune':
            ckpt = (SUITE / f"{job['variant']}_pretrain_checkpoint").read_text().strip()
            assert Path(ckpt).is_file()
            overrides += ['--override', f'train.pretrain_checkpoint={ckpt}']
        previous = sorted((SUITE / job['stage']).glob(f"*_{job['name']}"))
        if previous and (previous[-1] / 'checkpoints/last.pt').exists():
            overrides += ['--override', f'train.resume={previous[-1]}/checkpoints/last.pt']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), PYTHONPATH=str(SUITE / 'source'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled', OMP_NUM_THREADS='4', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        status('running', launcher_pid=os.getpid())
        with (SUITE / 'logs' / f"{job['name']}.log").open('a') as log:
            subprocess.run([sys.executable, '-m', f"brepprediff.training.{job['stage']}", '--config', job['config'], *overrides], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = sorted((SUITE / job['stage']).glob(f"*_{job['name']}"))[-1]
            if job['stage'] == 'pretrain':
                ckpt = directory / 'checkpoints/last.pt'
                assert ckpt.is_file()
                (SUITE / f"{job['variant']}_pretrain_checkpoint").write_text(str(ckpt))
            else:
                status('testing', run=str(directory))
                subprocess.run([sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint', str(directory / 'checkpoints/best.pt'), '--split', 'test', '--output', str(SUITE / 'results' / f"{job['name']}.json"), '--batch-size', str(cfg['train']['batch_size']), '--num-workers', str(cfg['train']['num_workers']), '--device', 'cuda'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status('completed', run=str(directory))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    os.chdir(ROOT); SUITE.mkdir(parents=True, exist_ok=True)
    lock = (SUITE / 'launcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = prepare()
    for path, digest in json.loads((SUITE / 'provenance.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    report(jobs)
    if args.prepare_only:
        print(f'Validated {len(jobs)} jobs: {SUITE}'); return
    (SUITE / 'launcher.pid').write_text(str(os.getpid()))
    for job in jobs:
        try: run(job, jobs)
        except Exception as exc:
            dump(SUITE / 'state' / f"{job['name']}.json", dict(job, status='failed', error=str(exc))); report(jobs); raise
    print(f'Completed: {REPORT}', flush=True)


if __name__ == '__main__': main()
