#!/usr/bin/env python3
"""Whole-encoder message-passing ablation, exact saved protocol, GPUs 0 and 1."""
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
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
TAG = 'inductive9_message_passing_pre100_ft200_seed42_20260916'
PRETRAIN_GPU = 0
MIN_FREE_MIB = 50000
SUITE = ROOT / 'runs' / TAG
REPORT = ROOT / 'reports' / f'{TAG}.md'
TASKS = ['brepprediff_seg', 'fusion360seg', 'mfcadpp_seg', 'tmcad_cls', 'solidletters_cls', 'cadsynth_seg', 'mfinstseg_seg']

def dump(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        out.update(flatten(v, key + '.') if isinstance(v, dict) else {key: v})
    return out

def prepare():
    for sub in ['configs', 'logs', 'results']:
        (SUITE / sub).mkdir(parents=True, exist_ok=True)
    jobs = []
    sources = [('pretrain', 'pretrain', next((ROOT / 'runs/inductive9_rotate_mix/pretrain').glob(f'*_{BASE}')))]
    for task in TASKS:
        for head in ['mlp']:
            matches = list((ROOT / 'runs/inductive9_rotate_mix/finetune').glob(f'*_full_{task}_{head}_ft200_seed42_{BASE}'))
            assert len(matches) == 1
            sources.append(('finetune', f'{task}_{head}', matches[0]))
    for stage, name, source in sources:
        original = yaml.safe_load((source / 'config.yaml').read_text())
        cfg = copy.deepcopy(original)
        cfg['run'].update(name=f'{name}_{TAG}', output_dir=str(SUITE))
        assert cfg['model']['encoder_type'] == 'edge_update_attention'
        cfg['model']['encoder_type'] = 'message_passing'
        assert cfg['train']['resume'] is None
        assert cfg['seed'] == 42 and cfg['train']['rotation_augmentation_probability'] == 0.5
        assert cfg['train']['epochs'] == (100 if stage == 'pretrain' else 200)
        if stage == 'finetune':
            cfg['train']['pretrain_checkpoint'] = '__NEW_PRETRAIN_CHECKPOINT__'
        a, b = flatten(original), flatten(cfg)
        changes = {k: {'before': a.get(k), 'after': b.get(k)} for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        allowed = {'run.name', 'run.output_dir', 'train.pretrain_checkpoint', 'model.encoder_type'}
        assert changes.keys() <= allowed, changes
        path = SUITE / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(dict(name=name, stage=stage, config=str(path), baseline=str(source), changes=changes, status='pending'))
    dump(SUITE / 'manifest.json', jobs)
    return jobs

def report(jobs):
    rows = []
    lines = ['# Rotate-mix：Edge Update Attention → 现有 message_passing（ffn）消融', '', f'基线：`reports/{BASE}.md`。', '',
        '仅将 encoder_type 改为现有 message_passing（ffn 别名）：使用原 GraphMessageLayer 的 MLP([src, edge]) 消息和邻居均值聚合、原 post-norm 和 SiLU FFN，edge embedding 在各层不更新。这是整个 encoder 层类型替换，不是只移除 attention。保留离散输入和离散监督；num_heads 保留原配置但本 encoder 不使用。',
        '从基线实际 config.yaml 复制配置：预训练 100 epochs、微调 200 epochs、seed 42、50% SO(3)、原 batch/梯度累积、数据划分、Mean+Max 分类池化和验证选优规则。',
        '预训练使用 GPU 0 单卡保持 batch 128；随后 GPU 0、1 每卡一路微调，只测试 MLP head，共七个下游任务。复用完整离散属性基线，差值为 message_passing encoder 减去 attention encoder（百分点）。这是算子替换消融，不保证参数量相等；单种子不代表统计显著性。', '',
        f'实验目录：`{SUITE}`；配置差异见 `manifest.json`，状态见 `status.json`。', '',
        '| Task / Head | 状态 | Accuracy % | ΔAcc | Macro-F1 % | ΔF1 | Weighted-F1 % | ΔWF1 | mIoU % | ΔmIoU |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for job in jobs:
        result = SUITE / 'results' / f"{job['name']}.json"
        row = [job['name'], job['status']]
        if result.exists():
            new = json.loads(result.read_text())
            old = json.loads((Path(job['baseline']) / 'test_metrics.json').read_text())
            assert new['samples'] == old['samples']
            assert new['split'] == old['split'] == 'test'
            record = dict(name=job['name'], samples=new['samples'],
                          best_epoch=new['checkpoint_epoch'], result=str(result),
                          baseline=str(Path(job['baseline']) / 'test_metrics.json'))
            for metric in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_iou']:
                n, o = new['metrics'][metric], old['metrics'][metric]
                row.extend([f'{100*n:.4f}', f'{100*(n-o):+.4f}'])
                record.update({metric: n, f'baseline_{metric}': o, f'delta_pp_{metric}': 100*(n-o)})
            rows.append(record)
        else:
            row += ['—'] * 8
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', f'下游测试完成：{len(rows)}/7。原始指标与基线见实验目录 results.csv（指标为 0–1，差值为百分点）。']
    if rows:
        with (SUITE / 'results.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    REPORT.write_text('\n'.join(lines) + '\n')
    dump(SUITE / 'status.json', jobs)

def free_memory(gpu):
    p = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    return int(p.stdout.strip())

def run_job(job, gpu):
    lock_dir = ROOT / 'runs' / 'rotate_mix_gpu_locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        while free_memory(gpu) < MIN_FREE_MIB:
            time.sleep(30)
        run_job_locked(job, gpu)


def run_job_locked(job, gpu):
    cfg = yaml.safe_load(Path(job['config']).read_text())
    args = []
    previous = sorted((SUITE / job['stage']).glob(f"*_{cfg['run']['name']}"))
    if previous:
        checkpoints = previous[-1] / 'checkpoints'
        resume = checkpoints / 'last.pt'
        if not resume.exists():
            candidates = sorted(checkpoints.glob('epoch_*.pt'))
            resume = candidates[-1] if candidates else None
        if resume is not None and resume.exists():
            args = ['--override', f'train.resume={resume}']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(SUITE / 'source'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (SUITE / 'logs' / f"{job['name']}.log").open('a') as log:
        subprocess.run([sys.executable, '-m', f"brepprediff.training.{job['stage']}", '--config', job['config'], *args], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs = sorted((SUITE / job['stage']).glob(f"*_{cfg['run']['name']}"))
        run = runs[-1]
        if job['stage'] == 'pretrain':
            (SUITE / 'pretrain_checkpoint').write_text(str(run / 'checkpoints/last.pt'))
        else:
            subprocess.run([sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint', str(run / 'checkpoints/best.pt'), '--split', 'test', '--output', str(SUITE / 'results' / f"{job['name']}.json"), '--batch-size', str(cfg['train']['batch_size']), '--num-workers', str(cfg['train']['num_workers']), '--device', 'cuda'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--worker')
    p.add_argument('--gpu', type=int)
    args = p.parse_args()
    os.chdir(ROOT)
    if args.worker:
        if args.gpu not in (0, 1):
            raise ValueError('This experiment is restricted to physical GPUs 0 and 1')
        jobs = json.loads((SUITE / 'manifest.json').read_text())
        run_job(next(j for j in jobs if j['name'] == args.worker), args.gpu)
        return
    SUITE.mkdir(parents=True, exist_ok=True)
    lock = (SUITE / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = prepare() if not (SUITE / 'manifest.json').exists() else json.loads((SUITE / 'manifest.json').read_text())
    provenance_path = SUITE / 'provenance.json'
    if not provenance_path.exists():
        shutil.copytree(ROOT / 'src/brepprediff', SUITE / 'source/brepprediff',
                        ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
        paths = list((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve()]
        paths += list((ROOT / 'data').rglob('*.yaml')) + list((ROOT / 'data/splits').glob('*.txt'))
        paths += [Path(j['baseline']) / 'config.yaml' for j in jobs]
        paths += [Path(j['baseline']) / 'test_metrics.json' for j in jobs[1:]]
        hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        dump(provenance_path, dict(python=sys.executable, hashes=hashes))
    else:
        for name, expected in json.loads(provenance_path.read_text())['hashes'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Experiment source changed since preparation: {name}')
    report(jobs)
    if args.prepare_only:
        print(f'Validated {len(jobs)} configs; {SUITE}', flush=True)
        return
    (SUITE / 'launcher.pid').write_text(str(os.getpid()))
    pre = jobs[0]
    if not (SUITE / 'pretrain_checkpoint').exists():
        while free_memory(PRETRAIN_GPU) < MIN_FREE_MIB:
            time.sleep(30)
        pre.update(status='running', gpu=PRETRAIN_GPU)
        report(jobs)
        try:
            run_job(pre, PRETRAIN_GPU)
        except Exception:
            pre['status'] = 'failed'
            report(jobs)
            raise
    pre['status'] = 'completed'
    checkpoint = (SUITE / 'pretrain_checkpoint').read_text().strip()
    assert Path(checkpoint).is_file()
    for job in jobs[1:]:
        path = Path(job['config'])
        cfg = yaml.safe_load(path.read_text())
        cfg['train']['pretrain_checkpoint'] = checkpoint
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        job['changes']['train.pretrain_checkpoint']['after'] = checkpoint
        job['status'] = 'completed' if (SUITE / 'results' / f"{job['name']}.json").exists() else 'pending'
    dump(SUITE / 'manifest.json', jobs)
    pending = [j for j in jobs[1:] if j['status'] == 'pending']
    active = {}
    while pending or active:
        for gpu, (process, job) in list(active.items()):
            code = process.poll()
            if code is not None:
                job.update(status='completed' if code == 0 else 'failed', returncode=code)
                del active[gpu]
        for gpu in (0, 1):
            if pending and gpu not in active and free_memory(gpu) >= MIN_FREE_MIB:
                job = pending.pop(0)
                process = subprocess.Popen([sys.executable, __file__, '--worker', job['name'], '--gpu', str(gpu)])
                job.update(status='running', gpu=gpu, pid=process.pid)
                active[gpu] = process, job
        report(jobs)
        if pending or active:
            time.sleep(30)
    report(jobs)
    if any(j['status'] == 'failed' for j in jobs):
        raise SystemExit('Some jobs failed; see logs and status.json')
    print(f'Completed: {REPORT}', flush=True)

if __name__ == '__main__':
    main()
