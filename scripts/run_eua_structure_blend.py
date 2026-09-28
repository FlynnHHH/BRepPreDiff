#!/usr/bin/env python3
"""E1/E2/A1/A2, exact saved protocol, Blend MLP only; reuse pretrained encoders, GPU 4."""
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
TAG = 'eua_structure_blend_pre100_ft200_seed42_20260918'
PRETRAIN_SUITE = ROOT / 'runs/eua_structure_fusion360seg_pre100_ft200_seed42_20260917'
GPU = 4
MIN_FREE_MIB = 50000
SUITE = ROOT / 'runs' / TAG
REPORT = ROOT / 'reports' / f'{TAG}.md'
VARIANTS = {'E1': ('fixed', 'softmax'), 'E2': ('edge_only', 'softmax'),
            'A1': ('full', 'uniform'), 'A2': ('fixed', 'uniform')}

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
    pre = list((ROOT / 'runs/inductive9_rotate_mix/pretrain').glob(f'*_{BASE}'))
    fine = list((ROOT / 'runs/inductive9_rotate_mix/finetune').glob(f'*_full_brepprediff_seg_mlp_ft200_seed42_{BASE}'))
    assert len(pre) == len(fine) == 1
    sources = [(variant, stage, source) for variant in VARIANTS
               for stage, source in [('finetune', fine[0])]]
    for variant, stage, source in sources:
        name = f'{variant}_{stage}'
        original = yaml.safe_load((source / 'config.yaml').read_text())
        cfg = copy.deepcopy(original)
        cfg['run'].update(name=f'{name}_{TAG}', output_dir=str(SUITE))
        assert cfg['model']['encoder_type'] == 'edge_update_attention'
        cfg['model']['edge_update_mode'], cfg['model']['attention_mode'] = VARIANTS[variant]
        assert cfg['train']['resume'] is None
        assert cfg['seed'] == 42 and cfg['train']['rotation_augmentation_probability'] == 0.5
        assert cfg['train']['epochs'] == (100 if stage == 'pretrain' else 200)
        if stage == 'finetune':
            checkpoint = (PRETRAIN_SUITE / f'{variant}_pretrain_checkpoint').read_text().strip()
            assert Path(checkpoint).is_file()
            cfg['train']['pretrain_checkpoint'] = checkpoint
            (SUITE / f'{variant}_pretrain_checkpoint').write_text(checkpoint)
            assert cfg['model']['num_classes'] == 3 and cfg['model']['finetune_head'] == 'mlp'
        a, b = flatten(original), flatten(cfg)
        changes = {k: {'before': a.get(k), 'after': b.get(k)} for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        allowed = {'run.name', 'run.output_dir', 'train.pretrain_checkpoint', 'model.edge_update_mode', 'model.attention_mode'}
        assert changes.keys() <= allowed, changes
        path = SUITE / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(dict(name=name, variant=variant, stage=stage, config=str(path), baseline=str(source), changes=changes, status='pending'))
    dump(SUITE / 'manifest.json', jobs)
    return jobs

def report(jobs):
    baseline = Path(next(j['baseline'] for j in jobs if j['stage'] == 'finetune')) / 'test_metrics.json'
    old = json.loads(baseline.read_text())
    rows = []
    lines = ['# EUA 结构消融：Blend（BRepPreDiff 三分类过渡面分割）/ MLP', '',
             'E0：复用完整 EUA；E1：attention + 固定初始边；E2：attention + MLP([0,0,edge])；A1：均匀权重 + 完整边更新；A2：均匀权重 + 固定初始边。',
             '固定边不冻结初始 embedding 参数。保留模块形状和初始化，关闭路径不参与计算；注册参数量一致不等于有效容量一致。预训练 edge_context 的端点交互仍保留。',
             '复制基线实际配置：Inductive9 预训练 100 epochs，batch 128，LR 1e-4；Blend MLP 微调 200 epochs；seed 42，50% SO(3)，其余 batch/累积、损失、验证选优及数据划分不变。',
             '仅使用 GPU 4，每次一个变体；复用 Fusion360Seg 消融中各变体的 100-epoch 预训练权重，仅重新微调 Blend，并用 validation accuracy 最优 checkpoint 测试。单 seed 结果不代表统计显著性。',
             f'基线结果：`{baseline}`。实验目录：`{SUITE}`。', '',
             '| 变体 | 预训练 | 微调/测试 | Accuracy % | ΔAcc pp | Macro-F1 % | mIoU % | ΔmIoU pp |',
             '|---|---|---|---:|---:|---:|---:|---:|',
             f"| E0 | 已完成 | 复用基线 | {100*old['metrics']['accuracy']:.4f} | 0 | {100*old['metrics']['macro_f1']:.4f} | {100*old['metrics']['macro_iou']:.4f} | 0 |"]
    for variant in VARIANTS:
        fine = next(j for j in jobs if j['variant'] == variant)
        pre = {'status': '复用100 epochs'}
        result = SUITE / 'results' / f"{fine['name']}.json"
        values = ['—'] * 5
        if result.exists():
            new = json.loads(result.read_text())
            assert new['split'] == old['split'] == 'test' and new['samples'] == old['samples']
            m, b = new['metrics'], old['metrics']
            values = [f"{100*m['accuracy']:.4f}", f"{100*(m['accuracy']-b['accuracy']):+.4f}",
                      f"{100*m['macro_f1']:.4f}", f"{100*m['macro_iou']:.4f}", f"{100*(m['macro_iou']-b['macro_iou']):+.4f}"]
            row = dict(variant=variant, samples=new['samples'], best_epoch=new['checkpoint_epoch'], result=str(result))
            for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_iou']:
                row.update({k:m[k], 'baseline_'+k:b[k], 'delta_pp_'+k:100*(m[k]-b[k])})
            rows.append(row)
        lines.append('| ' + ' | '.join([variant, pre['status'], fine['status'], *values]) + ' |')
    if rows:
        with (SUITE / 'results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if len(rows) == 4:
        by_id = {r['variant']:r for r in rows}
        lines += ['', '交互量 I = (E0−E1)−(A1−A2)，单位百分点：']
        for k in ['accuracy', 'macro_iou']:
            value = 100*(old['metrics'][k]-by_id['E1'][k]-by_id['A1'][k]+by_id['A2'][k])
            lines.append(f'- {k}: {value:+.4f}')
    lines += ['', f'新增测试结果：{len(rows)}/4；完整运行状态见 status.json，配置差异见 manifest.json。']
    REPORT.write_text('\n'.join(lines)+'\n')
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
            (SUITE / f"{job['variant']}_pretrain_checkpoint").write_text(str(run / 'checkpoints/last.pt'))
        else:
            subprocess.run([sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint', str(run / 'checkpoints/best.pt'), '--split', 'test', '--output', str(SUITE / 'results' / f"{job['name']}.json"), '--batch-size', str(cfg['train']['batch_size']), '--num-workers', str(cfg['train']['num_workers']), '--device', 'cuda'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    SUITE.mkdir(parents=True, exist_ok=True)
    lock = (SUITE / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = prepare() if not (SUITE / 'manifest.json').exists() else json.loads((SUITE / 'manifest.json').read_text())
    provenance = SUITE / 'provenance.json'
    if not provenance.exists():
        shutil.copytree(PRETRAIN_SUITE / 'source/brepprediff', SUITE / 'source/brepprediff', ignore=shutil.ignore_patterns('__pycache__'))
        paths = list((SUITE / 'source').rglob('*.py')) + [Path(__file__).resolve()]
        paths += list((ROOT / 'data').rglob('*.yaml')) + list((ROOT / 'data/splits').glob('*.txt'))
        paths += [Path(j['baseline'])/'config.yaml' for j in jobs]
        paths += [Path((SUITE/f"{j['variant']}_pretrain_checkpoint").read_text().strip()) for j in jobs]
        paths += [Path(j['baseline'])/'test_metrics.json' for j in jobs if j['stage']=='finetune']
        dump(provenance, {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    else:
        for path, expected in json.loads(provenance.read_text()).items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Experiment dependency changed: {path}')
    report(jobs)
    if args.prepare_only:
        print(f'Validated {len(jobs)} configs; {SUITE}', flush=True)
        return
    (SUITE / 'launcher.pid').write_text(str(os.getpid()))
    for job in jobs:
        marker = SUITE / f"{job['variant']}_pretrain_checkpoint"
        result = SUITE / 'results' / f"{job['name']}.json"
        if (job['stage']=='pretrain' and marker.exists() and Path(marker.read_text().strip()).is_file()) or (job['stage']=='finetune' and result.exists()):
            job['status']='completed'; report(jobs); continue
        if job['stage']=='finetune':
            if not marker.exists():
                job['status']='blocked_pretrain_failed'; report(jobs); continue
            cfg_path = Path(job['config'])
            cfg = yaml.safe_load(cfg_path.read_text())
            checkpoint = marker.read_text().strip()
            assert Path(checkpoint).is_file()
            cfg['train']['pretrain_checkpoint']=checkpoint
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            job['changes']['train.pretrain_checkpoint']['after']=checkpoint
            dump(SUITE/'manifest.json',jobs)
        job.update(status='waiting_for_gpu',gpu=GPU); report(jobs)
        # Hold the shared GPU lock throughout this stage and check free memory.
        lock_dir=ROOT/'runs/rotate_mix_gpu_locks'; lock_dir.mkdir(exist_ok=True)
        with (lock_dir/f'gpu{GPU}.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            while free_memory(GPU)<MIN_FREE_MIB: time.sleep(30)
            job['status']='running'; report(jobs)
            try:
                run_job_locked(job,GPU)
                job['status']='completed'
            except Exception as exc:
                job.update(status='failed',error=str(exc))
                print(f"Failed {job['name']}: {exc}",flush=True)
            report(jobs)
    if any(j['status']!='completed' for j in jobs):
        raise SystemExit('Some jobs failed; inspect status.json and logs')
    print(f'Completed: {REPORT}',flush=True)

if __name__ == '__main__':
    main()
