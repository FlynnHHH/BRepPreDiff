#!/usr/bin/env python3
"""Reproduce saved rotate-mix protocol with zero discrete pretraining weights."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
LOSS_ONLY_TAG = 'inductive9_rotate_mix_discrete_off_pre100_ft200_seed42_20260910'
GEOMETRY_ONLY = os.environ.get('BREPPREDIFF_GEOMETRY_ONLY') == '1'
TAG = 'inductive9_rotate_mix_geometry_only_pre100_ft200_seed42_20260910' if GEOMETRY_ONLY else LOSS_ONLY_TAG
PRETRAIN_GPU = 1 if GEOMETRY_ONLY else 0
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
        for head in ['mlp', 'diffloss']:
            matches = list((ROOT / 'runs/inductive9_rotate_mix/finetune').glob(f'*_full_{task}_{head}_ft200_seed42_{BASE}'))
            assert len(matches) == 1
            sources.append(('finetune', f'{task}_{head}', matches[0]))
    for stage, name, source in sources:
        original = yaml.safe_load((source / 'config.yaml').read_text())
        cfg = copy.deepcopy(original)
        cfg['run'].update(name=f'{name}_{TAG}', output_dir=str(SUITE))
        if GEOMETRY_ONLY:
            cfg['model']['use_discrete_attributes'] = False
        assert cfg['train']['resume'] is None
        assert cfg['seed'] == 42 and cfg['train']['rotation_augmentation_probability'] == 0.5
        assert cfg['train']['epochs'] == (100 if stage == 'pretrain' else 200)
        if stage == 'pretrain':
            cfg['diffusion'].update(categorical_loss_weight=0.0, relation_loss_weight=0.0)
        else:
            cfg['train']['pretrain_checkpoint'] = '__NEW_PRETRAIN_CHECKPOINT__'
        a, b = flatten(original), flatten(cfg)
        changes = {k: {'before': a.get(k), 'after': b.get(k)} for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        allowed = {'run.name', 'run.output_dir', 'train.pretrain_checkpoint', 'diffusion.categorical_loss_weight', 'diffusion.relation_loss_weight', 'model.use_discrete_attributes'}
        assert changes.keys() <= allowed, changes
        path = SUITE / 'configs' / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append(dict(name=name, stage=stage, config=str(path), baseline=str(source), changes=changes, status='pending'))
    dump(SUITE / 'manifest.json', jobs)
    return jobs

def report(jobs):
    lines = ['# Rotate-mix：关闭离散预训练损失对照', '', f'基线：`reports/{BASE}.md`。', '',
        '仅将 categorical_loss_weight 从 0.5 改为 0、relation_loss_weight 从 0.3 改为 0；保留离散输入特征和网络结构。',
        '从基线实际 config.yaml 复制配置：预训练 100 epochs、微调 200 epochs、seed 42、50% SO(3)、原 batch/梯度累积、数据划分、Mean+Max 分类池化和验证选优规则。',
        '预训练使用 GPU 0 单卡保持 batch 128；随后 GPU 0–3 每卡一路微调。基线结果复用，差值为关闭损失后减去基线（百分点）。', '',
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
            for metric in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_iou']:
                n, o = new['metrics'][metric], old['metrics'][metric]
                row.extend([f'{100*n:.4f}', f'{100*(n-o):+.4f}'])
        else:
            row += ['—'] * 8
        lines.append('| ' + ' | '.join(row) + ' |')
    if GEOMETRY_ONLY:
        lines[0] = '# Rotate-mix：仅几何输入预训练与微调'
        lines[4] = '预训练、微调和评估均设置 model.use_discrete_attributes=false；绕过面类型、边类型和关系类型 embedding，且不读取离散监督目标。两项离散损失权重为 0。缓存中的离散字段保留但不参与模型计算。'
        lines[6] = 'GPU 1 单卡预训练，随后 GPU 0–3 共享每卡互斥锁；启动前至少 50000 MiB 空闲。保持原 batch，不采用多卡 DDP。表中差值相对 full-loss 基线。'
        lines += ['', '## 相对正在运行的“仅关闭离散损失”实验', '',
                  '| Task / Head | ΔAcc | ΔMacro-F1 | ΔWeighted-F1 | ΔmIoU |',
                  '|---|---:|---:|---:|---:|']
        for job in jobs[1:]:
            new_path = SUITE / 'results' / f"{job['name']}.json"
            old_path = ROOT / 'runs' / LOSS_ONLY_TAG / 'results' / f"{job['name']}.json"
            values = ['待两组完成'] * 4
            if new_path.exists() and old_path.exists():
                new = json.loads(new_path.read_text())
                old = json.loads(old_path.read_text())
                assert new['samples'] == old['samples']
                values = [f"{100*(new['metrics'][k]-old['metrics'][k]):+.4f}" for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_iou']]
            lines.append('| ' + ' | '.join([job['name']] + values) + ' |')
        lines += ['', '保留连续几何中的夹角、法向等信息和图连接关系；它们可能蕴含类型线索，本实验消融的是显式离散属性输入。',
                  '保留未使用模块的初始化顺序和 checkpoint 结构，以保持其他参数的随机初始化一致；离散 embedding 不参与前向传播、无梯度。']
    REPORT.write_text('\n'.join(lines) + '\n')
    dump(SUITE / 'status.json', jobs)

def free_memory(gpu):
    p = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    return int(p.stdout.strip())

def run_job(job, gpu):
    lock_dir = ROOT / 'runs' / 'rotate_mix_gpu_locks'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'gpu{gpu}.lock').open('a') as lock:
        # The already-running loss-only pretrain predates these locks.
        while GEOMETRY_ONLY and gpu == 0 and not (ROOT / 'runs' / LOSS_ONLY_TAG / 'pretrain_checkpoint').exists():
            time.sleep(30)
        fcntl.flock(lock, fcntl.LOCK_EX)
        while free_memory(gpu) < MIN_FREE_MIB:
            time.sleep(30)
        run_job_locked(job, gpu)


def run_job_locked(job, gpu):
    cfg = yaml.safe_load(Path(job['config']).read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (SUITE / 'logs' / f"{job['name']}.log").open('a') as log:
        subprocess.run([sys.executable, '-m', f"brepprediff.training.{job['stage']}", '--config', job['config']], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
        jobs = json.loads((SUITE / 'manifest.json').read_text())
        run_job(next(j for j in jobs if j['name'] == args.worker), args.gpu)
        return
    SUITE.mkdir(parents=True, exist_ok=True)
    lock = (SUITE / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = prepare() if not (SUITE / 'manifest.json').exists() else json.loads((SUITE / 'manifest.json').read_text())
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
        job['status'] = 'completed' if (SUITE / 'results' / f"{job['name']}.json").exists() else 'pending'
    pending = [j for j in jobs[1:] if j['status'] == 'pending']
    active = {}
    while pending or active:
        for gpu, (process, job) in list(active.items()):
            code = process.poll()
            if code is not None:
                job.update(status='completed' if code == 0 else 'failed', returncode=code)
                del active[gpu]
        for gpu in range(4):
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
