#!/usr/bin/env python3
"""Auditable rotate-mix ratio suite, memory-gated GPU queues and live report."""
import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_TAG = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
TASKS = {'brepprediff_seg': 'BRepPreDiff', 'fusion360seg': 'Fusion360Seg',
         'mfcadpp_seg': 'MFCAD++', 'tmcad_cls': 'TMCAD', 'solidletters_cls': 'SolidLetters',
         'cadsynth_seg': 'CADSynth', 'mfinstseg_seg': 'MFInstSeg'}
RATIOS = [0.1, 0.5, 1, 1.5, 2, 3]

def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def report(suite, jobs, baselines):
    rows = []
    for job in baselines + jobs:
        path = Path(job['result'])
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        audit = json.loads(Path(job['audit']).read_text())
        m = payload['metrics']
        rows.append(dict(task=job['task'], head=job['head'], ratio=job['ratio'],
                         actual_ratio=audit['actual_ratio_percent'], train_samples=audit['selected_samples'],
                         test_samples=payload['samples'], best_epoch=payload['checkpoint_epoch'],
                         **{k: m[k] for k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_iou']},
                         result=str(path)))
    rows.sort(key=lambda r: (r['task'], r['head'], r['ratio']))
    if rows:
        with (suite / 'results.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete = sum(Path(j['result']).exists() for j in jobs)
    lines = ['# Rotate mix 多监督比率微调', '', f'新实验完成：{complete}/{len(jobs)}。100% 结果复用匹配的已有实验。', '',
             '统一 seed 42、200 epochs、全参数微调、50% canonical + 50% SO(3) rotation；分类使用 Mean+Max pooling。',
             '继承各自 100% 配置的 batch size、梯度累积、优化器及 head 参数，每 epoch 在完整 validation 上按 accuracy 选择 best，再评估完整 test。',
             '监督比例按训练 CAD 模型计，样本数向上取整；seed 42 嵌套子集，类别覆盖优先后按 SHA-256 排序。极低比例可能无法覆盖所有类别；该抽样使用完整训练集标签决定覆盖顺序。',
             '各比例固定 epoch 而非固定优化步数；单种子结果不代表统计显著性。GPU 0–3 每卡最多一个本次任务，GPU 4 最多三路，启动前检查显存。',
             '', '| 数据集 | Head | 标注比例 | 实际比例 | 训练 CAD | Best epoch | Accuracy % | Macro-F1 % | Weighted-F1 % | mIoU % |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {TASKS[r['task']]} | {r['head']} | {r['ratio']:g}% | {r['actual_ratio']:.4f}% | {r['train_samples']} | {r['best_epoch']} | " + ' | '.join(f'{100*r[k]:.4f}' for k in ['accuracy','macro_f1','weighted_f1','macro_iou']) + ' |')
    lines += ['', '## 标注子集审计', '', '| 数据集 | 比例 | CAD 数 | 覆盖类别 / 总类别 |', '|---|---:|---:|---:|']
    seen = set()
    for job in jobs:
        key = (job['task'], job['ratio'])
        if key in seen:
            continue
        seen.add(key)
        audit = json.loads(Path(job['audit']).read_text())
        support = audit['model_support_per_class']
        lines.append(f"| {TASKS[job['task']]} | {job['ratio']:g}% | {audit['selected_samples']} | {sum(v > 0 for v in support.values())} / {len(support)} |")
    lines += ['', '## 相对 100% 的差值（百分点）', '', '| 数据集 | Head | 比例 | ΔAccuracy | ΔMacro-F1 | ΔmIoU |', '|---|---|---:|---:|---:|---:|']
    full = {(r['task'], r['head']): r for r in rows if r['ratio'] == 100}
    for r in rows:
        if r['ratio'] == 100:
            continue
        b = full[r['task'], r['head']]
        lines.append(f"| {TASKS[r['task']]} | {r['head']} | {r['ratio']:g}% | " + ' | '.join(f'{100*(r[k]-b[k]):+.4f}' for k in ['accuracy','macro_f1','macro_iou']) + ' |')
    if complete == len(jobs):
        plot_results(suite, rows)
        lines += ['', f'![监督比例曲线]({suite / "ratio_curves.png"})', '']
    lines += ['', '## 复现与运行状态', '', f'实验目录：`{suite}`；`manifest.json` 保存配置、encoder、原始基线、split audit 和结果路径，`status.json` 保存队列状态；完整数值见 `results.csv`。', '']
    for j in jobs:
        if j.get('status') == 'failed':
            lines.append(f"- 失败：{j['name']}，见 `{j['log']}`。")
    (ROOT / 'reports' / f'{suite.name}.md').write_text('\n'.join(lines) + '\n')

def plot_results(suite, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 3 * len(tasks)), squeeze=False)
    for i, task in enumerate(tasks):
        for ax, metric in zip(axes[i], ['accuracy', 'macro_iou']):
            for head in ['mlp', 'diffloss']:
                selected = sorted([r for r in rows if r['task'] == task and r['head'] == head], key=lambda r: r['ratio'])
                if selected:
                    ax.plot([r['ratio'] for r in selected], [100*r[metric] for r in selected], marker='o', label=head)
            ax.set_xscale('log')
            ax.set_xticks(RATIOS + [100], [str(x) for x in RATIOS + [100]])
            ax.set(title=f'{TASKS[task]}: {metric}', xlabel='Labeled training CADs (%)', ylabel='%')
            ax.grid(alpha=.25)
            ax.legend()
    fig.tight_layout()
    fig.savefig(suite / 'ratio_curves.png', dpi=180)
    fig.savefig(suite / 'ratio_curves.pdf')
    plt.close(fig)


def prepare(suite, tasks, heads):
    jobs, baselines = [], []
    split_root = suite / 'splits'
    split_root.mkdir(exist_ok=True)
    for task in tasks:
        sources = {}
        for head in heads:
            matches = list((ROOT / 'runs/inductive9_rotate_mix/finetune').glob(f'*_full_{task}_{head}_ft200_seed42_{BASE_TAG}'))
            assert len(matches) == 1, (task, head, matches)
            sources[head] = matches[0]
        base = yaml.safe_load((sources[heads[0]] / 'config.yaml').read_text())
        cmd = [sys.executable, 'scripts/create_supervision_ratio_splits.py', '--config', str(sources[heads[0]] / 'config.yaml'), '--task-name', TASKS[task], '--ratios', *map(str, RATIOS + [100]), '--seed', '42', '--output-dir', str(split_root)]
        subprocess.run(cmd, check=True)
        for head, source in sources.items():
            original = yaml.safe_load((source / 'config.yaml').read_text())
            assert original['data']['train_split'] == base['data']['train_split']
            assert original['train']['epochs'] == 200
            assert original['train']['rotation_augmentation_probability'] == 0.5
            assert Path(original['train']['pretrain_checkpoint']).is_file()
            for ratio in [100] + RATIOS:
                tag = f'{ratio:g}'.replace('.', 'p')
                split = split_root / f'{TASKS[task]}_r{tag}_train.txt'
                job = dict(task=task, head=head, ratio=ratio, audit=str(split.with_suffix('.audit.json')), baseline=str(source), encoder=original['train']['pretrain_checkpoint'])
                if ratio == 100:
                    job['result'] = str(source / 'test_metrics.json')
                    assert Path(job['result']).is_file()
                    baselines.append(job)
                    continue
                name = f'{task}_{head}_r{tag}_{suite.name}'
                cfg = copy.deepcopy(original)
                cfg['run'].update(name=name, output_dir=str(suite), save_every_epochs=200, show_progress=False)
                cfg['data']['train_split'] = str(split)
                cfg['train'].update(resume=None, num_workers=4)
                cfg['wandb']['enabled'] = False
                config_path = suite / 'configs' / f'{name}.yaml'
                config_path.parent.mkdir(exist_ok=True)
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                job.update(name=name, config=str(config_path), result=str(suite / 'results' / f'{name}.json'), log=str(suite / 'logs' / f'{name}.log'), status='pending', attempts=0)
                jobs.append(job)
    # Interleave datasets/heads at each ratio.
    jobs.sort(key=lambda j: (j['ratio'], list(TASKS).index(j['task']), j['head']))
    write_json(suite / 'manifest.json', dict(jobs=jobs, baselines=baselines))
    return jobs, baselines

def free_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    return {int(a): int(b) for a,b in (line.split(',') for line in result.stdout.splitlines())}

def worker(suite, job, gpu):
    cfg = yaml.safe_load(Path(job['config']).read_text())
    cmd = [sys.executable, '-m', 'brepprediff.training.finetune', '--config', job['config']]
    subprocess.run(cmd, check=True)
    run = sorted((suite / 'finetune').glob(f"*_{job['name']}"))[-1]
    subprocess.run([sys.executable, '-m', 'brepprediff.training.evaluate', '--checkpoint', str(run / 'checkpoints/best.pt'), '--split', 'test', '--output', job['result'], '--batch-size', str(cfg['train']['batch_size']), '--num-workers', '4', '--device', 'cuda'], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=list(TASKS), default=list(TASKS))
    p.add_argument('--heads', nargs='+', choices=['mlp','diffloss'], default=['mlp','diffloss'])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--worker')
    a = p.parse_args()
    os.chdir(ROOT)
    os.environ.update(PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1', WANDB_MODE='disabled', OMP_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    suite = a.suite.resolve()
    suite.mkdir(parents=True, exist_ok=True)
    if a.worker:
        manifest = json.loads((suite / 'manifest.json').read_text())
        worker(suite, next(j for j in manifest['jobs'] if j['name'] == a.worker), os.environ['CUDA_VISIBLE_DEVICES'])
        return
    lock = (suite / 'launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (suite / 'launcher.pid').write_text(str(os.getpid()))
    for folder in ['logs', 'results']:
        (suite / folder).mkdir(exist_ok=True)
    if (suite / 'manifest.json').exists():
        manifest = json.loads((suite / 'manifest.json').read_text())
        jobs, baselines = manifest['jobs'], manifest['baselines']
    else:
        jobs, baselines = prepare(suite, a.tasks, a.heads)
    report(suite, jobs, baselines)
    if a.prepare_only:
        return
    running = []
    last_launch = 0
    while True:
        for item in running[:]:
            proc, job, gpu, handle = item
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            running.remove(item)
            if code == 0 and Path(job['result']).exists():
                job['status'] = 'complete'
            else:
                oom = 'out of memory' in Path(job['log']).read_text().lower()
                job['status'] = 'pending' if oom and job['attempts'] < 3 else 'failed'
                job['retry_solo'] = oom
            print(time.strftime('%F %T'), job['name'], job['status'], flush=True)
            report(suite, jobs, baselines)
        for j in jobs:
            if Path(j['result']).exists():
                j['status'] = 'complete'
        pending = [j for j in jobs if j['status'] == 'pending']
        write_json(suite / 'status.json', jobs)
        if not pending and not running:
            break
        if pending and time.time() - last_launch >= 45:
            try:
                memory = free_memory()
            except (subprocess.SubprocessError, ValueError):
                time.sleep(10)
                continue
            for gpu in [0,1,2,3,4]:
                active = [x for x in running if x[2] == gpu]
                job = pending[0]
                capacity = 1 if gpu != 4 or job.get('retry_solo') else 3
                # Conservative 22 GiB admission budget plus 8 GiB reserve.
                if any(x[1].get('retry_solo') for x in active):
                    continue
                if len(active) >= capacity or memory.get(gpu, 0) < 30720:
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                handle = Path(job['log']).open('a')
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--suite', str(suite), '--worker', job['name']], env=env, stdout=handle, stderr=subprocess.STDOUT)
                job.update(status='running', gpu=gpu, pid=proc.pid, attempts=job['attempts'] + 1)
                running.append((proc,job,gpu,handle))
                last_launch = time.time()
                print(time.strftime('%F %T'), 'started', job['name'], 'gpu', gpu, flush=True)
                break
        time.sleep(5)
    report(suite, jobs, baselines)
    write_json(suite / 'status.json', jobs)
    print('Suite finished', flush=True)
    if any(j['status'] == 'failed' for j in jobs):
        sys.exit(1)

if __name__ == '__main__':
    main()
