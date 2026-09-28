#!/usr/bin/env python3
"""Refresh a compact report for the two independent transfer ablation queues."""
import argparse
import csv
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'runs/lodo_transfer_pre100_ft200_seed42_20260918'
REPORT = ROOT / 'reports/transfer_ablation_results_20260918.md'
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
TASKS = {'fusion360seg':'fusion360seg', 'blend':'brepprediff_seg', 'tmcad':'tmcad_cls'}


def summarize():
    lines = ['# 全量预训练 Linear probe / Leave-one-dataset-out', '',
             'GPU 2：完整 Inductive9 checkpoint 冻结编码器的纯线性 probe。GPU 1：Fusion360Seg/Blend；GPU 0：TMCAD。各目标独立排除后预训练 100 epochs，再全量微调 200 epochs。下游完整训练划分，seed 42，validation-best 测试。', '',
             '设计与证据边界见 [实验设计](lodo_transfer_design_20260918.md)。未完成指标留空，不把验证集结果当作测试结果。', '',
             '| Dataset | Arm | Status | Test Accuracy % | Macro-F1 % | mIoU % | ΔAcc vs full FT (pp) |',
             '|---|---|---|---:|---:|---:|---:|']
    rows = []
    for target, task in TASKS.items():
        baseline = list((ROOT/'runs/inductive9_rotate_mix/finetune').glob(f'*_full_{task}_mlp_ft200_seed42_{BASE}/test_metrics.json'))
        old = json.loads(baseline[0].read_text()) if len(baseline) == 1 else None
        for arm in ('baseline', 'probe', 'finetune'):
            result = baseline[0] if arm == 'baseline' and old else SUITE/'results'/f'{target}_{arm}.json'
            statefile = SUITE/'state'/f'{target}_{arm}.json'
            state = json.loads(statefile.read_text()) if statefile.exists() else {}
            status = state.get('status', 'pending')
            values = ['—'] * 4
            if result.exists():
                r = json.loads(result.read_text()); m = r['metrics']
                if old: assert (r['split'], r['samples']) == (old['split'], old['samples'])
                status = 'completed'
                values = [f"{100*m[k]:.4f}" for k in ('accuracy', 'macro_f1', 'macro_iou')]
                delta = 100*(m['accuracy']-old['metrics']['accuracy']) if old else None
                values.append(f'{delta:+.4f}' if delta is not None else '—')
                rows.append(dict(dataset=target, arm=arm, samples=r['samples'], epoch=r['checkpoint_epoch'],
                                 accuracy=m['accuracy'], macro_f1=m['macro_f1'], macro_iou=m['macro_iou'],
                                 delta_accuracy_pp=delta, result=str(result)))
            label = {'baseline':'Full pretrain + MLP FT', 'probe':'Full pretrain + Linear probe', 'finetune':'LODO pretrain + MLP FT'}[arm]
            lines.append('| ' + ' | '.join([target, label, status, *values]) + ' |')
    lines += ['', '预训练状态：']
    states = []
    for target in TASKS:
        for mode in ('pretrain','finetune','probe'):
            p = SUITE/'state'/f'{target}_{mode}.json'
            state = json.loads(p.read_text()).get('status','pending') if p.exists() else 'pending'
            states.append(state)
            if mode == 'pretrain': lines.append(f'- {target}: {state}')
    lines += ['', 'probe 与 FT 的差值同时受冻结、head 和分类 pooling 实现差异影响；不能解释为冻结编码器的单独因果效应。单 seed 结果不代表统计显著性。']
    tmp = REPORT.with_suffix('.tmp'); tmp.write_text('\n'.join(lines)+'\n'); tmp.replace(REPORT)
    if rows:
        with (SUITE/'results.csv').open('w') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return all(s in ('completed','failed') for s in states)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--watch',action='store_true'); args=p.parse_args()
    while True:
        done=summarize()
        if done or not args.watch: break
        time.sleep(30)
