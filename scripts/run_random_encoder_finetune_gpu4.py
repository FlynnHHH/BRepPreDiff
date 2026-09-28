#!/usr/bin/env python3
"""Random-initialized full fine-tuning controls for the transfer ablation."""
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
REFERENCE = ROOT / 'runs/lodo_transfer_pre100_ft200_seed42_20260918'
SUITE = ROOT / 'runs/random_encoder_ft200_seed42_20260920'
REPORT = ROOT / 'reports/random_encoder_ft200_seed42_20260920.md'
TARGETS = ('fusion360seg', 'blend', 'tmcad')
BASE = 'inductive9_rotate_mix_constant_pre100_ft200_seed42_20260908_final'
TASKS = dict(fusion360seg='fusion360seg', blend='brepprediff_seg', tmcad='tmcad_cls')


def dump(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def prepare():
    for directory in ('configs', 'logs', 'results', 'state'):
        (SUITE/directory).mkdir(parents=True, exist_ok=True)
    if (SUITE/'manifest.json').exists(): return
    jobs = []
    for target in TARGETS:
        path = REFERENCE/'configs'/f'{target}_finetune.yaml'
        original = yaml.safe_load(path.read_text())
        cfg = copy.deepcopy(original)
        cfg['run'].update(name=f'{target}_random_encoder', output_dir=str(SUITE))
        cfg['train'].update(pretrain_checkpoint=None, resume=None, encoder_freeze_mode='none', epochs=200)
        assert cfg['model']['finetune_head']=='mlp' and cfg['seed']==42
        if target=='tmcad': assert cfg['model']['graph_pooling']=='mean_max'
        # Reference config stored checkpoint=null; LODO loaded its checkpoint at runtime.
        check=copy.deepcopy(cfg); check['run']=original['run']
        assert check==original, 'Unexpected protocol change'
        destination=SUITE/'configs'/f'{target}.yaml'
        destination.write_text(yaml.safe_dump(cfg,sort_keys=False))
        jobs.append(dict(target=target,config=str(destination),name=cfg['run']['name'],reference=str(path)))
    shutil.copytree(REFERENCE/'source/brepprediff', SUITE/'source/brepprediff',
                    ignore=shutil.ignore_patterns('__pycache__'))
    paths=list((SUITE/'source').rglob('*.py'))+list((SUITE/'configs').glob('*.yaml'))+[Path(__file__).resolve()]
    for job in jobs:
        cfg=yaml.safe_load(Path(job['config']).read_text())
        paths += [Path(job['reference']),Path(cfg['data_config'])]
        paths += [ROOT/cfg['data'][f'{split}_split'] for split in ('train','val','test')]
    dump(SUITE/'provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    dump(SUITE/'manifest.json',jobs)


def report():
    lines=['# Random encoder 初始化后全量微调', '',
           'GPU 4 串行运行 Fusion360Seg、Blend、TMCAD；随机初始化完整编码器及 MLP head，不加载预训练权重，encoder_freeze_mode=none。',
           '复用先前迁移消融的源码快照与下游协议：seed 42、200 epochs、完整训练划分、50% SO(3)、相同 batch/累积、优化器、损失和 validation accuracy 选优。TMCAD 使用 mean_max。初次训练 resume=null，重启仅恢复本实验自己的 checkpoint。',
           '与先前 LODO 微调的实质区别只有预训练初始化；保存的 LODO 配置中 checkpoint 为 null，实际运行时由 override 注入。测试使用 validation 最佳 checkpoint。', '',
           '| 数据集 | 状态 | 随机初始化 Acc % | Macro-F1 % | mIoU % | 完整预训练 Acc % | LODO Acc % | 完整预训练−随机 pp | LODO−随机 pp |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for target in TARGETS:
        state=SUITE/'state'/f'{target}.json'
        status=json.loads(state.read_text()).get('status','pending') if state.exists() else 'pending'
        base=list((ROOT/'runs/inductive9_rotate_mix/finetune').glob(f'*_full_{TASKS[target]}_mlp_ft200_seed42_{BASE}/test_metrics.json'))
        assert len(base)==1
        full=json.loads(base[0].read_text()); lodo=json.loads((REFERENCE/'results'/f'{target}_finetune.json').read_text())
        result=SUITE/'results'/f'{target}.json'
        values=['—']*3; deltas=['—']*2
        if result.exists():
            r=json.loads(result.read_text()); m=r['metrics']
            assert (r['samples'],r['split'])==(full['samples'],full['split'])==(lodo['samples'],lodo['split'])
            values=[f'{100*m[k]:.4f}' for k in ('accuracy','macro_f1','macro_iou')]
            deltas=[f"{100*(a['metrics']['accuracy']-m['accuracy']):+.4f}" for a in (full,lodo)]
            rows.append(dict(dataset=target,accuracy=m['accuracy'],macro_f1=m['macro_f1'],macro_iou=m['macro_iou'],best_epoch=r['checkpoint_epoch'],samples=r['samples'],full_pretrain_accuracy=full['metrics']['accuracy'],lodo_accuracy=lodo['metrics']['accuracy'],result=str(result)))
        lines.append('| '+' | '.join([target,status,*values,f"{100*full['metrics']['accuracy']:.4f}",f"{100*lodo['metrics']['accuracy']:.4f}",*deltas])+' |')
    lines += ['', '仅单 seed 对照，不将百分点差异视为统计显著性。各任务状态、日志、配置及源码哈希见 `runs/random_encoder_ft200_seed42_20260920/`。']
    tmp=REPORT.with_suffix('.tmp'); tmp.write_text('\n'.join(lines)+'\n'); tmp.replace(REPORT)
    if rows:
        with (SUITE/'results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run(job):
    target=job['target']; state=SUITE/'state'/f'{target}.json'
    if state.exists() and json.loads(state.read_text())['status']=='completed': return
    cfg=yaml.safe_load(Path(job['config']).read_text())
    assert cfg['train']['pretrain_checkpoint'] is None and cfg['train']['encoder_freeze_mode']=='none'
    dump(state,dict(job,status='waiting_for_gpu',gpu=4)); report()
    lockdir=ROOT/'runs/rotate_mix_gpu_locks'; lockdir.mkdir(exist_ok=True)
    with (lockdir/'gpu4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        while True:
            p=subprocess.run(['nvidia-smi','-i','4','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if int(p.stdout.strip())>=50000: break
            time.sleep(20)
        previous=sorted((SUITE/'finetune').glob(f"*_{job['name']}"))
        overrides=[]
        if previous and (previous[-1]/'checkpoints/last.pt').exists():
            overrides=['--override',f'train.resume={previous[-1]}/checkpoints/last.pt']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(SUITE/'source'),PYTHONUNBUFFERED='1',WANDB_MODE='disabled',OMP_NUM_THREADS='4')
        dump(state,dict(job,status='running',gpu=4,pid=os.getpid())); report()
        with (SUITE/'logs'/f'{target}.log').open('a') as log:
            subprocess.run([sys.executable,'-m','brepprediff.training.finetune','--config',job['config'],*overrides],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            directory=sorted((SUITE/'finetune').glob(f"*_{job['name']}"))[-1]
            dump(state,dict(job,status='testing',gpu=4,run=str(directory))); report()
            subprocess.run([sys.executable,'-m','brepprediff.training.evaluate','--checkpoint',str(directory/'checkpoints/best.pt'),'--split','test','--output',str(SUITE/'results'/f'{target}.json'),'--batch-size',str(cfg['train']['batch_size']),'--num-workers','8','--device','cuda'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(state,dict(job,status='completed',gpu=4,run=str(directory))); report()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    os.chdir(ROOT); SUITE.mkdir(parents=True,exist_ok=True)
    lock=(SUITE/'launcher.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare()
    for path,digest in json.loads((SUITE/'provenance.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    report()
    if args.prepare_only:
        print('Validated 3 configs, same source snapshot, random initialization and full encoder training'); return
    (SUITE/'launcher.pid').write_text(str(os.getpid()))
    for job in json.loads((SUITE/'manifest.json').read_text()):
        try: run(job)
        except Exception as exc:
            dump(SUITE/'state'/f"{job['target']}.json",dict(job,status='failed',error=str(exc))); report(); raise
    print(f'All three experiments completed: {REPORT}',flush=True)


if __name__=='__main__': main()
