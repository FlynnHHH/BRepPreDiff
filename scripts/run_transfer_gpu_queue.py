#!/usr/bin/env python3
"""Route existing LODO jobs to GPUs, optionally adopting a live pretrain child."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import yaml
from run_lodo_transfer import ROOT, SUITE, dump, execute


def process_identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--targets', nargs='+', required=True, choices=['fusion360seg','blend','tmcad'])
    p.add_argument('--adopt-pretrain-pid', type=int)
    args = p.parse_args()
    os.chdir(ROOT)
    lock = (SUITE / f'worker_{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path, digest in json.loads((SUITE / 'provenance.json').read_text()).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    (SUITE / f'worker_{args.gpu}.pid').write_text(str(os.getpid()))
    jobs = json.loads((SUITE/'manifest.json').read_text())
    if args.adopt_pretrain_pid:
        job = next(j for j in jobs if j['target']==args.targets[0] and j['mode']=='pretrain')
        cfg = yaml.safe_load(Path(job['config']).read_text())
        identity = process_identity(args.adopt_pretrain_pid)
        if identity is not None:
            command = Path(f'/proc/{args.adopt_pretrain_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'brepprediff.training.pretrain' in command and job['config'] in command, command
        with (ROOT/'runs/rotate_mix_gpu_locks'/f'gpu{args.gpu}.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            dump(SUITE/'state'/f"{job['name']}.json",dict(job,status='running',gpu=args.gpu,pid=os.getpid(),training_pid=args.adopt_pretrain_pid,adopted=True))
            print(f'Adopted existing pretrain PID {args.adopt_pretrain_pid}; no training restart',flush=True)
            while identity is not None and process_identity(args.adopt_pretrain_pid)==identity:
                time.sleep(10)
            runs=sorted((SUITE/'pretrain').glob(f"*_{job['name']}"))
            run=runs[-1]
            checkpoint=run/'checkpoints/last.pt'
            log=run/'logs/pretrain.log'
            if checkpoint.exists() and log.exists() and 'finished pretraining' in log.read_text():
                import torch
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert saved['epoch']==cfg['train']['epochs'],saved['epoch']
                (SUITE/f"{job['target']}_pretrain_checkpoint").write_text(str(checkpoint))
                dump(SUITE/'state'/f"{job['name']}.json",dict(job,status='completed',gpu=args.gpu,run=str(run)))
            else:
                print('Adopted process exited before completion; resume own checkpoint below',flush=True)
    for target in args.targets:
        for mode in ('pretrain','finetune'):
            job=next(j for j in jobs if j['target']==target and j['mode']==mode)
            try:
                execute(job,args.gpu)
            except Exception as exc:
                dump(SUITE/'state'/f"{job['name']}.json",dict(job,status='failed',gpu=args.gpu,error=str(exc)))
                raise


if __name__=='__main__': main()
