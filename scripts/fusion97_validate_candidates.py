"""Validation-only post-training search for Fusion97 rotation and ensembles."""
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
RESULTS = {
    'context': 'runs/fusion97/round1/context/result.json',
    'rotation': 'runs/fusion97/round3/context_rotation/result.json',
    'operation': 'runs/fusion97/round3/context_operation/result.json',
    'rotation_operation': 'runs/fusion97/round4/context_rotation_operation/result.json',
    'rotation_mix': 'runs/fusion97/round4/context_rotation_mix/result.json',
    'rotation_seed43': 'runs/fusion97/round5/context_rotation_seed43/result.json',
}
CANDIDATES = {
    'rotation_tta4': (['rotation'], 4),
    'rotation_context': (['rotation', 'context'], 1),
    'rotation_operation': (['rotation', 'operation'], 1),
    'rotation_rotation_operation': (['rotation', 'rotation_operation'], 1),
    'rotation_rotation_mix': (['rotation', 'rotation_mix'], 1),
    'rotation_mix_operation': (['rotation', 'rotation_mix', 'rotation_operation'], 1),
    'rotation_seed43': (['rotation', 'rotation_seed43'], 1),
    'rotation_mix_seed43': (['rotation', 'rotation_mix', 'rotation_seed43'], 1),
    'rotation_all': (['rotation', 'rotation_mix', 'rotation_operation', 'rotation_seed43'], 1),
    'rotation_temporal': (['rotation', 'rotation_epoch100'], 1),
    'rotation_temporal_seed43': (['rotation', 'rotation_epoch100', 'rotation_seed43'], 1),
}


def checkpoint(result_path: Path) -> str:
    payload = json.loads(result_path.read_text())
    validation = payload['validation']
    assert validation['split'] == 'val' and not payload.get('test_accessed', False)
    return validation['checkpoint']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=(4,), default=4)
    parser.add_argument('--output-dir', default='runs/fusion97/validation_search_v1')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = output_dir / 'source'
    if not snapshot.exists():
        shutil.copytree(ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(__file__, snapshot / Path(__file__).name)
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(snapshot.rglob('*.py'))}
    result_paths = {name: ROOT / path for name, path in RESULTS.items()}
    missing = [str(path) for path in result_paths.values() if not path.exists()]
    while missing and args.wait:
        print(f'Waiting for {len(missing)} result files', flush=True)
        time.sleep(30)
        missing = [str(path) for path in result_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing prerequisite results: {missing}')
    checkpoints = {name: checkpoint(path) for name, path in result_paths.items()}
    rotation_checkpoint_dir = Path(checkpoints['rotation']).parent
    rotation_epoch100 = rotation_checkpoint_dir / 'epoch_0100.pt'
    if not rotation_epoch100.is_file():
        raise FileNotFoundError(f'Missing temporal checkpoint: {rotation_epoch100}')
    checkpoints['rotation_epoch100'] = str(rotation_epoch100)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               PYTHONPATH=str(snapshot / 'src'), WANDB_MODE='disabled')
    rows = []
    for name, (members, tta) in CANDIDATES.items():
        output = output_dir / f'{name}.json'
        command = [sys.executable, '-m', 'brepprediff.training.evaluate',
                   '--checkpoint', checkpoints[members[0]], '--split', 'val',
                   '--output', str(output), '--num-workers', '4',
                   '--rotation-tta', str(tta)]
        for member in members[1:]:
            command += ['--ensemble-checkpoint', checkpoints[member]]
        with (output_dir / f'{name}.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        assert result['split'] == 'val'
        rows.append({'candidate': name, 'members': members, 'rotation_tta': tta,
                     'accuracy': result['metrics']['accuracy'],
                     'macro_f1': result['metrics']['macro_f1'],
                     'output': str(output)})
    rows.sort(key=lambda row: row['accuracy'], reverse=True)
    summary = {'selection_split': 'val', 'test_accessed': False, 'gpu': args.gpu,
               'source_hashes': hashes, 'candidates': rows}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
