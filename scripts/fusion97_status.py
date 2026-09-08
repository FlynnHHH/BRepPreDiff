"""Print compact progress without reading terminal progress-bar logs."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1] / 'runs/fusion97'
for manifest in sorted(root.glob('*/*/manifest.json')):
    directory = manifest.parent
    data = json.loads(manifest.read_text())
    progress = {'run': str(directory.relative_to(root)), 'gpu': data['gpu']}
    for stage in ('pretrain', 'finetune'):
        logs = list((directory / stage).glob(f'*/logs/{stage}.log'))
        if not logs:
            continue
        content = logs[0].read_text()
        train = re.findall(r'epoch=(\d+) split=train .*total=([0-9.]+)', content)
        if train:
            progress[stage + '_epoch'] = int(train[-1][0])
        vals = re.findall(r'epoch=(\d+) split=val acc=([0-9.]+)', content)
        if vals:
            best = max(vals, key=lambda row: float(row[1]))
            progress['best_val_acc'] = float(best[1])
            progress['best_val_epoch'] = int(best[0])
            progress['last_val_acc'] = float(vals[-1][1])
    progress['complete'] = (directory / 'result.json').exists()
    print(json.dumps(progress))
