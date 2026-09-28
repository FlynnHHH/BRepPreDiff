"""Summarize preserved BPD pretraining and downstream runs without retraining.

Run with PYTHONPATH=src and an environment providing torch and PyYAML.
Each downstream model is instantiated from its run's embedded config on CPU.
"""
from pathlib import Path
from datetime import datetime
import csv
import json
import re
import statistics

import yaml
from brepprediff.config import feature_dims
from brepprediff.models.diffusion import DiffusionPretrainModel
from brepprediff.models.finetune import build_finetune_model

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'reports/bpd_pretrain_finetune_timing_20260925'
GROUPS = ['inductive_lr1e4_pre100_full_ft200', 'inductive9_rotate_mix']
DATASETS = {'finetune_train.txt': 'BRepPreDiff', 'fusion360seg_train.txt': 'Fusion360Seg',
            'mfcad_train.txt': 'MFCAD++', 'tmcad_train.txt': 'TMCAD',
            'solidletters_train_clean.txt': 'SolidLetters', 'cadsynth_train_clean.txt': 'CADSynth',
            'mfinstseg_train.txt': 'MFInstSeg'}


def parse_epochs(path):
    starts, records = {}, {}
    for line in path.read_text(errors='replace').splitlines():
        match = re.search(r'epoch=(\d+)(/\d+)? split=(train|val) (.*)', line)
        if not match:
            continue
        time = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
        epoch, split = int(match[1]), match[3]
        if match[4].startswith('start'):
            starts[epoch, split] = time
        elif (epoch, split) in starts:
            record = records.setdefault(epoch, {'epoch': epoch})
            record[split + '_start'] = starts.pop((epoch, split)).isoformat(' ')
            record[split + '_end'] = time.isoformat(' ')
            record[split + '_seconds'] = (time - datetime.fromisoformat(record[split + '_start'])).total_seconds()
    rows = [records[e] for e in sorted(records) if 'train_seconds' in records[e]]
    for row in rows:
        if 'val_end' in row:
            row['train_through_val_seconds'] = (datetime.fromisoformat(row['val_end']) - datetime.fromisoformat(row['train_start'])).total_seconds()
    for row, nxt in zip(rows, rows[1:]):
        if nxt['epoch'] == row['epoch'] + 1:
            row['whole_epoch_seconds'] = (datetime.fromisoformat(nxt['train_start']) - datetime.fromisoformat(row['train_start'])).total_seconds()
    return rows


def mean(rows, key):
    values = [row[key] for row in rows if key in row]
    return statistics.mean(values) if values else None


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries, all_epochs = [], []
    paths = list((ROOT / 'runs/inductive_lr1e4_encoder/pretrain').glob('*/logs/pretrain.log'))
    for group in GROUPS:
        paths.extend(sorted((ROOT / 'runs' / group / 'finetune').glob('*/logs/finetune.log')))
    for path in paths:
        config_path = path.parent.parent / 'config.yaml'
        config = yaml.safe_load(config_path.read_text())
        stage = path.stem
        model = (DiffusionPretrainModel if stage == 'pretrain' else build_finetune_model)(config, *feature_dims(config))
        count = sum(p.numel() for p in model.parameters())
        encoder = sum(p.numel() for p in model.encoder.parameters())
        log = path.read_text()
        logged_count = int(re.search(r'model parameters=(\d+)', log)[1])
        assert count == logged_count, (path, count, logged_count)
        loader = re.search(r'dataloader train ready: samples=(\d+) batches_per_rank=(\d+) batch_size_per_rank=(\d+).*world_size=(\d+)', log)
        epochs = parse_epochs(path)
        assert len(epochs) == config['train']['epochs'], (path, len(epochs))
        dataset = '九源 train-only' if stage == 'pretrain' else DATASETS.get(Path(config['data']['train_split']).name, Path(config['data']['train_split']).stem)
        metadata = list(path.parent.parent.glob('wandb/*/files/wandb-metadata.json'))
        gpu = sorted({json.loads(p.read_text()).get('gpu', 'unknown') for p in metadata})
        summary = dict(stage=stage, dataset=dataset, head=config['model'].get('finetune_head', 'pretrain'),
                       classes=config['model'].get('num_classes', ''), parameters=count, encoder_parameters=encoder,
                       prediction_type=config.get('label_diffusion', {}).get('prediction_type', ''),
                       sampling_steps=config.get('label_diffusion', {}).get('sampling_steps', ''),
                       other_parameters=count-encoder, batch_size=int(loader[3]), world_size=int(loader[4]),
                       accumulation=config['train'].get('gradient_accumulation_steps', 1),
                       rotation_probability=config['train'].get('rotation_augmentation_probability', 0),
                       samples=int(loader[1]), batches=int(loader[2]), epochs=len(epochs),
                       pooling=config['model'].get('graph_pooling', 'mean'), gpu='; '.join(gpu) or '未保留该 run 的硬件元数据',
                       group=path.relative_to(ROOT).parts[1], config=str(config_path.relative_to(ROOT)), log=str(path.relative_to(ROOT)))
        for key in ['train_seconds', 'val_seconds', 'train_through_val_seconds', 'whole_epoch_seconds']:
            summary[key] = mean(epochs[1:], key)
        summary['train_stdev_seconds'] = statistics.stdev(row['train_seconds'] for row in epochs[1:])
        summary['train_median_seconds'] = statistics.median(row['train_seconds'] for row in epochs[1:])
        summaries.append(summary)
        for epoch in epochs:
            all_epochs.append(dict(stage=stage, dataset=dataset, head=summary['head'], batch_size=summary['batch_size'], log=summary['log'], **epoch))
    csv_write(PREFIX.with_suffix('.csv'), summaries)
    csv_write(PREFIX.with_name(PREFIX.name + '_epochs.csv'), all_epochs)
    PREFIX.with_suffix('.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')
    pre = summaries[0]
    md = ['# BPD 预训练和下游微调参数量、epoch 用时', '',
          '统计日期：2026-09-25。此报告汇总历史运行，不是重新进行的受控 batch-size 测速。', '',
          '## 统计口径', '',
          '- 从每个 run 的 config.yaml 在 CPU 实例化模型，参数量逐一与历史日志核对一致；不加载权重，不修改配置。',
          '- Encoder 为 edge_update_attention，128 hidden、4 层、4 heads、711/63 输入。下游为全量微调；分类保留原配置 mean_max；全部下游运行 200 epochs。',
          '- 每个 run 剔除其第一轮。纯训练和验证统计 epoch 2–200（预训练 2–100）；整轮统计相邻 start-to-start，最后一轮无下一轮起点，因此下游为 2–199。',
          '- 训练时间包含数据加载、前后向和更新；训练+验证从训练开始到验证结束；整轮还含轮间 checkpoint 和记录开销。时间单位均为秒，原日志精度为秒。',
          '- batch 指每卡图数量；有效更新 batch 约为 batch × 卡数 × 梯度累积，末尾 batch 可能不足。', '',
          '## 参数量', '', '| 数据集 | Head | 类别数 | Encoder | Head/其他 | 总参数 |', '|---|---|---:|---:|---:|---:|']
    seen = set()
    for row in summaries:
        key = (row['dataset'], row['head'], row['parameters'])
        if key in seen:
            continue
        seen.add(key)
        md.append(f"| {row['dataset']} | {row['head']} | {row['classes']} | {row['encoder_parameters']:,} | {row['other_parameters']:,} | {row['parameters']:,} |")
    md += ['', '## 预训练', '',
           f"九源 488,099 个样本，单卡 A800-SXM4-80GB，batch 128，平均纯训练 **{pre['train_seconds']:.2f} ± {pre['train_stdev_seconds']:.2f} 秒/epoch**，整轮 **{pre['whole_epoch_seconds']:.2f} 秒**。没有验证。",
           '旧七源四卡 batch 32、64 仅保留 checkpoint，未保留可用耗时日志。不能填入实测时间，不能与九源单卡结果作为速度消融。', '',
           '## 下游微调耗时', '',
           '两套 cohort 分开列出，避免混合运行平均。A：inductive_lr1e4_pre100_full_ft200；B：inductive9_rotate_mix。两者预训练 checkpoint 不同，B 还使用旋转增强及梯度累积，因此不可将差异全归于 batch size。', '',
           '| 数据集 | Head | 组 | Batch | 累积 | 旋转概率 | 训练样本 | 训练 | 验证 | 训练+验证 | 整轮 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summaries[1:]:
        group = 'A' if row['group'] == GROUPS[0] else 'B'
        md.append(f"| {row['dataset']} | {row['head']} | {group} | {row['batch_size']} | {row['accumulation']} | {row['rotation_probability']} | {row['samples']:,} | {row['train_seconds']:.2f} | {row['val_seconds']:.2f} | {row['train_through_val_seconds']:.2f} | {row['whole_epoch_seconds']:.2f} |")
    md += ['', '## 复现与限制', '',
           '所有选取的日志 world_size 均为 1。历史运行可能受到共享 GPU、并发任务、缓存状态影响；硬件元数据按 run 记录在 CSV，缺失时不推定型号。DiffLoss 验证包含采样推理，具体采样配置见各 run 的 config.yaml。',
           '这里的 DiffLoss 使用各运行嵌入配置；prediction_type 和 sampling_steps 已导出至 CSV，不能将其耗时套用到其他采样步数。训练+验证和整轮的统计 epoch 集合不同，均值偶尔可能出现整轮略小于训练+验证。',
           '当前无 /dev/nvidia*，无法补跑 batch 16/32/64/128/256/512 的同硬件、同配置测速。现有全量下游表覆盖 128/256/512；few-shot 的 16/32 不混入全量训练对比。',
           '运行：`PYTHONPATH=src /home/nvme03/hhfeng/miniconda3/envs/meshpred/bin/python scripts/summarize_bpd_epoch_timing.py`。',
           '汇总 CSV 和 JSON 包含配置路径、日志来源、硬件、参数量、耗时及标准差；`_epochs.csv` 保留全部逐轮起止时间。', '', '## 日志来源', '']
    for row in summaries:
        md.append(f"- {row['dataset']} / {row['head']} / batch {row['batch_size']}：`{row['log']}`")
    PREFIX.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(f'Wrote {len(summaries)} runs, {len(all_epochs)} epochs; all parameter counts and epoch counts verified.')
    for row in summaries:
        if row['dataset'] in ['BRepPreDiff', '九源 train-only']:
            print({k:row[k] for k in ['dataset','head','batch_size','parameters','train_seconds','val_seconds','whole_epoch_seconds']})


if __name__ == '__main__':
    main()
