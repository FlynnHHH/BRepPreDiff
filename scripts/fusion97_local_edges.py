"""Build separate local-normal edge caches, preserving faces and split membership."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import time

import numpy as np

from brepprediff.brep.occ_extractor import OccBRepExtractor
from brepprediff.config import load_experiment_config
from brepprediff.data.graph import save_graph_npz

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/nvme03/hhfeng/fusion360segmentationdataset/s2.0.0/breps')


def build_one(job):
    source, target = map(Path, job)
    if target.exists():
        return {'id': source.stem, 'skipped': True}
    try:
        config = load_experiment_config(ROOT / 'configs/finetune_joint_fusion360seg_mlp.yaml',
                                       overrides=['brep.local_edge_normals=true'])
        extractor = OccBRepExtractor(config)
        sample = source.stem.rsplit('_', 1)[0]
        candidates = [RAW / 'step' / (sample + suffix) for suffix in ('.stp', '.step', '.STEP', '.STP')]
        step = next(p for p in candidates if p.exists())
        with np.load(source, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        shape = extractor._read_step(step)
        faces = extractor._indexed_faces(shape)
        assert extractor._map_size(faces) == arrays['face_cont'].shape[0], 'face count changed'
        labels = np.loadtxt(RAW / 'seg' / (sample + '.seg'), dtype=np.int64).reshape(-1)
        assert np.array_equal(labels, arrays['labels']), 'cached/raw label mismatch'
        edges = extractor._edge_arrays(shape, faces, arrays['face_cont'][:, 4:7])
        assert np.array_equal(edges['edge_index'], arrays['edge_index']), 'edge ordering changed'
        # Existing edge sampling is retained bit-for-bit; only the two relation
        # channels and the relation category change.
        arrays['edge_cont'][:, 1:3] = edges['edge_cont'][:, 1:3]
        arrays['edge_relation'] = edges['edge_relation']
        arrays['local_edge_version'] = np.array(1, dtype=np.int64)
        save_graph_npz(target, arrays)
        return {'id': sample, 'success': extractor.local_edge_success, 'fallback': extractor.local_edge_fallback}
    except Exception as error:
        return {'id': source.stem, 'error': repr(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val', 'test'])
    args = parser.parse_args()
    sources = {p.stem.rsplit('_', 1)[0]: p for p in (ROOT / 'cache/features/fusion360seg').glob('*.npz')}
    ids = []
    for split in args.splits:
        ids += [Path(x).stem for x in (ROOT / f'data/splits/fusion360seg_{split}.txt').read_text().splitlines() if x]
    if args.limit:
        ids = ids[:args.limit]
    destination = ROOT / 'cache/features/fusion360seg_localedge_v1'
    destination.mkdir(parents=True, exist_ok=True)
    jobs = [(str(sources[sample]), str(destination / sources[sample].name)) for sample in ids]
    started = time.time()
    report = ROOT / 'runs/fusion97/localedge_build'
    report.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(build_one, jobs, chunksize=8):
            results.append(result)
            if len(results) % 100 == 0:
                print(json.dumps({'done': len(results), 'total': len(jobs), 'seconds': time.time()-started}), flush=True)
    summary = {'count': len(results), 'seconds': time.time()-started,
               'success_edges': sum(r.get('success', 0) for r in results),
               'fallback_edges': sum(r.get('fallback', 0) for r in results),
               'errors': [r for r in results if 'error' in r]}
    (report / ('_'.join(args.splits) + f'_{len(jobs)}.json')).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
