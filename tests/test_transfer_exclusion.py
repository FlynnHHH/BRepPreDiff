import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('transfer_runner', Path(__file__).parents[1] / 'scripts/run_lodo_transfer.py')
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)


def test_geometry_fingerprint_ignores_labels_but_detects_geometry_changes(tmp_path):
    arrays = {key: np.arange(6, dtype=np.float32).reshape(2, 3) for key in _runner.KEYS}
    arrays['labels'] = np.array([0, 1])
    a = tmp_path / 'a.npz'; b = tmp_path / 'b.npz'; c = tmp_path / 'c.npz'
    np.savez(a, **arrays)
    arrays['labels'] = np.array([4, 5])
    np.savez_compressed(b, **arrays)
    assert _runner.fingerprint(a)[1] == _runner.fingerprint(b)[1]
    arrays['face_cont'] = arrays['face_cont'] + 1
    np.savez(c, **arrays)
    assert _runner.fingerprint(a)[1] != _runner.fingerprint(c)[1]
