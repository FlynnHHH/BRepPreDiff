# Contributing to BRepPreDiff

Thank you for improving BRepPreDiff.

## Development setup

```bash
conda env create -f environment.yml
conda activate brepprediff
python -m pip install -e ".[dev]"
```

## Before opening a pull request

1. Keep data preparation settings in `data/*.yaml` and training settings in `configs/*.yaml`.
2. Do not commit datasets, caches, checkpoints, run logs, visualization results, or release archives.
3. Add or update tests for behavior changes.
4. Run `pytest -q` from the repository root.
5. Keep changes focused and explain configuration or checkpoint compatibility changes in the PR.

## Reporting bugs

Include the operating system, Python/PyTorch/pythonocc versions, the command that failed, the
relevant configuration, and a minimal traceback. Do not attach private CAD data; provide a small
reproducible model when possible.
