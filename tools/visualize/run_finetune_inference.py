from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "brepprediff").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(
        "Could not find the BRepPreDiff repo root. "
        "Run this script from inside the extracted viewer directory under the BRepPreDiff repository."
    )


def main() -> None:
    viewer_dir = Path(__file__).resolve().parent
    repo_root = _find_repo_root(viewer_dir)
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in sys.argv):
        sys.argv.extend(["--output-dir", str(viewer_dir / "results")])

    from brepprediff.inference.finetune_visualize import main as infer_main

    infer_main()


if __name__ == "__main__":
    main()
