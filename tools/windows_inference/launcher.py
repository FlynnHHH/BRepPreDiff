from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    app_dir = Path(__file__).resolve().parent
    source_dir = app_dir / "src"
    checkpoint = app_dir / "model" / "best.pt"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"程序文件不完整，缺少目录：{source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"程序文件不完整，缺少模型：{checkpoint}")

    sys.path.insert(0, str(source_dir))
    from brepprediff.inference.step_to_seg import main as infer_main

    return infer_main([*sys.argv[1:], "--checkpoint", str(checkpoint)])


if __name__ == "__main__":
    raise SystemExit(main())
