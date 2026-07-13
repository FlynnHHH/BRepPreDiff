from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PACKAGE_NAME = "Blendit-Windows-Inference"
ASSET_FILES = (
    "environment-windows.yml",
    "install_env.bat",
    "run_inference.bat",
    "README_zh-CN.md",
)


def _latest_checkpoint(repo_root: Path) -> Path:
    candidates = list((repo_root / "runs" / "finetune").glob("*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No finetune best.pt checkpoint was found. Pass one explicitly with --checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_package(repo_root: Path, checkpoint: Path, output_dir: Path) -> tuple[Path, Path]:
    asset_dir = repo_root / "tools" / "windows_inference"
    source_dir = repo_root / "src" / "blendit"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing Blendit source directory: {source_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    for filename in (*ASSET_FILES, "launcher.py"):
        if not (asset_dir / filename).is_file():
            raise FileNotFoundError(f"Missing Windows package asset: {asset_dir / filename}")

    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = output_dir / PACKAGE_NAME
    if package_dir.exists():
        shutil.rmtree(package_dir)
    app_dir = package_dir / "app"
    model_dir = app_dir / "model"
    model_dir.mkdir(parents=True)

    for filename in ASSET_FILES:
        shutil.copy2(asset_dir / filename, package_dir / filename)
    shutil.copy2(asset_dir / "launcher.py", app_dir / "launcher.py")
    shutil.copytree(
        source_dir,
        app_dir / "src" / "blendit",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    packaged_checkpoint = model_dir / "best.pt"
    shutil.copy2(checkpoint, packaged_checkpoint)

    checkpoint_hash = _sha256(packaged_checkpoint)
    info = (
        f"package={PACKAGE_NAME}\n"
        f"source_checkpoint={checkpoint.relative_to(repo_root) if checkpoint.is_relative_to(repo_root) else checkpoint}\n"
        f"checkpoint_sha256={checkpoint_hash}\n"
        "seg_labels=NonTransition:0,VBF:6,EBF:4\n"
    )
    (package_dir / "PACKAGE_INFO.txt").write_text(info, encoding="utf-8")

    archive_path = Path(
        shutil.make_archive(
            str(output_dir / PACKAGE_NAME),
            "zip",
            root_dir=output_dir,
            base_dir=PACKAGE_NAME,
        )
    )
    archive_hash_path = archive_path.with_suffix(f"{archive_path.suffix}.sha256")
    archive_hash_path.write_text(f"{_sha256(archive_path)}  {archive_path.name}\n", encoding="ascii")
    return archive_path, archive_hash_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Blendit source-based Windows inference ZIP.")
    parser.add_argument("--checkpoint", default=None, help="Finetune best.pt to include. Defaults to the newest best.pt.")
    parser.add_argument("--output-dir", default="dist", help="Package output directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else _latest_checkpoint(repo_root)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    archive_path, hash_path = build_package(repo_root, checkpoint, output_dir.resolve())
    print(f"Built: {archive_path}")
    print(f"SHA256: {hash_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
