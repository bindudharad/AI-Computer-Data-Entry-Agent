#!/usr/bin/env python3
"""Package ATLAS AI + MPF plugin into a clean ZIP.

Excludes virtualenvs, caches, logs, screenshots, debug captures and previous
archives. Run from the project root:

    python package_zip.py
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ATLAS-AI-MPF-v1.4.0.zip"

EXCLUDE_DIRS = {
    ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "__pycache__", "logs", "screenshots", "debug", "reports", "profiles",
}
EXCLUDE_SUFFIXES = {".pyc", ".zip", ".log", ".db"}
EXCLUDE_FILES = {"memory.db"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path == OUT:
                continue
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if path.name in EXCLUDE_FILES:
                continue
            files.append(path)
    return sorted(files)


def main() -> None:
    files = iter_files()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))
    size = OUT.stat().st_size
    print(f"{OUT.name}: {len(files)} files, {size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
