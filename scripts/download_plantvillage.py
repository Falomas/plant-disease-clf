"""Скачивает цветной PlantVillage и копирует 8 нужных классов в data/filtered."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.classes import CLASS_FOLDERS, SOURCE_REPO, SOURCE_VARIANT

FILTERED = ROOT / "data" / "filtered"
CLONE_DIR = ROOT / "data" / "_upstream"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def run(cmd):
    # git сам печатает прогресс. Ошибка клонирования не должна молча пройти.
    subprocess.run(cmd, check=True)


def remove_tree(path):
    # Git на Windows помечает pack-файлы только для чтения.
    def onerror(func, failed, _exc):
        os.chmod(failed, stat.S_IWRITE)
        func(failed)

    shutil.rmtree(path, onerror=onerror)


def main():
    FILTERED.mkdir(parents=True, exist_ok=True)
    if CLONE_DIR.exists():
        remove_tree(CLONE_DIR)

    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            SOURCE_REPO,
            str(CLONE_DIR),
        ]
    )
    paths = [f"{SOURCE_VARIANT}/{name}" for name in CLASS_FOLDERS]
    run(["git", "-C", str(CLONE_DIR), "sparse-checkout", "set", "--cone", *paths])

    commit = subprocess.check_output(
        ["git", "-C", str(CLONE_DIR), "rev-parse", "HEAD"], text=True
    ).strip()

    copied = 0
    for name in CLASS_FOLDERS:
        src = CLONE_DIR / SOURCE_VARIANT / name
        if not src.is_dir():
            sys.exit(f"нет папки класса: {src}")
        dst = FILTERED / name
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        files = [
            p
            for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not files:
            sys.exit(f"в классе нет изображений: {name}")
        for src_file in files:
            shutil.copy2(src_file, dst / src_file.name)
        copied += len(files)
        print(f"{name}: {len(files)}")

    (FILTERED / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    remove_tree(CLONE_DIR)
    print(f"всего файлов: {copied}")
    print(f"коммит источника: {commit}")


if __name__ == "__main__":
    main()
