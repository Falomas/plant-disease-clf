"""Строит стратифицированный сплит 70/15/15 и записывает манифест."""

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.classes import (
    CLASS_FOLDERS,
    SOURCE_REPO,
    SOURCE_VARIANT,
    SPLIT_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)

FILTERED = ROOT / "data" / "filtered"
MANIFEST = ROOT / "splits" / "split_manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_split(n_files):
    # int() отрезает хвост. Он остаётся в test, чтобы сумма была ровно n_files.
    n_train = int(n_files * TRAIN_RATIO)
    n_val = int(n_files * VAL_RATIO)
    n_test = n_files - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        sys.exit(f"слишком мало файлов для трёх частей: {n_files}")
    return n_train, n_val, n_test


def main():
    source_file = FILTERED / "SOURCE_COMMIT.txt"
    if not source_file.is_file():
        sys.exit("сначала запустите scripts/download_plantvillage.py")
    source_commit = source_file.read_text(encoding="utf-8").strip()

    # Один генератор и сортировка до shuffle: тот же seed даёт тот же сплит.
    rng = random.Random(SPLIT_SEED)
    records = []
    class_counts = {}

    for class_name in CLASS_FOLDERS:
        folder = FILTERED / class_name
        files = sorted(
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            sys.exit(f"нет изображений класса {class_name}")
        rng.shuffle(files)
        n_train, n_val, n_test = assign_split(len(files))
        class_counts[class_name] = {
            "total": len(files),
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }
        for index, path in enumerate(files):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            records.append(
                {
                    "path": f"{class_name}/{path.name}",
                    "class_name": class_name,
                    "split": split,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        print(
            f"{class_name}: total={len(files)} train={n_train} val={n_val} test={n_test}"
        )

    records.sort(key=lambda item: item["path"])
    payload = {
        "dataset": "plantvillage-color-8class",
        "source_repo": SOURCE_REPO,
        "source_commit": source_commit,
        "source_variant": SOURCE_VARIANT,
        "split_seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio_remainder": True,
        "rounding": "n_train=int(n*0.70); n_val=int(n*0.15); n_test=n-n_train-n_val",
        "test_usage": "final_evaluation_only",
        "class_counts": class_counts,
        "files": records,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_bytes(text.encode("utf-8"))
    print(f"манифест: {MANIFEST}")
    print(f"файлов в манифесте: {len(records)}")


if __name__ == "__main__":
    main()
