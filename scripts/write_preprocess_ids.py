"""Записывает JSON спецификаций и их sha256. Это идентификаторы версий."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.preprocess import SPECS, canonical_bytes, spec_sha256

OUT_DIR = ROOT / "splits"


def main():
    versions = {}
    for name in ("v1_baseline", "v2_augmented"):
        path = OUT_DIR / f"preprocess_{name}.json"
        path.write_bytes(canonical_bytes(SPECS[name]))
        versions[name] = {
            "path": f"splits/{path.name}",
            "sha256": spec_sha256(name),
        }
        print(f"{name} {versions[name]['sha256']}")
    text = json.dumps(versions, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (OUT_DIR / "preprocess_versions.json").write_bytes(text.encode("utf-8"))


if __name__ == "__main__":
    main()
