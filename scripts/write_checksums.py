"""Пишет sha256 манифеста сплита и md5 каталога, который отслеживает DVC."""

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "splits" / "split_manifest.json"
DVC_FILE = ROOT / "data" / "filtered.dvc"
OUTPUT = ROOT / "splits" / "checksums.json"


def main():
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    pointer = yaml.safe_load(DVC_FILE.read_text(encoding="utf-8"))
    out = pointer["outs"][0]
    payload = {
        "filtered_dvc_md5": out["md5"],
        "filtered_nfiles": out["nfiles"],
        "filtered_size_bytes": out["size"],
        "split_manifest_path": "splits/split_manifest.json",
        "split_manifest_sha256": manifest_sha,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    OUTPUT.write_bytes(text.encode("utf-8"))
    print(text, end="")


if __name__ == "__main__":
    main()
