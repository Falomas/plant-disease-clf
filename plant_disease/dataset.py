"""Чтение манифеста и картинок. Test-загрузчик отдельно от train."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from plant_disease.classes import CLASS_FOLDERS


class LeafDataset(Dataset):
    def __init__(self, records, image_root, transform):
        self.records = records
        self.image_root = Path(image_root)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_FOLDERS)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.image_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"нет снимка: {path}")
        image = Image.open(path).convert("RGB")
        label = self.class_to_idx[record["class_name"]]
        return self.transform(image), label


def load_records(manifest_path, split):
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = [item for item in payload["files"] if item["split"] == split]
    # Стабильный порядок: от него зависят seed аугментаций и 100 примеров задержки.
    records.sort(key=lambda item: item["path"])
    if not records:
        raise RuntimeError(f"в манифесте нет части {split}")
    return records
