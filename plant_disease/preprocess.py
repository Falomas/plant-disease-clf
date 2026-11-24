"""Две версии препроцессинга. Аугментации не сохраняются отдельным датасетом."""

import hashlib
import json

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Версия A: фиксированный resize и нормализация.
# Версия B: те же 224 и ImageNet, но перед этим случайные аугментации.
SPECS = {
    "v1_baseline": {
        "id": "v1_baseline",
        "steps": [
            {"name": "resize", "size": 224},
            {"name": "to_tensor"},
            {"mean": IMAGENET_MEAN, "name": "imagenet_normalize", "std": IMAGENET_STD},
        ],
    },
    "v2_augmented": {
        "augment_seed": 42,
        "base": "v1_baseline",
        "id": "v2_augmented",
        "steps": [
            # scale не даёт вырезать крошечный кусок листа.
            {"name": "random_resized_crop", "scale": [0.8, 1.0], "size": 224},
            {"name": "horizontal_flip", "p": 0.5},
            {"degrees": 15, "name": "rotation"},
            {
                "brightness": 0.2,
                "contrast": 0.2,
                "hue": 0.05,
                "name": "color_jitter",
                "saturation": 0.2,
            },
            {"name": "to_tensor"},
            {"mean": IMAGENET_MEAN, "name": "imagenet_normalize", "std": IMAGENET_STD},
        ],
    },
}


def canonical_bytes(spec):
    text = json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def spec_sha256(name):
    return hashlib.sha256(canonical_bytes(SPECS[name])).hexdigest()


def _step(step):
    name = step["name"]
    if name == "resize":
        return transforms.Resize((step["size"], step["size"]))
    if name == "random_resized_crop":
        return transforms.RandomResizedCrop(step["size"], scale=tuple(step["scale"]))
    if name == "horizontal_flip":
        return transforms.RandomHorizontalFlip(p=step["p"])
    if name == "rotation":
        # degrees=15 означает поворот от -15 до +15 градусов.
        return transforms.RandomRotation(degrees=step["degrees"])
    if name == "color_jitter":
        return transforms.ColorJitter(
            brightness=step["brightness"],
            contrast=step["contrast"],
            saturation=step["saturation"],
            hue=step["hue"],
        )
    if name == "to_tensor":
        return transforms.ToTensor()
    if name == "imagenet_normalize":
        return transforms.Normalize(mean=step["mean"], std=step["std"])
    raise ValueError(f"неизвестный шаг: {name}")


def build_transform(name):
    if name not in SPECS:
        raise ValueError(f"неизвестная версия: {name}")
    return transforms.Compose([_step(step) for step in SPECS[name]["steps"]])


def reset_seed(name):
    # Один seed на проход по файлам. Повтор того же порядка воспроизводит кадры.
    seed = SPECS[name].get("augment_seed")
    if seed is not None:
        torch.manual_seed(seed)
    return seed
