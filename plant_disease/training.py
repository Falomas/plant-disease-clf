"""Один проход обучения. Test вызывается один раз, после выбора чекпоинта по val."""

import copy
import json
import os
import subprocess
import time
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from plant_disease.classes import CLASS_FOLDERS
from plant_disease.config import ROOT, load_train_config
from plant_disease.dataset import LeafDataset, load_records
from plant_disease.metrics import confusion_update, empty_confusion, scores_from_confusion
from plant_disease.models import build_model, trainable_count
from plant_disease.preprocess import build_transform

# Локальный каталог. Он в .gitignore, сервер MLflow не нужен.
MLRUNS = ROOT / "mlruns"
SUMMARY = ROOT / "experiments" / "summary.json"
LATENCY_IMAGES = 100


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(text.encode("utf-8"))


def load_spec(name):
    specs = read_json(ROOT / "experiments" / "specs.json")
    if name not in specs:
        raise SystemExit(f"неизвестный опыт: {name}")
    return specs[name]


def make_loader(records, image_root, transform, batch_size, shuffle, seed):
    dataset = LeafDataset(records, image_root, transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def evaluate(model, loader, device, num_classes):
    # Общая функция для val и для единственного прохода по test.
    model.eval()
    matrix = empty_confusion(num_classes)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predicted = model(images).argmax(dim=1)
            confusion_update(matrix, labels, predicted)
    accuracy, macro_f1 = scores_from_confusion(matrix)
    return accuracy, macro_f1, matrix


def measure_latency_ms(model, records, image_root, transform, batch_size, device):
    # Одни и те же 100 снимков и тот же размер батча для всех опытов.
    sample = records[:LATENCY_IMAGES]
    loader = make_loader(sample, image_root, transform, batch_size, False, 0)
    model.eval()
    batches = [(images.to(device),) for images, _labels in loader]
    with torch.inference_mode():
        for (images,) in batches:
            model(images)
        started = time.perf_counter()
        seen = 0
        for (images,) in batches:
            model(images)
            seen += images.shape[0]
        elapsed = time.perf_counter() - started
    return elapsed / seen * 1000


def train_one_epoch(model, loader, optimizer, loss_fn, device, seed):
    model.train()
    # Замороженные блоки не обновляют свой BatchNorm.
    for module in model.features:
        if not any(param.requires_grad for param in module.parameters()):
            module.eval()
    # Seed эпохи делает аугментации повторяемыми, но разными между эпохами.
    torch.manual_seed(seed)
    total_loss = 0.0
    seen = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        seen += labels.shape[0]
    return total_loss / seen


def upsert_summary(entry):
    if SUMMARY.is_file():
        payload = read_json(SUMMARY)
    else:
        payload = {
            "chosen_reason": None,
            "chosen_run_id": None,
            "rule": (
                "Максимальный macro-F1 на test среди моделей, "
                "чья средняя задержка не больше 2× baseline."
            ),
            "runs": [],
        }
    payload["runs"] = [item for item in payload["runs"] if item["name"] != entry["name"]]
    payload["runs"].append(entry)
    payload["runs"].sort(key=lambda item: item["name"])
    write_json(SUMMARY, payload)


def run_experiment(name):
    # MLflow 3 прячет файловый backend. Нам нужен именно каталог mlruns.
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    spec = load_spec(name)
    config = load_train_config()
    device = torch.device("cpu")
    image_root = Path(config.data_dir) / config.data.image_subdir
    manifest = ROOT / config.data.split_manifest
    train_records = load_records(manifest, "train")
    val_records = load_records(manifest, "val")
    test_records = load_records(manifest, "test")

    train_tf = build_transform(spec["train_preprocess"])
    eval_tf = build_transform(config.augmentations.eval_version)
    train_loader = make_loader(
        train_records,
        image_root,
        train_tf,
        config.train.batch_size,
        True,
        config.data.split_seed,
    )
    val_loader = make_loader(
        val_records, image_root, eval_tf, config.train.batch_size, False, 0
    )
    test_loader = make_loader(
        test_records, image_root, eval_tf, config.train.batch_size, False, 0
    )

    model = build_model(
        spec["architecture"],
        config.model.num_classes,
        spec["unfreeze_last_blocks"],
    ).to(device)
    optimizer = torch.optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.train.learning_rate,
    )
    loss_fn = nn.CrossEntropyLoss()

    checksums = read_json(ROOT / "splits" / "checksums.json")
    versions = read_json(ROOT / "splits" / "preprocess_versions.json")
    sha = git_sha()
    note = (
        f"{spec['title']}. Потолок эпох {config.train.epochs} из-за CPU и pretrained-весов. "
        "Чекпоинт выбирается по val macro-F1. Test считается один раз в конце. "
        + (
            f"Early stopping, patience {config.train.early_stopping_patience}."
            if spec["early_stopping"]
            else "Early stopping выключен, идём до потолка эпох."
        )
    )

    mlflow.set_tracking_uri(MLRUNS.as_uri())
    mlflow.set_experiment("plant-disease")
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("mlflow.note.content", note)
        mlflow.log_params(
            {
                "architecture": spec["architecture"],
                "batch_size": config.train.batch_size,
                "early_stopping": spec["early_stopping"],
                "epochs_cap": config.train.epochs,
                "eval_preprocess": config.augmentations.eval_version,
                "filtered_dvc_md5": checksums["filtered_dvc_md5"],
                "git_sha": sha,
                "learning_rate": config.train.learning_rate,
                "patience": config.train.early_stopping_patience,
                "split_manifest_sha256": checksums["split_manifest_sha256"],
                "split_seed": config.data.split_seed,
                "train_preprocess": spec["train_preprocess"],
                "train_preprocess_sha256": versions[spec["train_preprocess"]]["sha256"],
                "trainable_params": trainable_count(model),
                "unfreeze_last_blocks": spec["unfreeze_last_blocks"],
            }
        )
        config_path = ROOT / "artifacts" / "runs" / f"{name}-config.json"
        write_json(config_path, config.model_dump())
        mlflow.log_artifact(str(config_path), artifact_path="config")

        best_f1 = -1.0
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale = 0
        for epoch in range(1, config.train.epochs + 1):
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                config.data.split_seed + epoch,
            )
            val_acc, val_f1, _val_cm = evaluate(
                model, val_loader, device, config.model.num_classes
            )
            mlflow.log_metrics(
                {"train_loss": loss, "val_accuracy": val_acc, "val_macro_f1": val_f1},
                step=epoch,
            )
            print(f"эпоха {epoch}: loss={loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
                if spec["early_stopping"] and stale >= config.train.early_stopping_patience:
                    print(f"early stopping на эпохе {epoch}")
                    break

        model.load_state_dict(best_state)
        test_acc, test_f1, test_cm = evaluate(
            model, test_loader, device, config.model.num_classes
        )
        latency_ms = measure_latency_ms(
            model,
            test_records,
            image_root,
            eval_tf,
            config.train.batch_size,
            device,
        )
        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_f1,
                "latency_ms_per_image": latency_ms,
                "test_accuracy": test_acc,
                "test_macro_f1": test_f1,
            }
        )
        matrix_path = ROOT / "artifacts" / "runs" / f"{name}-confusion.json"
        write_json(
            matrix_path,
            {
                "labels": CLASS_FOLDERS,
                "matrix": test_cm.tolist(),
            },
        )
        mlflow.log_artifact(str(matrix_path), artifact_path="metrics")
        ckpt_path = ROOT / "artifacts" / "runs" / f"{name}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": spec["architecture"],
                "best_epoch": best_epoch,
                "classes": CLASS_FOLDERS,
                "state_dict": best_state,
                "unfreeze_last_blocks": spec["unfreeze_last_blocks"],
            },
            ckpt_path,
        )
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")
        mlflow.pytorch.log_model(model, name="model")
        run_id = run.info.run_id

    entry = {
        "architecture": spec["architecture"],
        "best_epoch": best_epoch,
        "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
        "early_stopping": spec["early_stopping"],
        "eval_preprocess": config.augmentations.eval_version,
        "git_sha": sha,
        "latency_ms_per_image": latency_ms,
        "name": name,
        "run_id": run_id,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "title": spec["title"],
        "train_preprocess": spec["train_preprocess"],
        "unfreeze_last_blocks": spec["unfreeze_last_blocks"],
        "val_macro_f1": best_f1,
    }
    upsert_summary(entry)
    print(f"run_id={run_id} test_f1={test_f1:.4f} latency_ms={latency_ms:.3f}")
    return entry
