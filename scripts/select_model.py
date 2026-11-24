"""Выбирает модель по test macro-F1 и задержке не больше 2× baseline."""

import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.training import MLRUNS, SUMMARY, read_json, write_json

BASELINE = "exp1_mobilenet_frozen"
MODEL_NAME = "plant-disease-clf"
BEST_PATH = ROOT / "artifacts" / "best_model.pt"


def main():
    payload = read_json(SUMMARY)
    runs = {item["name"]: item for item in payload["runs"]}
    expected = {
        "exp1_mobilenet_frozen",
        "exp2_mobilenet_finetune",
        "exp3_efficientnet_early_stop",
    }
    if set(runs) != expected:
        sys.exit(f"в summary должны быть три опыта, сейчас: {sorted(runs)}")

    baseline = runs[BASELINE]
    limit = baseline["latency_ms_per_image"] * 2
    allowed = [
        item for item in runs.values() if item["latency_ms_per_image"] <= limit
    ]
    chosen = max(allowed, key=lambda item: item["test_macro_f1"])
    reason = (
        f"Лимит задержки {limit:.3f} мс/снимок — это 2× baseline "
        f"({baseline['latency_ms_per_image']:.3f}). "
        f"Среди моделей внутри лимита максимальный test macro-F1 "
        f"у {chosen['name']}: {chosen['test_macro_f1']:.4f}."
    )
    payload["chosen_reason"] = reason
    payload["chosen_run_id"] = chosen["run_id"]
    payload["latency_limit_ms"] = limit
    write_json(SUMMARY, payload)

    source = ROOT / chosen["checkpoint"]
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, BEST_PATH)
    dvc = Path(sys.executable).with_name("dvc.exe")
    subprocess.run([str(dvc), "add", str(BEST_PATH)], cwd=ROOT, check=True)
    subprocess.run([str(dvc), "push", str(BEST_PATH) + ".dvc"], cwd=ROOT, check=True)

    mlflow.set_tracking_uri(MLRUNS.as_uri())
    client = MlflowClient()
    for item in runs.values():
        if item["run_id"] == chosen["run_id"]:
            client.set_tag(item["run_id"], "production", "true")
        else:
            client.set_tag(item["run_id"], "production", "false")
    version = mlflow.register_model(f"runs:/{chosen['run_id']}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "production", "true")
    client.set_registered_model_tag(MODEL_NAME, "production_run_id", chosen["run_id"])
    print(reason)
    print(f"run_id={chosen['run_id']}")
    print(f"checkpoint={BEST_PATH}")


if __name__ == "__main__":
    main()
