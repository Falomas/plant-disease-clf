"""Выбирает модель по val macro-F1 и задержке не больше 2× baseline."""

import os

# Локальный каталог mlruns. MLflow 3 иначе отказывается от файлового backend.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plant_disease.training import MLRUNS, SUMMARY, read_json, write_json

BASELINE = "exp1_mobilenet_frozen"
MODEL_NAME = "plant-disease-clf"
BEST_PATH = ROOT / "artifacts" / "best_model.pt"
EXPORT = ROOT / "experiments" / "mlflow_export.json"
RULE = (
    "Максимальный macro-F1 на val среди моделей, "
    "чья средняя задержка не больше 2× baseline."
)


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
    if not allowed:
        sys.exit("ни один опыт не уложился в лимит задержки")
    chosen = max(allowed, key=lambda item: item["val_macro_f1"])
    comparison = []
    export_runs = []
    for item in sorted(runs.values(), key=lambda row: row["name"]):
        production = item["run_id"] == chosen["run_id"]
        comparison.append(
            {
                "latency_ms_per_image": item["latency_ms_per_image"],
                "name": item["name"],
                "run_id": item["run_id"],
                "test_accuracy": item["test_accuracy"],
                "test_macro_f1": item["test_macro_f1"],
                "val_macro_f1": item["val_macro_f1"],
                "within_latency_limit": item["latency_ms_per_image"] <= limit,
            }
        )
        export_runs.append(
            {
                "latency_ms_per_image": item["latency_ms_per_image"],
                "name": item["name"],
                "production": production,
                "run_id": item["run_id"],
                "test_accuracy": item["test_accuracy"],
                "test_macro_f1": item["test_macro_f1"],
                "val_macro_f1": item["val_macro_f1"],
            }
        )
    lines = [
        (
            f"{item['name']}: val macro-F1 {item['val_macro_f1']:.4f}, "
            f"задержка {item['latency_ms_per_image']:.3f} мс/снимок."
        )
        for item in comparison
    ]
    reason = (
        f"Лимит {limit:.3f} мс/снимок — это 2× baseline "
        f"({baseline['latency_ms_per_image']:.3f}). "
        + " ".join(lines)
        + f" Максимальный val macro-F1 у {chosen['name']}."
    )
    payload["chosen_reason"] = reason
    payload["chosen_run_id"] = chosen["run_id"]
    payload["comparison"] = comparison
    payload["latency_limit_ms"] = limit
    payload["rule"] = RULE
    write_json(SUMMARY, payload)
    write_json(
        EXPORT,
        {
            "model_name": MODEL_NAME,
            "production_run_id": chosen["run_id"],
            "rule": RULE,
            "runs": export_runs,
        },
    )

    source = ROOT / chosen["checkpoint"]
    if source.is_file():
        BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, BEST_PATH)
        track_checkpoint(BEST_PATH)
    else:
        print(f"чекпоинт не найден, копия не делалась: {source.name}")

    tag_production(chosen["run_id"], [item["run_id"] for item in runs.values()])
    print(reason)
    print(f"run_id={chosen['run_id']}")


def track_checkpoint(path: Path) -> None:
    # На Linux это команда dvc из PATH, не dvc.exe.
    dvc = shutil.which("dvc")
    if not dvc:
        print("dvc не найден в PATH, указатель не обновлялся")
        return
    subprocess.run([dvc, "add", str(path)], cwd=ROOT, check=True)
    remotes = subprocess.run(
        [dvc, "remote", "list"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if not remotes.stdout.strip():
        print("DVC remote не задан, push пропущен")
        return
    subprocess.run([dvc, "push", str(path) + ".dvc"], cwd=ROOT, check=True)


def tag_production(chosen_run_id: str, run_ids: list[str]) -> None:
    if not MLRUNS.is_dir():
        print("каталога mlruns нет, тег production только в experiments/mlflow_export.json")
        return
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(MLRUNS.as_uri())
    client = MlflowClient()
    for run_id in run_ids:
        client.set_tag(run_id, "production", "true" if run_id == chosen_run_id else "false")
    version = mlflow.register_model(f"runs:/{chosen_run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, version.version, "production", "true")
    client.set_registered_model_tag(MODEL_NAME, "production_run_id", chosen_run_id)


if __name__ == "__main__":
    main()
