"""Контракт /health и /predict. Препроцессинг проверяется без загрузки модели."""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from plant_disease.preprocess import build_transform
from plant_disease.service import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "late_blight.jpg"
EXPECTED_CLASS = "Tomato___Late_blight"


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    monkeypatch_module.setenv("MODEL_ARTIFACT_PATH", "artifacts/best_model.pt")
    return TestClient(create_app())


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_fixture_class(client):
    payload = _predict(client, FIXTURE.read_bytes(), "late_blight.jpg", "image/jpeg")
    assert payload["class"] == EXPECTED_CLASS
    # Повтор не меняет ответ: веса и режим модели те же.
    again = _predict(client, FIXTURE.read_bytes(), "late_blight.jpg", "image/jpeg")
    assert again == payload


def test_predict_rejects_bad_format(client):
    response = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Неверный формат изображения."


def test_predict_response_schema(client):
    payload = _predict(client, FIXTURE.read_bytes(), "late_blight.jpg", "image/jpeg")
    assert set(payload) == {
        "class",
        "confidence",
        "probabilities",
        "model_version",
        "preprocess_version",
    }
    assert payload["preprocess_version"] == "v1_baseline"
    assert payload["model_version"] == "5fd5ed281bce46088edaef1277d0d4fd"
    assert isinstance(payload["confidence"], float)
    assert payload["class"] in payload["probabilities"]
    assert abs(sum(payload["probabilities"].values()) - 1) < 1e-4


def test_preprocess_without_model():
    transform = build_transform("v1_baseline")
    image = Image.new("RGB", (32, 40), (12, 80, 30))
    first = transform(image)
    second = transform(image)
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def _predict(client, raw, name, content_type):
    response = client.post(
        "/predict",
        files={"file": (name, raw, content_type)},
    )
    assert response.status_code == 200
    return response.json()
