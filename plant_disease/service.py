"""HTTP-инференс. Препроцессинг тот же, что на test итогового опыта."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from plant_disease.config import load_inference_config
from plant_disease.models import build_model
from plant_disease.preprocess import build_transform


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    preprocess_version: str


class PredictResponse(BaseModel):
    # В JSON поле называется class: это имя класса.
    class_name: str = Field(serialization_alias="class")
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    preprocess_version: str


def load_checkpoint(path: Path):
    # Свои веса подставляются сразу, без докачки ImageNet.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(blob["classes"])
    model = build_model(
        blob["architecture"],
        len(classes),
        blob["unfreeze_last_blocks"],
        pretrained=False,
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, classes


def create_app() -> FastAPI:
    config = load_inference_config()
    checkpoint = Path(config.model_artifact_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"нет файла модели: {checkpoint.name}")
    model, classes = load_checkpoint(checkpoint)
    transform = build_transform(config.preprocess_version)
    app = FastAPI(title="Классификация заболеваний растений")
    app.state.model = model
    app.state.classes = classes
    app.state.transform = transform
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=config.model_version,
            preprocess_version=config.preprocess_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        # Запрос только читает модель. Веса и режим eval не меняются.
        raw = await file.read()
        image = Image.open(BytesIO(raw)).convert("RGB")
        batch = transform(image).unsqueeze(0)
        with torch.inference_mode():
            probs = torch.softmax(app.state.model(batch), dim=1)[0]
        scores = {
            name: float(probs[index].item())
            for index, name in enumerate(app.state.classes)
        }
        label = max(scores, key=scores.get)
        return PredictResponse(
            class_name=label,
            confidence=scores[label],
            probabilities=scores,
            model_version=config.model_version,
            preprocess_version=config.preprocess_version,
        )

    return app
