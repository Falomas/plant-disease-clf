"""HTTP-инференс. Препроцессинг тот же, что на test итогового опыта."""

from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from plant_disease.classes import CLASS_FOLDERS
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


def _softmax(logits):
    shifted = logits - logits.max()
    exp = torch.exp(torch.as_tensor(shifted, dtype=torch.float32))
    return exp / exp.sum()


def load_onnx(path: Path):
    # В контейнере инференс идёт через onnxruntime, не через PyTorch.
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.get_outputs()[0]
    if output.shape[-1] not in (None, len(CLASS_FOLDERS)):
        raise RuntimeError("у ONNX-модели неожиданное число классов")
    return session, list(CLASS_FOLDERS)


def load_model(path: Path):
    # Локальные тесты остаются на .pt. Контейнер передаёт .onnx.
    if path.suffix.lower() == ".onnx":
        return load_onnx(path)
    return load_checkpoint(path)


def create_app() -> FastAPI:
    config = load_inference_config()
    checkpoint = Path(config.model_artifact_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"нет файла модели: {checkpoint.name}")
    model, classes = load_model(checkpoint)
    transform = build_transform(config.preprocess_version)
    app = FastAPI(title="Классификация заболеваний растений")
    app.state.model = model
    app.state.classes = classes
    app.state.transform = transform
    app.state.config = config

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        # 422 остаётся ответом схемы. Остальное не уходит клиенту как traceback.
        if isinstance(exc, RequestValidationError):
            return await request_validation_exception_handler(request, exc)
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса."})

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=config.model_version,
            preprocess_version=config.preprocess_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, file: UploadFile = File(...)):
        # Запрос только читает модель. Веса и режим eval не меняются.
        limit = config.server.max_upload_bytes
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > limit + 65536:
            raise HTTPException(status_code=413, detail="Файл больше допустимого размера.")
        raw = await file.read()
        if len(raw) > limit:
            raise HTTPException(status_code=413, detail="Файл больше допустимого размера.")
        content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
        if content_type and not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Неверный формат изображения.")
        if not raw:
            raise HTTPException(status_code=400, detail="Неверный формат изображения.")
        try:
            image = Image.open(BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Неверный формат изображения.")
        features = transform(image)
        if checkpoint.suffix.lower() == ".onnx":
            # Сессия onnxruntime только читает граф, веса не меняются.
            logits = app.state.model.run(
                ["logits"], {"input": features.unsqueeze(0).numpy()}
            )[0][0]
            probs = _softmax(logits)
            scores = {
                name: float(probs[index].item())
                for index, name in enumerate(app.state.classes)
            }
        else:
            batch = features.unsqueeze(0)
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
