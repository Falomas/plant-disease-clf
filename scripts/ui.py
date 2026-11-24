"""Интерфейс Gradio. Своей модели нет: снимок уходит в POST /predict."""

import os
from pathlib import Path

import gradio as gr
import requests


def _api_url():
    # Адрес не зашит: его задают в окружении.
    return os.environ.get("API_URL", "").strip().rstrip("/")


def _error_text(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return f"Ошибка сервиса: {detail}"
    if isinstance(detail, list):
        return "Сервис не принял запрос. Нужен файл изображения."
    return f"Сервис вернул ошибку {response.status_code}."


def predict(image_path):
    api_url = _api_url()
    if not api_url:
        return "Не задан адрес API. Укажите переменную API_URL."
    if not image_path:
        return "Сначала выберите изображение листа."
    path = Path(image_path)
    try:
        with path.open("rb") as handle:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": (path.name, handle, "image/jpeg")},
                timeout=30,
            )
    except requests.RequestException:
        return f"Не удалось связаться с сервисом по адресу {api_url}. Сначала запустите API."
    if response.status_code != 200:
        return _error_text(response)
    body = response.json()
    percent = float(body["confidence"]) * 100
    return (
        f"Класс: {body['class']}\n"
        f"Уверенность: {percent:.1f}%\n"
        f"Версия модели: {body['model_version']}"
    )


def main():
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Изображение листа"),
        outputs=gr.Textbox(label="Результат"),
        title="Классификация заболеваний растений",
        description="Снимок отправляется в POST /predict. В интерфейсе нет своей модели.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
