# Зафиксированный снимок python:3.11-slim-bookworm, не тег latest.
FROM python:3.11-slim-bookworm@sha256:c16857aad54501ae4802f5b161f7c85ddbc7da3aae2de469201049149ee5204b AS builder

ENV POETRY_VERSION=2.5.2 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"
COPY pyproject.toml poetry.lock ./
# Зависимости только из lock-файла, без установки самого проекта.
RUN poetry install --no-root --only main

FROM python:3.11-slim-bookworm@sha256:c16857aad54501ae4802f5b161f7c85ddbc7da3aae2de469201049149ee5204b AS runtime

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser plant_disease ./plant_disease
COPY --chown=appuser:appuser configs ./configs

# Публичный ONNX с Google Drive, около 16 МБ. В git этого файла нет.
RUN mkdir -p /app/artifacts \
    && python -c "import os, urllib.request; p='/app/artifacts/best_model.onnx'; urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1sspa7N29JOkXcEMpzGhPy10Wj9GGMWmP', p); assert os.path.getsize(p) > 1000000, p" \
    && chown -R appuser:appuser /app/artifacts

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ARTIFACT_PATH=/app/artifacts/best_model.onnx

USER appuser
EXPOSE 8000
# 0.0.0.0, чтобы проброс порта с хоста доходил до процесса.
CMD ["python", "-m", "uvicorn", "plant_disease.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
