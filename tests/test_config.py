"""Train и inference читаются из YAML. Пустой обязательный параметр даёт короткую ошибку."""

import pytest

from plant_disease.config import ConfigError, load_inference_config, load_train_config

DATA_DIR_ERROR = (
    "Не задан обязательный параметр: DATA_DIR. "
    "Проверьте configs/train.yaml и переменную DATA_DIR."
)
MODEL_PATH_ERROR = (
    "Не задан обязательный параметр: MODEL_ARTIFACT_PATH. "
    "Проверьте configs/inference.yaml и переменную MODEL_ARTIFACT_PATH."
)


def test_train_and_inference_load_without_code_changes(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", "artifacts/model.onnx")
    train = load_train_config(env_file=None)
    inference = load_inference_config(env_file=None)
    assert train.data.split_seed == 42
    assert train.data.preprocess_versions == ["v1_baseline", "v2_augmented"]
    assert train.augmentations.train_version == "v2_augmented"
    assert train.augmentations.eval_version == "v1_baseline"
    assert inference.preprocess_version == "v2_augmented"
    assert inference.server.max_upload_bytes == 5_242_880


def test_empty_data_dir_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("DATA_DIR", " ")
    with pytest.raises(ConfigError) as caught:
        load_train_config(env_file=None)
    message = str(caught.value)
    assert message == DATA_DIR_ERROR
    assert "Traceback" not in message


def test_missing_model_path_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MODEL_ARTIFACT_PATH", raising=False)
    with pytest.raises(ConfigError) as caught:
        load_inference_config(env_file=None)
    message = str(caught.value)
    assert message == MODEL_PATH_ERROR
    assert "Traceback" not in message
