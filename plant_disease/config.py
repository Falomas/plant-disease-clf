"""Загрузка YAML-конфигурации. Пути к данным и модели берутся только из окружения."""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from plant_disease.classes import SPLIT_SEED
from plant_disease.preprocess import SPECS

ROOT = Path(__file__).resolve().parents[1]
TRAIN_YAML = ROOT / "configs" / "train.yaml"
INFERENCE_YAML = ROOT / "configs" / "inference.yaml"

# Имена полей и соответствующие переменные окружения.
ENV_NAMES = {
    "data_dir": "DATA_DIR",
    "model_artifact_path": "MODEL_ARTIFACT_PATH",
}


class ConfigError(Exception):
    """Короткая ошибка конфигурации без внутреннего traceback."""


class DataSection(BaseModel):
    split_seed: int
    split_manifest: str = Field(min_length=1)
    # Папка со снимками относительно DATA_DIR, без абсолютного пути.
    image_subdir: str = Field(min_length=1)
    preprocess_versions: list[str] = Field(min_length=1)


class ModelSection(BaseModel):
    architecture: str = Field(min_length=1)
    num_classes: int = Field(gt=0)
    image_size: int = Field(gt=0)


class TrainSection(BaseModel):
    # Потолок эпох. На CPU он небольшой, ранний останов смотрит на val.
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    early_stopping_patience: int = Field(gt=0)


class AugmentSection(BaseModel):
    train_version: str = Field(min_length=1)
    eval_version: str = Field(min_length=1)


class TrainConfig(BaseSettings):
    # Каталог снимков. В YAML его нет.
    data_dir: str = Field(min_length=1)
    data: DataSection
    model: ModelSection
    train: TrainSection
    augmentations: AugmentSection

    @field_validator("data_dir")
    @classmethod
    def _strip_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("пусто")
        return value


def _param_name(loc: tuple) -> str:
    parts = [str(part) for part in loc if part != ""]
    if len(parts) == 1 and parts[0] in ENV_NAMES:
        return ENV_NAMES[parts[0]]
    return ".".join(parts) if parts else "конфигурация"


def _display_path(path: Path) -> str:
    # В тексте ошибки не должно быть абсолютного пути этой машины.
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _config_error(exc: ValidationError, where: str) -> ConfigError:
    # Одно предложение на параметр, без цепочки pydantic.
    names = []
    for err in exc.errors():
        name = _param_name(err.get("loc", ()))
        if name not in names:
            names.append(name)
    listed = ", ".join(names)
    return ConfigError(f"Не задан обязательный параметр: {listed}. Проверьте {where}.")


def _with_yaml(model, yaml_path: Path, env_file: str | None):
    class Settings(model):
        model_config = SettingsConfigDict(
            env_file=env_file,
            env_file_encoding="utf-8",
            extra="ignore",
        )

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # Окружение важнее файла: путь нельзя зафиксировать в YAML.
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                YamlConfigSettingsSource(
                    settings_cls,
                    yaml_file=yaml_path,
                    yaml_file_encoding="utf-8",
                ),
            )

    return Settings


def _known_version(name: str) -> None:
    if name not in SPECS:
        raise ConfigError(
            f"Неизвестная версия препроцессинга: {name}. "
            f"Доступны: {', '.join(SPECS)}."
        )


def load_train_config(
    yaml_path: Path | None = None,
    env_file: str | None = ".env",
) -> TrainConfig:
    """Возвращает уже разобранный train-конфиг. Его можно отдать в MLflow."""
    path = Path(yaml_path) if yaml_path else TRAIN_YAML
    if not path.is_file():
        raise ConfigError(f"Не найден файл конфигурации {_display_path(path)}.")
    settings_cls = _with_yaml(TrainConfig, path, env_file)
    try:
        cfg = settings_cls()
    except ValidationError as exc:
        raise _config_error(exc, f"{_display_path(path)} и переменную DATA_DIR") from None
    if cfg.data.split_seed != SPLIT_SEED:
        raise ConfigError(
            f"Параметр data.split_seed должен быть {SPLIT_SEED}, "
            "как в зафиксированном сплите."
        )
    for name in cfg.data.preprocess_versions:
        _known_version(name)
    _known_version(cfg.augmentations.train_version)
    _known_version(cfg.augmentations.eval_version)
    return cfg


class ServerSection(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    max_upload_bytes: int = Field(gt=0)
    request_timeout_seconds: int = Field(gt=0)


class InferenceConfig(BaseSettings):
    # Файл модели. В YAML его нет.
    model_artifact_path: str = Field(min_length=1)
    # Версия preprocessing. Для итогового опыта это v1_baseline, как на test.
    preprocess_version: str = Field(min_length=1)
    # Идентификатор модели, не путь к файлу.
    model_version: str = Field(min_length=1)
    server: ServerSection

    @field_validator("model_artifact_path")
    @classmethod
    def _strip_model_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("пусто")
        return value


def load_inference_config(
    yaml_path: Path | None = None,
    env_file: str | None = ".env",
) -> InferenceConfig:
    """Возвращает разобранный inference-конфиг. Код обучения для этого не нужен."""
    path = Path(yaml_path) if yaml_path else INFERENCE_YAML
    if not path.is_file():
        raise ConfigError(f"Не найден файл конфигурации {_display_path(path)}.")
    settings_cls = _with_yaml(InferenceConfig, path, env_file)
    try:
        cfg = settings_cls()
    except ValidationError as exc:
        raise _config_error(
            exc, f"{_display_path(path)} и переменную MODEL_ARTIFACT_PATH"
        ) from None
    _known_version(cfg.preprocess_version)
    return cfg
