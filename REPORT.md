# Отчёт о сдаче

## Репозиторий

https://github.com/Falomas/plant-disease-clf

## Тег и коммит

Имя тега: `v1.0.0`. Тег аннотированный. Он указывает на merge-коммит в `main`, в котором этот файл уже есть. История `main` не переписывалась.

Полный SHA этого коммита — это `git rev-parse v1.0.0^{}`. Тот же SHA записан в сообщении тега.

Коммит документации, поверх которого добавлен отчёт (pull request 22): `2dd45c096355c89a65a95e3d27479ce1bd3144ed`.

## Эксперимент

Итоговый run MLflow: `5fd5ed281bce46088edaef1277d0d4fd`.

Это EfficientNet-B0, опыт `exp3_efficientnet_early_stop`. На run стоит тег `production`. Локальный каталог `mlruns/` в git не входит.

## Данные и препроцессинг

Контрольные суммы из [DATASET_CARD.md](DATASET_CARD.md):

| Что | Где | Идентификатор |
|---|---|---|
| Отфильтрованные снимки | DVC, `data/filtered.dvc` | md5 `79713a6fc71568742eec07af27f5093c.dir` |
| Манифест сплита | `splits/split_manifest.json` | sha256 `c492e8872d6d20fb7ca19cbb70e750a412e66ac0ecbccf3458c13704b6a82a93` |
| Препроцессинг `v1_baseline` | `splits/preprocess_v1_baseline.json` | sha256 `b634c08ddfb8c8cd8ee6ecd287cbf70990ab91c2434ac0055a63c2af14784dc8` |
| Препроцессинг `v2_augmented` | `splits/preprocess_v2_augmented.json` | sha256 `1550d83c22fd005335cac4927ed1d960de9edd9ba101edc9ed9074a4fcf6445c` |

## Артефакт ONNX

Путь через DVC: `artifacts/best_model.onnx.dvc`.

Сам файл `artifacts/best_model.onnx` в git не входит. md5 `0a032a9a2b40b6a062981946bc5b7dc9`, размер 16061189 байт. В контейнере он смонтирован как `/models/best_model.onnx` (`MODEL_ARTIFACT_PATH`).

Чекпойнт, из которого собран ONNX: `artifacts/best_model.pt.dvc`, md5 `f0996e849d89ec519a0ad9d0ce98bdd7`.

## Образ

`plant-disease-clf:1.0.0-4c2cc99`

Запуск:

```
docker run --rm -p 8000:8000 -e MODEL_ARTIFACT_PATH=/models/best_model.onnx -v "./artifacts:/models:ro" plant-disease-clf:1.0.0-4c2cc99
```

## Проверка

- `poetry run pytest` на этом теге: 8 passed.
- `bash scripts/smoke_test.sh` на контейнере из образа `plant-disease-clf:1.0.0-4c2cc99`: ответ `/health` со `status` `ok` и `model_loaded` `true`, версия модели `5fd5ed281bce46088edaef1277d0d4fd`; `/docs` открывается; `/predict` по `tests/fixtures/late_blight.jpg` возвращает класс `Tomato___Late_blight`. Скрипт печатает `smoke ok`.
- Ошибочный запрос: файл не-изображения на `/predict` получает HTTP 400 и текст `Неверный формат изображения.`
