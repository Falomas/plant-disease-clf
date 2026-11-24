# Классификация заболеваний растений

Учебный проект: классификация заболеваний растений по изображениям листьев.

Сейчас в репозитории только воспроизводимое окружение. Обучения, сервиса и Docker здесь ещё нет.

## Требования

- Python 3.11. Другие версии Poetry не возьмёт: в `pyproject.toml` указано `>=3.11,<3.12`.
- Poetry 2.

Проверено на Windows: Python 3.11.9 и Poetry 2.5.2.

## Установка Poetry

Если команда `poetry` уже находится, этот шаг не нужен.

Официальный установщик, Windows (PowerShell):

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -3.11 -
```

Установщик кладёт `poetry.exe` в `%APPDATA%\Python\Scripts`. Добавьте этот каталог в `PATH` пользователя и откройте терминал заново:

```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:APPDATA\Python\Scripts",
    "User"
)
```

Тот же установщик на Linux и macOS:

```bash
curl -sSL https://install.python-poetry.org | python3.11 -
```

Дальше нужен каталог `$HOME/.local/bin` в `PATH`.

## Восстановление окружения

Команды из корня репозитория. Отдельные пакеты через pip ставить не нужно: версии берутся из `poetry.lock`.

```powershell
poetry env use 3.11
poetry install
Copy-Item .env.example .env
```

На Linux и macOS последняя команда такая:

```bash
cp .env.example .env
```

`poetry install` ставит основные зависимости и группу `dev` (pytest, pytest-cov, ruff, black). Виртуальное окружение создаётся в `.venv/` — так записано в `poetry.toml`.

PyTorch и torchvision ставятся из CPU-индекса `https://download.pytorch.org/whl/cpu`, без CUDA.

Проверка, что окружение живое:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

На Linux и macOS:

```bash
.venv/bin/python -c "import torch; print(torch.__version__)"
```

## Данные

Восемь классов PlantVillage. Источник, лицензия, сплит и версии препроцессинга описаны в [DATASET_CARD.md](DATASET_CARD.md).

Скачивание и проверка, что файлы те же:

```powershell
poetry run python scripts/download_plantvillage.py
poetry run python scripts/make_split.py
poetry run python scripts/write_checksums.py
```

Ожидаемый sha256 манифеста: `c492e8872d6d20fb7ca19cbb70e750a412e66ac0ecbccf3458c13704b6a82a93`.
