# Карточка датасета

Подмножество цветных снимков листьев PlantVillage: 8 классов, три культуры. Ниже источник, лицензия, состав, сплит и две версии препроцессинга.

## Источник

- Репозиторий: https://github.com/spMohanty/PlantVillage-Dataset
- Коммит: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`
- Каталог: `raw/color`
- Статья: Mohanty, Hughes, Salathé. Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science, 2016. DOI `10.3389/fpls.2016.01419`.
- Первое открытое объявление набора: Hughes, Salathé. An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060, 2015.

Скрипт `scripts/download_plantvillage.py` клонирует репозиторий разреженно и переключается именно на этот коммит. Берётся только `raw/color` и только восемь папок ниже.

## Лицензия

В arXiv:1511.08060 авторы пишут, что снимки выпущены под Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0) и что алгоритм, обученный на этих данных, попадает под ту же лицензию.

Карточка Hugging Face того же автора, на которую ссылается README репозитория, указывает `license: cc-by-sa-3.0`: https://huggingface.co/datasets/mohanty/PlantVillage

В самом git-репозитории файла LICENSE нет. Для этой работы снимки используются как учебный набор и в git не кладутся.

## Состав после фильтра

| Класс | Всего | train | val | test |
|---|---:|---:|---:|---:|
| Apple___Apple_scab | 630 | 441 | 94 | 95 |
| Apple___Black_rot | 621 | 434 | 93 | 94 |
| Apple___healthy | 1645 | 1151 | 246 | 248 |
| Potato___Late_blight | 1000 | 700 | 150 | 150 |
| Potato___healthy | 152 | 106 | 22 | 24 |
| Tomato___Early_blight | 1000 | 700 | 150 | 150 |
| Tomato___Late_blight | 1909 | 1336 | 286 | 287 |
| Tomato___healthy | 1591 | 1113 | 238 | 240 |
| Итого | 8548 | 5981 | 1279 | 1288 |

Имена папок совпадают с источником. В задании у healthy иногда стоит заглавная буква. В репозитории она строчная: `Apple___healthy`, `Potato___healthy`, `Tomato___healthy`.

## Разбиение

- Доли: 70% train, 15% val, 15% test.
- Стратификация по классу.
- `split_seed = 42`.
- Округление: `n_train = int(n * 0.70)`, `n_val = int(n * 0.15)`, `n_test = n - n_train - n_val`. Хвост остаётся в test, поэтому сумма частей равна числу файлов.
- Порядок до перемешивания: классы как в `plant_disease/classes.py`, внутри класса пути отсортированы. Один генератор `random.Random(42)` идёт по классам в этом порядке.
- Test не используется для подбора гиперпараметров. Он нужен только для итоговой оценки.

Манифест `splits/split_manifest.json` хранит для каждого файла путь, класс, часть сплита, размер и sha256.

## Что версионируется

Две версии препроцессинга — это код и JSON-спецификации, а не две копии картинок. Аугментации случайные, сохранять их как второй датасет бессмысленно.

| Что | Где | Идентификатор |
|---|---|---|
| Отфильтрованные снимки | DVC, `data/filtered.dvc` | md5 `79713a6fc71568742eec07af27f5093c.dir` |
| Манифест сплита | git, `splits/split_manifest.json` | sha256 `c492e8872d6d20fb7ca19cbb70e750a412e66ac0ecbccf3458c13704b6a82a93` |
| Версия A, `v1_baseline` | `splits/preprocess_v1_baseline.json` | sha256 `b634c08ddfb8c8cd8ee6ecd287cbf70990ab91c2434ac0055a63c2af14784dc8` |
| Версия B, `v2_augmented` | `splits/preprocess_v2_augmented.json` | sha256 `1550d83c22fd005335cac4927ed1d960de9edd9ba101edc9ed9074a4fcf6445c` |

DVC-каталог содержит 8549 файлов и 139885192 байта: 8548 снимков и служебный `SOURCE_COMMIT.txt`. В git этих файлов нет.

Сводка контрольных сумм продублирована в `splits/checksums.json` и `splits/preprocess_versions.json`.

### Версия A, v1_baseline

1. Resize до 224×224.
2. Перевод в тензор.
3. Нормализация ImageNet: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.

Преобразование детерминированное.

### Версия B, v2_augmented

Та же нормализация ImageNet и выход 224×224. Вместо обычного resize стоят аугментации. В спецификации записан `augment_seed = 42`, но `reset_seed` обучение не вызывает. Повтор кадра задаёт `torch.manual_seed(split_seed + epoch)` в `plant_disease/training.py`: при `split_seed = 42` это 43, 44, 45 и 46 на эпохах 1–4. Порядок батчей на train задаёт генератор DataLoader с тем же `split_seed`. Отдельная папка с аугментированными картинками не создаётся.

1. RandomResizedCrop 224, scale `(0.8, 1.0)`. Нижняя граница scale зафиксирована здесь, чтобы не вырезать крошечный фрагмент листа.
2. RandomHorizontalFlip, p `0.5`.
3. RandomRotation на ±15°.
4. ColorJitter: brightness `0.2`, contrast `0.2`, saturation `0.2`, hue `0.05`. В задании величины ColorJitter не заданы, поэтому они записаны в спецификации и входят в её sha256.
5. Перевод в тензор и нормализация ImageNet, как в версии A.

Код: `plant_disease/preprocess.py`.

## Как получить те же файлы

Общего DVC remote нет: для него понадобилось бы внешнее хранилище и секрет. Локальный remote автора лежит вне репозитория, путь записан только в `.dvc/config.local`. Этот файл в git не входит. Преподавателю он не нужен.

Повторяемый путь — скрипт скачивания и lock сплита:

```powershell
poetry run python scripts/download_plantvillage.py
poetry run python scripts/make_split.py
poetry run python scripts/write_checksums.py
```

`download_plantvillage.py` и `make_split.py` используют только Python 3.11 и git. `write_checksums.py` читает `.dvc`-файл через PyYAML из окружения проекта.

После `make_split.py` sha256 файла `splits/split_manifest.json` должен быть равен `c492e8872d6d20fb7ca19cbb70e750a412e66ac0ecbccf3458c13704b6a82a93`. Это и есть проверка, что набор и сплит те же. Если заново выполнить `dvc add data/filtered`, md5 каталога должен быть `79713a6fc71568742eec07af27f5093c.dir`.

## Ограничения

- Снимки сделаны в контролируемых условиях, часто на однородном фоне. Это не полевая съёмка.
- Взяты 8 классов из 38. Остальные культуры и болезни не входят.
- Классы несбалансированы: у Potato___healthy 152 файла, у Tomato___Late_blight 1909.
- Сплит режет отдельные файлы, а не физический лист. Несколько фото одного листа могут попасть в разные части. Официальное разбиение авторов другое: 80/20 и с группировкой по листу. Здесь оно не используется, потому что заданы доли 70/15/15 и seed 42.
- Test не выбирает ни чекпоинт, ни победителя. Чекпоинт внутри опыта и выбор среди трёх опытов смотрят на val macro-F1. Test считается один раз в конце.
- Версия B не является вторым сохранённым датасетом.
