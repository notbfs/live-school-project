# Live School Project

Проект для распознавания жестов руки на базе `PyTorch` и `ResNet18`.

Пайплайн состоит из трех шагов:
1. Подготовка датасета из `.parquet` в формат `ImageFolder`.
2. Обучение классификатора жестов.
3. Запуск предсказаний на изображениях или в реальном времени с камеры.

## Возможности

- Конвертация parquet-датасета в папки `dataset/train` и `dataset/val`.
- Обучение модели `ResNet18` и сохранение чекпоинта.
- Предсказания на изображениях из папки `realtest`.
- Live-распознавание с веб-камеры через OpenCV.

## Структура проекта

- `prepare_dataset.py` - подготавливает датасет из parquet-файлов.
- `dataset.py` - создает `DataLoader` для train/val.
- `train.py` - обучает модель и сохраняет `hand_gesture_model.pth`.
- `infer.py` - делает предсказания на изображениях из `realtest/`.
- `live.py` - запускает распознавание жестов с камеры в реальном времени.

## Требования

- Python 3.10+
- Веб-камера (только для `live.py`)
- Зависимости:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `pillow`
  - `pyarrow`

## Установка

### 1) Клонирование проекта

```bash
git clone <github.com/notbfs/live-school-project>
cd live-school-project
```

Если проект уже у вас на компьютере, просто перейдите в папку проекта.

### 2) Создание виртуального окружения

```bash
python -m venv .venv
```

### 3) Активация окружения

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

Windows (cmd):
```bash
.venv\Scripts\activate.bat
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 4) Установка зависимостей

```bash
pip install --upgrade pip
pip install torch torchvision opencv-python pillow pyarrow
```

## Подготовка датасета

1. Откройте `prepare_dataset.py`.
2. Проверьте значение `LOCAL_PARQUET_DIR` - это путь к папке с `.parquet` файлами.
3. Запустите:

```bash
python prepare_dataset.py
```

После выполнения появится структура:

```text
dataset/
  train/
    <class_id>/
  val/
    <class_id>/
```

## Обучение модели

```bash
python train.py
```

После обучения создается файл:

- `hand_gesture_model.pth`

## Предсказания на изображениях

1. Создайте папку `realtest/` в корне проекта.
2. Добавьте в нее изображения (`.jpg`, `.jpeg`, `.png`).
3. Запустите:

```bash
python infer.py
```

В консоли будут выведены имя файла, предсказанный класс и уверенность модели.

## Live-распознавание с камеры

```bash
python live.py
```

- Нажмите `q` для выхода.
- В левом верхнем углу отображается предсказанный жест и confidence.
- Справа показывается пример изображения соответствующего класса из `realtest/` (если найден).

## Полезные параметры

- `prepare_dataset.py`:
  - `TRAIN_RATIO` - доля train-части (по умолчанию `0.9`).
  - `JPEG_QUALITY` - качество сохраняемых JPG.
- `dataset.py`:
  - `BATCH_SIZE` - размер батча.
- `train.py`:
  - `EPOCHS` - количество эпох.
  - `LR` - скорость обучения.

## Частые проблемы

- Камера не открывается:
  - Проверьте, что камера не занята другим приложением.
  - Проверьте разрешения на доступ к камере в системе.
- Ошибка загрузки модели:
  - Убедитесь, что файл `hand_gesture_model.pth` находится в корне проекта.
- Нет CUDA:
  - Скрипты автоматически используют CPU, это нормально.