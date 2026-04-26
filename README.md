# Live School Project — распознавание жестов руки

Проект для обучения и запуска нейросети, распознающей жесты руки по изображению с камеры или из папки с тестовыми изображениями.
Модель: `ResNet18` на `PyTorch`.

## Что умеет проект

- Подготовка датасета из `.parquet` в структуру `dataset/train` и `dataset/val`
- Обучение классификатора жестов
- Проверка качества на валидации
- Инференс на папке изображений
- Live-распознавание с веб-камеры:
  - через OpenCV-окно (`live.py`)
  - через GUI на Tkinter (`app.py`)

## Структура проекта

- `prepare_dataset.py` — конвертация parquet-данных в изображения и разбиение train/val
- `dataset.py` — загрузчики данных (`DataLoader`)
- `train.py` — обучение модели
- `test_folder.py` — оценка accuracy на `dataset/val`
- `infer.py` — предсказания для изображений из папки `realtest`
- `live.py` — онлайн-распознавание с камеры в окне OpenCV
- `app.py` — графическое приложение (Tkinter) с показом результата и примера жеста
- `dataset/` — датасет в формате `ImageFolder` (`train/<class>`, `val/<class>`)

## Требования

- Python 3.10+ (желательно)
- Камера (для `live.py` и `app.py`)
- Зависимости:
  - `torch`
  - `torchvision`
  - `opencv-python`
  - `pillow`
  - `pyarrow`

## Установка

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install torch torchvision opencv-python pillow pyarrow
```

## Подготовка датасета

1. Укажи путь к parquet-файлам в `prepare_dataset.py`:
   - `LOCAL_PARQUET_DIR`
2. Запусти:

```bash
python prepare_dataset.py
```

После этого появится структура:

```text
dataset/
  train/
    <class_id>/
  val/
    <class_id>/
```

## Обучение

```bash
python train.py
```

По умолчанию в `train.py` модель сохраняется как:

- `model.pth`

> ВАЖНО: в `live.py`, `app.py`, `infer.py`, `test_folder.py` по умолчанию ожидается файл `hand_gesture_model.pth`.
> После обучения либо переименуй `model.pth` в `hand_gesture_model.pth`, либо приведи `MODEL_PATH` во всех скриптах к одному имени.

## Проверка качества

```bash
python test_folder.py
```

Скрипт выводит итоговую точность на `dataset/val`.

## Инференс на папке изображений

1. Создай папку `realtest/` и положи туда изображения (`.jpg`, `.jpeg`, `.png`)
2. Запусти:

```bash
python infer.py
```

В консоли появятся предсказанные классы и confidence.

## Live-распознавание с камеры

### Вариант 1: OpenCV-окно

```bash
python live.py
```

- Нажми `q` для выхода.

### Вариант 2: GUI (Tkinter)

```bash
python app.py
```

- Кнопка **«Сделать снимок и распознать»** выполняет предсказание.
- Справа отображается пример изображения предсказанного класса из датасета.

## Полезные настройки

- В `train.py`:
  - `EPOCHS` — число эпох
  - `LR` — learning rate
- В `dataset.py`:
  - `BATCH_SIZE`
- В `prepare_dataset.py`:
  - `TRAIN_RATIO`
  - `JPEG_QUALITY`

## Возможные проблемы

- **Камера не открывается**
  Проверь, что камера не занята другим приложением и доступны разрешения ОС.
- **Ошибка загрузки модели**
  Проверь путь и имя файла чекпоинта (`MODEL_PATH`) во всех скриптах.
- **CUDA недоступна**
  Скрипты автоматически перейдут на CPU.

## Идеи для улучшения

- Добавить `requirements.txt`
- Сохранять лучшую модель по валидационной метрике
- Добавить аугментации и раннюю остановку
- Логировать метрики (TensorBoard/W&B)
- Экспорт модели в ONNX/torchscript