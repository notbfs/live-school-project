import os
import random
from pathlib import Path
from PIL import Image
from io import BytesIO
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

# ---------------- Настройки ----------------
LOCAL_PARQUET_DIR = r"C:\Users\andre\.cache\huggingface\hub\datasets--cj-mills--hagrid-classification-512p-no-gesture-150k\snapshots\70afa88ad4d25ce1402e76b6f6b10c00eb44e7fa\data"  # путь к parquet
TARGET_DIR = "dataset"
TRAIN_RATIO = 0.9
JPEG_QUALITY = 85  # сохраняем в JPEG для ускорения и экономии места
# ------------------------------------------

def prepare_folder_structure(classes):
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, str(cls)), exist_ok=True)

def save_single_image(args):
    img_obj, lbl, idx, split = args
    try:
        if isinstance(img_obj, dict) and "bytes" in img_obj:
            img = Image.open(BytesIO(img_obj["bytes"])).convert("RGB")
        else:
            return
        save_path = os.path.join(TARGET_DIR, split, str(lbl), f"{lbl}_{idx}.jpg")
        img.save(save_path, format="JPEG", quality=JPEG_QUALITY)
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении: {e}")

def save_images_from_parquet(parquet_file, split_ratio=TRAIN_RATIO):
    table = pq.read_table(parquet_file)
    data = table.to_pydict()
    images = data["image"]
    labels = data["label"]

    classes = sorted(list(set(labels)))
    prepare_folder_structure(classes)

    per_class = {cls: [] for cls in classes}
    for img_obj, lbl in zip(images, labels):
        per_class[lbl].append(img_obj)

    tasks = []
    for cls, imgs in per_class.items():
        random.shuffle(imgs)
        split_idx = int(len(imgs) * split_ratio)
        for i, img in enumerate(imgs):
            split = "train" if i < split_idx else "val"
            tasks.append((img, cls, i, split))

    # --- Многопроцессорная запись ---
    with Pool(cpu_count()) as p:
        p.map(save_single_image, tasks)

    print(f"[INFO] Конвертация {parquet_file} завершена.")

def main():
    parquet_files = list(Path(LOCAL_PARQUET_DIR).glob("*.parquet"))
    if not parquet_files:
        print("[ERROR] parquet файлы не найдены в", LOCAL_PARQUET_DIR)
        return

    for parquet_file in parquet_files:
        print(f"[INFO] Обрабатываем {parquet_file}")
        save_images_from_parquet(parquet_file)

    print("[INFO] Датасет полностью подготовлен в папке 'dataset/'")

if __name__ == "__main__":
    main()
