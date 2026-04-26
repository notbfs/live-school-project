import os
import random

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "hand_gesture_model.pth"
DATASET_DIR = "dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, idx_to_class


def get_random_example_path(label: str):
    candidates = []
    for split in ("train", "val"):
        class_dir = os.path.join(DATASET_DIR, split, str(label))
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                candidates.append(os.path.join(class_dir, f))

    if not candidates:
        return None

    return random.choice(candidates)


def main():
    model, idx_to_class = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    print("Нажмите 'q', чтобы выйти")

    last_example_label = None
    last_example_img = None

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Подготовка кадра для модели
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            label = idx_to_class[pred.item()]
            confidence = conf.item()

            # Рисуем результат на кадре
            text = f"{label} ({confidence:.2%})"
            cv2.rectangle(frame, (10, 10), (10 + 320, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Показать пример этого жеста в правом верхнем углу
            if label != last_example_label or last_example_img is None:
                example_path = get_random_example_path(label)
                if example_path is not None:
                    img = cv2.imread(example_path)
                    if img is not None:
                        last_example_img = img
                        last_example_label = label

            if last_example_img is not None:
                # Масштабируем пример и вписываем в правый верхний угол
                h, w = frame.shape[:2]
                ex_h, ex_w = 160, 160
                example_resized = cv2.resize(last_example_img, (ex_w, ex_h))

                x_start = max(0, w - ex_w - 10)
                y_start = 10
                x_end = x_start + ex_w
                y_end = y_start + ex_h

                frame[y_start:y_end, x_start:x_end] = example_resized

            cv2.imshow("Live Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

