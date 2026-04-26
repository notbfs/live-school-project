import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "hand_gesture_model.pth"
IMAGE_DIR = "realtest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    # 🔹 загрузка чекпоинта
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(class_to_idx)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("✅ Модель загружена")

    with torch.no_grad():
        for fname in os.listdir(IMAGE_DIR):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(IMAGE_DIR, fname)
            image = Image.open(path).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            label = idx_to_class[pred.item()]
            confidence = conf.item()

            print(f"{fname:30s} → {label} ({confidence:.2%})")

if __name__ == "__main__":
    main()
