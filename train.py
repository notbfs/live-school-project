import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 4
LR = 1e-4
MODEL_PATH = "hand_gesture_model.pth"

def main():
    train_loader, val_loader, class_to_idx = get_dataloaders()

    num_classes = len(class_to_idx)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {loss_sum:.4f} | Train Acc: {train_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx
    }, MODEL_PATH)

    print("✅ Модель сохранена в hand_gesture_model.pth")

if __name__ == "__main__":
    main()
