import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = "hand_gesture_model.pth"
TEST_DIR = "dataset/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v:k for k,v in class_to_idx.items()}

num_classes = len(class_to_idx)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

correct = 0
total = 0
for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.endswith((".png",".jpg",".jpeg")):
            label_name = os.path.basename(root)
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output,1).item()
            if idx_to_class[pred] == label_name:
                correct +=1
            total +=1

print(f"[INFO] Accuracy on test set: {correct/total:.4f} ({correct}/{total})")
