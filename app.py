import os
import random

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


MODEL_PATH = "hand_gesture_model.pth"
REALTEST_DIR = "realtest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def is_label_match(filename: str, label: str):
    name_without_ext = os.path.splitext(filename)[0].lower()
    normalized_label = str(label).lower().replace("-", "_").replace(" ", "_")
    tokens = name_without_ext.replace("-", "_").replace(" ", "_").split("_")
    return normalized_label in tokens or name_without_ext.startswith(f"{normalized_label}_")


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


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание жестов руки")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1100x600")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", font=("Segoe UI", 11), padding=8)
        style.configure("TFrame", background="#1e1e1e")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(15, 0))

        try:
            self.model, self.idx_to_class = load_model()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель:\n{e}")
            raise

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру")
            raise RuntimeError("Camera not available")

        title_label = tk.Label(
            left_frame,
            text="Живое видео с камеры",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        title_label.pack(anchor="w", pady=(0, 5))

        self.video_label = tk.Label(left_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=(10, 0))

        self.capture_button = ttk.Button(
            btn_frame,
            text="Сделать снимок и распознать",
            command=self.capture_and_predict,
        )
        self.capture_button.pack(side=tk.LEFT)

        result_title = tk.Label(
            right_frame,
            text="Результат распознавания",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        result_title.pack(anchor="w", pady=(0, 5))

        self.result_var = tk.StringVar(value="Результат: —")
        self.result_label = tk.Label(
            right_frame,
            textvariable=self.result_var,
            font=("Segoe UI", 14),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        self.result_label.pack(anchor="w", pady=(0, 15))

        example_title = tk.Label(
            right_frame,
            text="Пример этого жеста",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        example_title.pack(anchor="w", pady=(0, 5))

        self.example_label = tk.Label(right_frame, bg="#000000")
        self.example_label.pack(fill=tk.BOTH, expand=True)

        self.example_img_tk = None

        self.current_frame = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            img = img.resize((640, 480))

            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)

        self.root.after(20, self.update_video)

    def capture_and_predict(self):
        if self.current_frame is None:
            messagebox.showwarning("Внимание", "Кадр с камеры пока не получен.")
            return

        try:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(1)

            label = self.idx_to_class[pred.item()]
            confidence = conf.item()

            self.result_var.set(f"Результат: {label} ({confidence:.2%})")

            self.show_example_image(label)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить распознавание:\n{e}")

    def show_example_image(self, label):
        candidates = []
        label_dirs = [
            os.path.join(REALTEST_DIR, str(label)),
            os.path.join(REALTEST_DIR, "train", str(label)),
            os.path.join(REALTEST_DIR, "val", str(label)),
        ]
        for directory in label_dirs:
            if not os.path.isdir(directory):
                continue
            for f in os.listdir(directory):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    candidates.append(os.path.join(directory, f))

        if not candidates and os.path.isdir(REALTEST_DIR):
            for f in os.listdir(REALTEST_DIR):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and is_label_match(f, label):
                    candidates.append(os.path.join(REALTEST_DIR, f))

        if not candidates:
            self.example_label.configure(image="")
            self.example_img_tk = None
            return

        example_path = random.choice(candidates)
        try:
            img = Image.open(example_path).convert("RGB")
            img = img.resize((320, 320))
            self.example_img_tk = ImageTk.PhotoImage(image=img)
            self.example_label.configure(image=self.example_img_tk)
        except Exception as e:
            print(f"Не удалось загрузить пример изображения: {e}")

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

