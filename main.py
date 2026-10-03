import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import center_of_mass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_PATH = "mnist_cnn.pth"
EPOCHS = 5
RETRAIN = os.environ.get("RETRAIN", "0") == "1"   # set RETRAIN=1 to force retraining
SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"  # set GRADIO_SHARE=1 for a public link

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:  # older Pillow
    RESAMPLE = Image.BILINEAR


# ==========================================
# 1. Architecture
# ==========================================
class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        return self.fc2(x)


# ==========================================
# 2. Training / loading
# ==========================================
def get_model():
    model = EnhancedCNN().to(device)

    if os.path.exists(MODEL_PATH) and not RETRAIN:
        print(f"Loading saved weights from {MODEL_PATH} ...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model

    train_transform = transforms.Compose([
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    print("Loading dataset...")
    train_data = datasets.MNIST("./data", train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training model ({EPOCHS} epochs) on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                correct += (model(data).argmax(1) == target).sum().item()
        print(f"Epoch {epoch + 1}/{EPOCHS} | loss {running_loss / len(train_loader):.4f} "
              f"| test acc {100 * correct / len(test_data):.2f}%")

    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete, weights saved.")
    return model


# ==========================================
# 3. Preprocessing (white digit on black, like MNIST)
# ==========================================
def preprocess_canvas_image(pil_img):
    pil_img = pil_img.convert("L")
    bbox = pil_img.getbbox()
    if bbox is None:
        return None

    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    cropped = blurred.crop(bbox)
    cropped.thumbnail((20, 20), RESAMPLE)

    centered = Image.new("L", (28, 28), 0)
    offset = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
    centered.paste(cropped, offset)

    img_np = np.array(centered, dtype=np.float32) / 255.0
    if img_np.sum() == 0:
        return None

    cy, cx = center_of_mass(img_np)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_x = int(round(14.0 - cx))
        shift_y = int(round(14.0 - cy))
        img_np = np.roll(img_np, (shift_y, shift_x), axis=(0, 1))

    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def predict_probs(model, pil_img):
    tensor = preprocess_canvas_image(pil_img)
    if tensor is None:
        return None
    model.eval()
    with torch.no_grad():
        return F.softmax(model(tensor.to(device)), dim=1)[0].cpu().numpy()


# ==========================================
# 4. Desktop GUI (Tkinter)
# ==========================================
def run_tk_gui(model):
    import tkinter as tk

    class DigitRecognizerGUI:
        def __init__(self, model):
            self.model = model
            self.root = tk.Tk()
            self.root.title("Handwritten Digit Recognizer")

            main_frame = tk.Frame(self.root)
            main_frame.pack(padx=10, pady=10)

            left = tk.Frame(main_frame)
            left.pack(side=tk.LEFT, padx=10)

            self.canvas = tk.Canvas(left, width=280, height=280, bg="black")
            self.canvas.pack(pady=5)

            self.image = Image.new("L", (280, 280), 0)
            self.draw = ImageDraw.Draw(self.image)

            self.canvas.bind("<B1-Motion>", self.paint)
            self.canvas.bind("<ButtonRelease-1>", self.reset_prev_pos)
            self.prev_x = self.prev_y = None

            btns = tk.Frame(left)
            btns.pack(pady=5)
            tk.Button(btns, text="Predict", command=self.predict_digit,
                      font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
            tk.Button(btns, text="Clear", command=self.clear_canvas,
                      font=("Arial", 11)).pack(side=tk.LEFT, padx=5)

            right = tk.LabelFrame(main_frame, text=" Class Probabilities (High → Low) ",
                                  font=("Arial", 11, "bold"))
            right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

            self.proba_label = tk.Label(right, text="Draw a digit\nand click Predict",
                                        font=("Courier", 11), justify=tk.LEFT, anchor="nw")
            self.proba_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.label_result = tk.Label(self.root, text="Draw a digit (0-9) to see predictions.",
                                         font=("Arial", 13, "bold"))
            self.label_result.pack(pady=10)

            self.root.mainloop()

        def paint(self, event):
            size = 20
            if self.prev_x is not None and self.prev_y is not None:
                self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y,
                                        fill="white", width=size, capstyle=tk.ROUND, smooth=True)
                self.draw.line([self.prev_x, self.prev_y, event.x, event.y],
                               fill=255, width=size, joint="curve")
            self.prev_x, self.prev_y = event.x, event.y

        def reset_prev_pos(self, event):
            self.prev_x = self.prev_y = None

        def clear_canvas(self):
            self.canvas.delete("all")
            self.image = Image.new("L", (280, 280), 0)
            self.draw = ImageDraw.Draw(self.image)
            self.label_result.config(text="Draw a digit (0-9) to see predictions.")
            self.proba_label.config(text="Draw a digit\nand click Predict")

        def predict_digit(self):
            probs = predict_probs(self.model, self.image)
            if probs is None:
                self.label_result.config(text="Canvas is empty!")
                self.proba_label.config(text="Canvas is empty!")
                return

            ranked = sorted(enumerate(probs * 100), key=lambda x: x[1], reverse=True)
            text = "Rank    Digit    Confidence\n" + "-" * 26 + "\n"
            for rank, (digit, p) in enumerate(ranked, 1):
                text += f" #{rank:<2}    [{digit}]     {p:5.1f}%\n"
            self.proba_label.config(text=text)

            (d1, c1), (d2, c2) = ranked[0], ranked[1]
            self.label_result.config(
                text=f"Prediction: {d1} ({c1:.1f}%)  |  Runner-up: {d2} ({c2:.1f}%)")

    DigitRecognizerGUI(model)


# ==========================================
# 5. Web UI (Gradio)
# ==========================================
def _to_ink_image(sketch):
    """Convert whatever the Gradio sketch component returns into a
    single-channel PIL image with a WHITE digit on a BLACK background."""
    if sketch is None:
        return None

    # Gradio 4/5 ImageEditor returns {"background", "layers", "composite"}
    if isinstance(sketch, dict):
        layers = sketch.get("layers") or []
        arrays = []
        for layer in layers:
            if layer is None:
                continue
            arr = np.array(layer)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arrays.append(arr[:, :, 3])   # strokes live in the alpha channel
        if arrays:
            ink = np.maximum.reduce(arrays).astype("uint8")
            return Image.fromarray(ink)
        sketch = sketch.get("composite", sketch.get("image"))
        if sketch is None:
            return None

    if isinstance(sketch, Image.Image):
        sketch = np.array(sketch)

    if not isinstance(sketch, np.ndarray):
        return None

    if sketch.ndim == 3 and sketch.shape[2] == 4:
        alpha = sketch[:, :, 3]
        rgb = sketch[:, :, :3].mean(axis=2)
        if alpha.min() < 255:                 # transparent background -> alpha is the ink
            ink = alpha
        else:                                 # opaque background
            ink = rgb
    elif sketch.ndim == 3:
        ink = sketch[:, :, :3].mean(axis=2)
    else:
        ink = sketch

    ink = ink.astype("uint8")
    if ink.mean() > 127:                      # white paper, dark ink -> invert
        ink = 255 - ink
    return Image.fromarray(ink)


def launch_gradio_app(model):
    import gradio as gr
    print("Gradio version:", gr.__version__)

    def predict_gradio(sketch):
        try:
            pil_img = _to_ink_image(sketch)
            if pil_img is None:
                return None
            probs = predict_probs(model, pil_img)
            if probs is None:
                return None
            return {str(i): float(probs[i]) for i in range(10)}
        except Exception as e:
            import traceback
            traceback.print_exc()          # visible in terminal / Colab output
            raise gr.Error(f"Prediction failed: {e}")

    brush = gr.Brush(colors=["#FFFFFF"], default_size=20, color_mode="fixed")
    canvas = gr.ImageEditor(
        label="Draw a digit",
        type="numpy",
        image_mode="RGBA",
        canvas_size=(280, 280),
        sources=[],
        brush=brush,
        eraser=gr.Eraser(default_size=20),
        transforms=[],
        layers=False,
    )

    interface = gr.Interface(
        fn=predict_gradio,
        inputs=canvas,
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Handwritten Digit Recognizer",
        description="Draw a digit (0-9) with the brush, then press Submit.",
        flagging_mode="never" if int(gr.__version__.split(".")[0]) >= 5 else None,
    ) if int(gr.__version__.split(".")[0]) >= 5 else gr.Interface(
        fn=predict_gradio,
        inputs=canvas,
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Handwritten Digit Recognizer",
        description="Draw a digit (0-9) with the brush, then press Submit.",
        allow_flagging="never",
    )

    # No stdout redirect, so any tunnel/launch error is visible.
    # Colab and local runs don't need share=True.
    interface.queue().launch(share=SHARE, debug=True)


# ==========================================
# 6. Entry point
# ==========================================
def in_notebook():
    return "google.colab" in sys.modules or "ipykernel" in sys.modules


if __name__ == "__main__":
    model = get_model()

    has_display = os.environ.get("DISPLAY", "") != "" or os.name == "nt" or sys.platform == "darwin"
    if has_display and not in_notebook():
        try:
            print("Opening desktop Tkinter GUI...")
            run_tk_gui(model)
        except Exception as e:
            print(f"Tkinter failed ({e}). Falling back to Gradio...")
            launch_gradio_app(model)
    else:
        print("No desktop display (or notebook) detected. Launching Gradio...")
        launch_gradio_app(model)