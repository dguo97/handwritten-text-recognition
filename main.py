import os
import io
import contextlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import center_of_mass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import tkinter as tk

# ==========================================
# 1. Architecture & Transforms Setup
# ==========================================
train_transform = transforms.Compose([
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train_and_save_model(model_path="mnist_cnn.pth"):
    print("Loading dataset for training...")
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = EnhancedCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training model (5 epochs)...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5 complete. Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Saved weights to {model_path}")
    return model

def load_or_train_model(model_path="mnist_cnn.pth"):
    model = EnhancedCNN()
    if os.path.exists(model_path):
        print(f"Loading existing checkpoint from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        model = train_and_save_model(model_path)
        model.eval()
    return model

def preprocess_canvas_image(pil_img):
    bbox = pil_img.getbbox()
    if bbox is None:
        return None

    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    cropped = blurred.crop(bbox)
    cropped.thumbnail((20, 20), Image.Resampling.BILINEAR)

    centered_img = Image.new("L", (28, 28), "black")
    offset = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
    centered_img.paste(cropped, offset)

    img_np = np.array(centered_img, dtype=np.float32) / 255.0
    cy, cx = center_of_mass(img_np)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_x = np.round(14.0 - cx).astype(int)
        shift_y = np.round(14.0 - cy).astype(int)
        img_np = np.roll(img_np, (shift_y, shift_x), axis=(0, 1))

    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_tensor = (img_tensor - 0.5) / 0.5
    return img_tensor

# ==========================================
# 2. GUI Implementation (Desktop / Tkinter)
# ==========================================
class DigitRecognizerGUI:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")

        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(left_frame, width=280, height=280, bg="black")
        self.canvas.pack(pady=5)

        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_prev_pos)

        self.prev_x = None
        self.prev_y = None

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5)

        self.btn_predict = tk.Button(btn_frame, text="Predict", command=self.predict_digit, font=("Arial", 11, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_canvas, font=("Arial", 11))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        right_frame = tk.LabelFrame(main_frame, text=" Class Probabilities (High → Low) ", font=("Arial", 11, "bold"))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.proba_label = tk.Label(
            right_frame, 
            text="Draw a digit\nand click Predict", 
            font=("Courier", 11), 
            justify=tk.LEFT,
            anchor="nw"
        )
        self.proba_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.label_result = tk.Label(self.root, text="Draw a digit (0-9) to see predictions.", font=("Arial", 13, "bold"))
        self.label_result.pack(pady=10)

        self.root.mainloop()

    def paint(self, event):
        brush_size = 20
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y,
                                    fill="white", width=brush_size, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.prev_x, self.prev_y, event.x, event.y],
                           fill="white", width=brush_size, joint="round")
            
        self.prev_x = event.x
        self.prev_y = event.y

    def reset_prev_pos(self, event):
        self.prev_x = None
        self.prev_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit (0-9) to see predictions.")
        self.proba_label.config(text="Draw a digit\nand click Predict")

    def predict_digit(self):
        img_tensor = preprocess_canvas_image(self.image)
        if img_tensor is None:
            self.label_result.config(text="Canvas is empty!")
            self.proba_label.config(text="Canvas is empty!")
            return

        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]

        probs_np = probabilities.numpy() * 100
        ranked_predictions = sorted(enumerate(probs_np), key=lambda x: x[1], reverse=True)

        ranking_text = "Rank    Digit    Confidence\n" + "-" * 26 + "\n"
        for rank, (digit, prob) in enumerate(ranked_predictions, 1):
            ranking_text += f" #{rank:<2}    [{digit}]     {prob:5.1f}%\n"

        self.proba_label.config(text=ranking_text)

        top_digit, top_conf = ranked_predictions[0]
        second_digit, second_conf = ranked_predictions[1]
        self.label_result.config(
            text=f"Prediction: {top_digit} ({top_conf:.1f}%)  |  Runner-up: {second_digit} ({second_conf:.1f}%)"
        )

# ==========================================
# 3. Web Interface Implementation (Gradio)
# ==========================================
def launch_gradio_app(model):
    import gradio as gr

    def predict_gradio(sketch):
        if sketch is None:
            return "Please draw a digit."

        if isinstance(sketch, dict):
            image = sketch.get("composite", sketch.get("layers", [None])[0] if sketch.get("layers") else sketch.get("image", None))
        else:
            image = sketch

        if image is None:
            return "Please draw a digit."

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 4:
                alpha = image[:, :, 3]
                pil_img = Image.fromarray(alpha.astype("uint8"))
            else:
                pil_img = Image.fromarray(image.astype("uint8")).convert("L")
                img_np = np.array(pil_img)
                if np.mean(img_np) > 127:
                    pil_img = Image.fromarray(255 - img_np)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("L")
            img_np = np.array(pil_img)
            if np.mean(img_np) > 127:
                pil_img = Image.fromarray(255 - img_np)
        else:
            return "Invalid image format."

        img_tensor = preprocess_canvas_image(pil_img)
        if img_tensor is None:
            return "Canvas is empty!"

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]

        return {str(i): float(probabilities[i]) for i in range(10)}

    canvas = gr.Sketchpad(canvas_size=(280, 280), image_mode="L")
    interface = gr.Interface(
        fn=predict_gradio,
        inputs=canvas,
        outputs=gr.Label(num_top_classes=3),
        title="Handwritten Digit Recognizer",
        description="Draw a digit (0–9) on the canvas to see predictions."
    )

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        app, local_url, share_url = interface.launch(share=True, prevent_thread_lock=True)

    if share_url:
        print(f"\n* Running on public URL: {share_url}\n")

    interface.block_thread()

# ==========================================
# 4. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    model = load_or_train_model("mnist_cnn.pth")

    if os.environ.get("DISPLAY", "") != "" or os.name == "nt":
        try:
            print("Opening desktop Tkinter GUI...")
            DigitRecognizerGUI(model)
        except Exception as e:
            print(f"Tkinter failed ({e}). Falling back to Gradio web interface...")
            launch_gradio_app(model)
    else:
        print("Headless cloud environment detected (e.g., Google Colab). Launching Gradio web interface...")
        launch_gradio_app(model)