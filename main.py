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
# 1. Advanced Architecture & Augmentations
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=8),
    transforms.ElasticTransform(alpha=15.0, sigma=3.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading dataset...")
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.silu(out)

class HighAccuracyCNN(nn.Module):
    def __init__(self):
        super(HighAccuracyCNN, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.res1 = ResBlock(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.res2 = ResBlock(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.mid_conv(x)
        x = self.res2(x)
        x = self.pool2(x)
        return self.head(x)

model = HighAccuracyCNN().to(device)
epochs = 10
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.003, steps_per_epoch=len(train_loader), epochs=epochs
)

# ==========================================
# 2. Model Training with Progress Logging
# ==========================================
print(f"Training high-accuracy ResNet model ({epochs} epochs)...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    train_acc = (correct / total) * 100
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Training Acc: {train_acc:.2f}%")

model.eval()
print("Training complete! Model saved to mnist_high_accuracy.pth")
torch.save(model.state_dict(), "mnist_high_accuracy.pth")

# ==========================================
# 3. Canvas Preprocessing Helper
# ==========================================
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
    return img_tensor.to(device)

# ==========================================
# 4. Desktop Tkinter GUI Implementation
# ==========================================
class DigitRecognizerGUI:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer (High Accuracy ResNet)")

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

        probs_np = probabilities.cpu().numpy() * 100
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
# 5. Gradio Fallback for Headless Environments
# ==========================================
def launch_gradio_app(model):
    import gradio as gr

    def predict_gradio(sketch_dict):
        image = sketch_dict.get("composite", sketch_dict.get("image", None)) if isinstance(sketch_dict, dict) else sketch_dict
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
        app, local_url, share_url = interface.launch(share=True, prevent_thread_lock=True, quiet=True)

    if share_url:
        print(share_url)

    interface.block_thread()

# ==========================================
# 6. Execution Entry Point
# ==========================================
if __name__ == "__main__":
    if os.environ.get("DISPLAY", "") != "" or os.name == "nt":
        try:
            print("Opening desktop Tkinter GUI...")
            DigitRecognizerGUI(model)
        except Exception as e:
            print(f"Tkinter failed ({e}). Falling back to Gradio web interface...")
            launch_gradio_app(model)
    else:
        print("Headless cloud environment detected. Launching Gradio web interface...")
        launch_gradio_app(model)