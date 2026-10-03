import os
import io
import base64
import threading
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import center_of_mass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from flask import Flask, render_template_string, request, jsonify

# ==========================================
# 1. Architecture & Augmentation Setup
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

print("Loading dataset...")
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

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

model = EnhancedCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 2. Model Training
# ==========================================
print("Training model (2.0 epochs)...")
for epoch in range(1):
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

model.eval()
print("Training complete!")
torch.save(model.state_dict(), "mnist_cnn.pth")

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
# 3. Flask Web Application Setup
# ==========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Digit Recognizer</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background: #f4f4f9; margin-top: 50px; }
        canvas { border: 3px solid #333; background: black; cursor: crosshair; border-radius: 8px; touch-action: none; }
        .container { display: inline-block; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        button { padding: 10px 20px; font-size: 16px; margin: 10px 5px; cursor: pointer; border: none; border-radius: 5px; background: #007BFF; color: white; }
        button:hover { background: #0056b3; }
        #clear-btn { background: #dc3545; }
        #clear-btn:hover { background: #a71d2a; }
        #result { font-size: 20px; font-weight: bold; margin-top: 15px; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Handwritten Digit Recognizer</h2>
        <canvas id="paintCanvas" width="280" height="280"></canvas>
        <div>
            <button onclick="predictDigit()">Predict</button>
            <button id="clear-btn" onclick="clearCanvas()">Clear</button>
        </div>
        <div id="result">Draw a digit and click Predict</div>
    </div>

    <script>
        const canvas = document.getElementById('paintCanvas');
        const ctx = canvas.getContext('2d');
        let painting = false;

        ctx.strokeStyle = 'white';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        canvas.addEventListener('mousedown', (e) => { painting = true; draw(e); });
        canvas.addEventListener('mouseup', () => { painting = false; ctx.beginPath(); });
        canvas.addEventListener('mousemove', draw);

        canvas.addEventListener('touchstart', (e) => { painting = true; draw(e.touches[0]); e.preventDefault(); });
        canvas.addEventListener('touchend', () => { painting = false; ctx.beginPath(); });
        canvas.addEventListener('touchmove', (e) => { draw(e.touches[0]); e.preventDefault(); });

        function draw(e) {
            if (!painting) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }

        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').innerText = "Draw a digit and click Predict";
        }

        clearCanvas();

        function predictDigit() {
            const dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerText = data.error;
                } else {
                    document.getElementById('result').innerText = `Prediction: ${data.top_digit} (${data.top_conf}%)`;
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    
    header, encoded = image_data.split(",", 1)
    binary_data = base64.b64decode(encoded)
    
    pil_img = Image.open(io.BytesIO(binary_data)).convert("L")
    
    img_tensor = preprocess_canvas_image(pil_img)
    if img_tensor is None:
        return jsonify({"error": "Canvas is empty!"})

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    probs_np = probabilities.numpy() * 100
    ranked_predictions = sorted(enumerate(probs_np), key=lambda x: x[1], reverse=True)
    
    top_digit, top_conf = ranked_predictions[0]

    return jsonify({
        "top_digit": int(top_digit),
        "top_conf": round(float(top_conf), 1)
    })

if __name__ == "__main__":
    def run_localtunnel():
        # Automatically spins up localtunnel to generate a public URL in Colab
        subprocess.run(["npx", "localtunnel", "--port", "5000"])

    threading.Thread(target=run_localtunnel, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)