import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy.ndimage import center_of_mass

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

model.eval()
print("Training complete! Opening interactive drawing window...")

# ==========================================
# 3. GUI with Ordered Probabilities Output
# ==========================================
class DigitRecognizerGUI:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognizer")

        # Main horizontal container
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # Left panel: Drawing canvas & control buttons
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

        # Right panel: Rank-ordered probability list
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

        # Top summary label below the drawing board
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

    def center_image_by_mass(self, img_np):
        cy, cx = center_of_mass(img_np)
        if np.isnan(cy) or np.isnan(cx):
            return img_np
        
        rows, cols = img_np.shape
        shift_x = np.round(cols / 2.0 - cx).astype(int)
        shift_y = np.round(rows / 2.0 - cy).astype(int)
        
        return np.roll(img_np, (shift_y, shift_x), axis=(0, 1))

    def predict_digit(self):
        bbox = self.image.getbbox()
        if bbox is None:
            self.label_result.config(text="Canvas is empty!")
            self.proba_label.config(text="Canvas is empty!")
            return

        # Preprocessing pipeline
        blurred = self.image.filter(ImageFilter.GaussianBlur(radius=1.5))
        cropped = blurred.crop(bbox)
        cropped.thumbnail((20, 20), Image.Resampling.BILINEAR)

        centered_img = Image.new("L", (28, 28), "black")
        offset = ((28 - cropped.width) // 2, (28 - cropped.height) // 2)
        centered_img.paste(cropped, offset)

        img_array = np.array(centered_img, dtype=np.float32) / 255.0
        img_array = self.center_image_by_mass(img_array)

        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) / 0.5

        # Run inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)[0]

        # Convert probabilities to a sorted list of (digit, percentage) tuples
        probs_np = probabilities.numpy() * 100
        ranked_predictions = sorted(enumerate(probs_np), key=lambda x: x[1], reverse=True)

        # Build side panel ranking text
        ranking_text = "Rank  Digit  Confidence\n" + "-" * 24 + "\n"
        for rank, (digit, prob) in enumerate(ranked_predictions, 1):
            ranking_text += f" #{rank:<2}    [{digit}]     {prob:5.1f}%\n"

        self.proba_label.config(text=ranking_text)

        # Update summary prediction label
        top_digit, top_conf = ranked_predictions[0]
        second_digit, second_conf = ranked_predictions[1]
        self.label_result.config(
            text=f"Prediction: {top_digit} ({top_conf:.1f}%)  |  Runner-up: {second_digit} ({second_conf:.1f}%)"
        )

if __name__ == "__main__":
    DigitRecognizerGUI(model)