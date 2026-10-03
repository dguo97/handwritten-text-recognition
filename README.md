# Handwritten Text Recognition

A PyTorch implementation of a Convolutional Neural Network (CNN) designed for handwritten character and digit recognition.

---

## 🚀 How to Run in Google Colab

You can run and test this project directly in Google Colab using free GPU acceleration without configuring a local environment.

1. **Open Google Colab:** Click the badge below or visit [colab.research.google.com](https://colab.research.google.com).

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dguo97/handwritten-text-recognition/blob/main/main.py)

2. **Setup and Run:** Create a new notebook (`+ New notebook`) and run the following commands in a code cell:

```bash
# Clone repository and enter project directory
!git clone [https://github.com/dguo97/handwritten-text-recognition.git](https://github.com/dguo97/handwritten-text-recognition.git)
%cd handwritten-text-recognition

# Install dependencies
!pip install -r requirements.txt

# Run script
!python main.py
```

---

## 📊 Dataset Overview

The project uses the classic **MNIST Handwritten Digit Dataset** sourced from Kaggle:
* **Dataset Link:** [Kaggle - MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### Dataset Features:
* **Classes:** 10 classes corresponding to handwritten digits (`0` through `9`).
* **Format:** 28x28 single-channel grayscale images.
* **Volume:** 70,000 images in total (60,000 training images and 10,000 testing images).
* **Storage:** Placed inside the local `data/` folder and preprocessed into normalized tensors for model training.

---

## 🧠 How It Works

The model follows a standard deep-learning pipeline for image classification:

1. **Feature Extraction (Convolutional Layers):**
   * Input images pass through 2D Convolutional layers (`nn.Conv2d`) combined with activation functions (ReLU) to extract low-level and high-level visual features (edges, curves, and character shapes).
   * Pooling layers (`nn.MaxPool2d`) reduce spatial dimensions while retaining key features, making the model computationally efficient and invariant to slight positional shifts.

2. **Classification (Fully Connected Layers):**
   * Extracted feature maps are flattened into a single feature vector.
   * Fully connected (Dense) layers map these features to output scores representing predicted character or digit classes.

3. **Training & Optimization:**
   * **Loss Function:** Cross-Entropy Loss evaluates the difference between predicted class probabilities and ground-truth targets.
   * **Optimizer:** Adam/SGD iteratively updates network weights via backpropagation to minimize error.