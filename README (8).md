# Emotion Detection with PyTorch 😠😊😢

A deep learning project that detects human emotions from facial images using a Convolutional Neural Network (CNN) built with PyTorch.

## 📋 Overview

This project trains a CNN model on the FER2013 dataset to classify facial expressions into 7 emotion categories in real time.

## 🎭 Emotion Classes

| Emotion | Label |
|---------|-------|
| 😠 Angry | 0 |
| 🤢 Disgust | 1 |
| 😨 Fear | 2 |
| 😊 Happy | 3 |
| 😐 Neutral | 4 |
| 😢 Sad | 5 |
| 😲 Surprise | 6 |

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~72% |
| Test Accuracy | ~59% |
| Dataset | FER2013 (28,709 train / 7,178 test) |
| Epochs | 10 |

## 🏗️ Model Architecture (CNN)

```
Conv2d(1 → 32)  + ReLU + MaxPool
Conv2d(32 → 64) + ReLU + MaxPool
Conv2d(64 → 128)+ ReLU + MaxPool
Dropout(0.25)
Linear(4608 → 256) + ReLU
Dropout(0.25)
Linear(256 → 7)
```

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — model building and training
- **torchvision** — data loading and transforms
- **Google Colab** — training environment
- **Kaggle API** — dataset download

## 🚀 How It Works

1. Download FER2013 dataset from Kaggle
2. Load images using `torchvision.datasets.ImageFolder`
3. Apply transforms: grayscale, resize to 48×48, normalize
4. Train a 3-layer CNN for 10 epochs
5. Evaluate on unseen test images
6. Predict emotion on sample images

## 📁 Project Structure

```
emotion-detection-pytorch/
└── emotion_detection_pytorch.ipynb   # Full notebook: setup, training, evaluation
```

## 🔮 Possible Improvements

- Use data augmentation (flips, rotations) to reduce overfitting
- Add more epochs with early stopping
- Try transfer learning with a pre-trained model (e.g. ResNet)
- Deploy as a real-time webcam app using OpenCV

## 👤 Author

**Negin Mohammadshahi**
AI & Software Engineer
[Portfolio](https://negin-mohammadshahi.github.io/) | [LinkedIn](https://www.linkedin.com/in/negin-mohammadshahi-908b3a79) | [GitHub](https://github.com/negin-mohammadshahi)
