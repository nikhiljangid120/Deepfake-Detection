# Deepfake Image Detection using Vision Transformer (ViT)
---
## 1. Introduction & Problem Statement
### 1.1 The Problem
In the era of Generative AI, creating hyper-realistic fake images ("Deepfakes") has become dangerously easy. These images can be used for misinformation, identity theft, and digital fraud. Distinguishing between a real photograph and an AI-generated image is becoming impossible for the human eye.

### 1.2 The Objective
The goal of this project is to build an automated, robust AI system that can:
1.  Analyze an input image.
2.  Identify subtle artifacts and patterns invisible to humans.
3.  Classify the image as **"Real"** or **"Fake"** with high confidence.

### 1.3 The Solution
We utilize a **Vision Transformer (ViT)**, a cutting-edge Deep Learning architecture. Unlike traditional methods that look at local pixel features, ViTs analyze the *global context* of an image, making them highly effective at spotting the structural inconsistencies often found in deepfakes.

---

## 2. System Architecture
### 2.1 Why Vision Transformers (ViT)?
Traditional Convolutional Neural Networks (CNNs) process images portion-by-portion. While effective, they sometimes miss the "big picture."
**ViTs** work differently:
- They treat an image like a sequence of text words.
- They use **Self-Attention** mechanisms to understand how every part of the image relates to every other part.
- This allows the model to detect if the lighting on a face matches the background, or if the texture of the skin is consistent with the rest of the scene.

### 2.2 The Pipeline
The system follows a strict 4-stage pipeline:
1.  **Input Processing**:
    -   Images are resized to **224x224 pixels**.
    -   Pixel values are normalized (scaled) to ensure mathematical stability.
2.  **Patch Embedding**:
    -   The 224x224 image is sliced into **196 small patches** (each 16x16 pixels).
    -   These patches are flattened into vectors and fed into the model.
3.  **Transformer Encoder**:
    -   The core "brain" of the model. It processes these patches through multiple layers of Multi-Head Self-Attention.
    -   It learns complex relationships and features.
4.  **Classification Head**:
    -   The final output vector is passed to a Linear Classifier.
    -   It outputs a probability score for two classes: `Real` vs `Fake`.

---

## 3. Implementation Details (The Whereabouts)
The project is modularized into specific Python scripts, each handling a distinct responsibility.

### 📂 `src/` Directory
| File Name | Responsibility | Key Functions |
| :--- | :--- | :--- |
| **`model.py`** | **Model Definition** | `DeepfakeViT`: Loads the pre-trained ViT-B/16 model and modifies the final layer for binary classification. |
| **`dataset.py`** | **Data Loading** | `create_dataloaders`: Scans folders, applies augmentations (flipping/rotation), and creates batches for training. |
| **`train.py`** | **Training Logic** | `train_one_epoch`: The learning loop where the model updates its weights. `validate`: Tests performance on unseen data. |
| **`inference.py`** | **Prediction Logic** | `predict_image`: Takes a single image file, preprocesses it, and returns the Real/Fake prediction. |
| **`app.py`** | **User Interface** | Launches a **Gradio** web server. Allows users to upload images via browser and see results immediately. |
| **`demo.py`** | **Automation** | A master script that generates dummy data, trains the model, and runs inference in one click. |
| **`utils.py`** | **Utilities** | `create_dummy_data`: Generates synthetic Red/Blue images for testing. `save_checkpoint`: Saves model progress. |

### 📂 `data/` Directory
- **`train/`**: Contains subfolders `real` and `fake` for training images.
- **`val/`**: Contains subfolders `real` and `fake` for validation images.

---

## 4. Current Status & Prototype Limitations
### 4.1 Prototype State
The system currently runs as a **Functional Prototype**.
- **Code**: The Detection Pipeline (ViT -> Classification) is 100% complete and production-ready.
- **UI**: A web-based interface is fully implemented.

### 4.2 The "Synthetic Data" Context (Crucial)
Because a large scanned dataset (50GB+) was not provided, the system is currently trained on a **Synthetic Proof-of-Concept Dataset**:
- **Real** is represented by **Red-dominant** images.
- **Fake** is represented by **Blue-dominant** images.

**Implication**: If you upload a random photo, the model will classify it based on its color profile (e.g., a Red shirt = Real).
**Solution**: To deploy this for facial detection, simply download a dataset like **FaceForensics++**, place it in the `data/` folder, and run the training script. The logic remains exactly the same.

---

## 5. How to Run the System
### Prerequisites
- Python 3.10 or newer (Python 3.13 requires a Virtual Environment).
- RAM: 8GB+ recommended.

### Steps
1.  **Setup Environment** (Fixes dependency issues):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Run the Full Demo** (Trains on dummy data + Tests):
    ```bash
    python src/demo.py
    ```
3.  **Launch the Web UI**:
    ```bash
    python src/app.py
    ```
    Then open the link (e.g., `http://127.0.0.1:7860`) in your browser.

---

## 6. Future Scope
1.  **Real Data Integration**: Training on 100,000+ real deepfake faces.
2.  **Video Detection**: processing video frame-by-frame to catch deepfakes in motion.
3.  **Explainability (XAI)**: Generating "Heatmaps" to show users *exactly which pixel* looks fake.
   
