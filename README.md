# Steel Surface Defect Classification

A deep learning project to automatically detect and classify surface defects in steel using **Transfer Learning with EfficientNetB0** on the NEU Surface Defect Dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Problem Statement

Surface defects in steel manufacturing (crazing, scratches, pitting, etc.) are traditionally inspected manually — which is slow and error-prone. This project automates defect detection using a CNN-based image classifier.

---

## About the Project

Steel is one of the most widely used materials in industries like construction, automotive, and manufacturing. During the production process, various surface defects can occur that compromise the quality and structural integrity of the final product. Detecting these defects early is critical — but doing it manually is time-consuming, inconsistent, and expensive.

This project builds an **automated surface defect detection system** using deep learning. Given an image of a steel surface, the model predicts which type of defect is present.

### How It Works

The pipeline has two stages:

**1. Training (`Model_Training/`)**
- Loads the NEU-DET dataset which contains 1,800 grayscale images across 6 defect categories
- Preprocesses images — resizing to 224×224 and building a clean train/val/test split
- Applies data augmentation (flipping, rotation, zoom, contrast) to improve generalisation
- Uses **EfficientNetB0** pretrained on ImageNet as a frozen feature extractor — reusing patterns the model already learned from millions of images instead of training from scratch
- Adds a custom classification head on top and trains only that part
- Evaluates performance using accuracy, classification report, and confusion matrix
- Saves the trained model and class names for inference

**2. Inference (`Predictor/`)**
- Loads the saved model and class names
- Takes any steel surface image as input
- Preprocesses it the same way as training
- Returns the predicted defect class and confidence score with a full probability table

### Why EfficientNetB0?

Training a deep CNN from scratch requires a huge dataset and a lot of compute. Instead, we use **Transfer Learning** — EfficientNetB0 was already trained on 1.2 million images (ImageNet) and learned powerful low-level features like edges, textures, and shapes. We freeze those weights and only train the final layers to recognise steel defects. This gives strong performance even with a small dataset of ~1,800 images.

---

## Project Structure

```
steel-surface-defect-classification/
│
├── Model_Training/
│   ├── NEU-DET/                             # Dataset (download from Kaggle)
│   │   ├── train/images/
│   │   └── validation/images/
│   ├── class_names.json                     # Saved class labels
│   ├── steel_defect_model.keras             # Trained model (generated after training)
│   └── Surface_Defect_Classification.ipynb  # Training pipeline
│
├── Predictor/
│   ├── Inference_IMG/                       # Put your test images here
│   ├── class_names.json                     # Saved class labels
│   ├── steel_defect_model.keras             # Trained model (copy from Model_Training)
│   └── Surface_Defect_Inference.ipynb       # Inference only
│
├── class_names.json                         # Class labels
├── LICENSE
└── README.md
```

---

## Model Architecture

| Component  | Details                                      |
|---|---|
| Base Model | EfficientNetB0 (ImageNet weights, frozen)    |
| Pooling    | GlobalAveragePooling2D                       |
| Dense      | 256 units · ReLU · L2 regularisation (0.01) |
| Dropout    | 0.6                                          |
| Output     | 6 units · Softmax                            |

---

## Defect Classes

| Class | Description |
|---|---|
| `crazing` | Network of fine cracks on the surface |
| `inclusion` | Foreign material embedded in steel |
| `patches` | Irregular rough patches |
| `pitted_surface` | Small pits/holes on the surface |
| `rolled-in_scale` | Scale pressed into the surface during rolling |
| `scratches` | Linear surface scratches |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/steel-surface-defect-classification.git
cd steel-surface-defect-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the **NEU Surface Defect Dataset** from [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) and place it inside `Model_Training/` as:
```
Model_Training/
└── NEU-DET/
    ├── train/images/
    └── validation/images/
```

### 4. Train the model
Open and run `Model_Training/Surface_Defect_Classification.ipynb`
This will generate `steel_defect_model.keras` and `class_names.json`.

### 5. Run inference
- Copy `steel_defect_model.keras` and `class_names.json` to `Predictor/`
- Add your test images to `Predictor/Inference_IMG/`
- Open and run `Predictor/Surface_Defect_Inference.ipynb`

---

## ☁️ Running on Google Colab

Both notebooks are Colab-ready. To persist files across sessions, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = "/content/drive/MyDrive/SteelDefectProject/"
```

To download results after inference:
```python
!zip -r prediction_results.zip predictions.csv class_names.json

from google.colab import files
files.download("prediction_results.zip")
```

---

## Training Config

| Parameter    | Value                                          |
|---|---|
| Image Size   | 224×224                                        |
| Batch Size   | 16                                             |
| Epochs       | 20 (with early stopping)                       |
| Optimizer    | Adam (lr=3e-4)                                 |
| Loss         | Categorical Crossentropy (label smoothing=0.1) |
| Augmentation | Flip, Rotation, Zoom, Contrast                 |

---

## License

This project is open source under the [MIT License](LICENSE).

---

## Acknowledgements

- Dataset: [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
- Base model: [EfficientNetB0](https://keras.io/api/applications/efficientnet/) via Keras Applications
