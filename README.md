# 🔩 Steel Surface Defect Classification

A deep learning project to automatically detect and classify surface defects in steel using **Transfer Learning with EfficientNetB0** on the NEU Surface Defect Dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Surface defects in steel manufacturing (crazing, scratches, pitting, etc.) are traditionally inspected manually — which is slow and error-prone. This project automates defect detection using a CNN-based image classifier.

---

## 🗂️ Project Structure

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

> ⚠️ `steel_defect_model.keras` is not included due to file size (~20MB). Train it using `Surface_Defect_Classification.ipynb` and copy it to the `Predictor/` folder.

---

## 🧠 Model Architecture

| Component  | Details                                      |
|---|---|
| Base Model | EfficientNetB0 (ImageNet weights, frozen)    |
| Pooling    | GlobalAveragePooling2D                       |
| Dense      | 256 units · ReLU · L2 regularisation (0.01) |
| Dropout    | 0.6                                          |
| Output     | 6 units · Softmax                            |

---

## 🏷️ Defect Classes

| Class | Description |
|---|---|
| `crazing` | Network of fine cracks on the surface |
| `inclusion` | Foreign material embedded in steel |
| `patches` | Irregular rough patches |
| `pitted_surface` | Small pits/holes on the surface |
| `rolled-in_scale` | Scale pressed into the surface during rolling |
| `scratches` | Linear surface scratches |

---

## 🚀 Getting Started

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

## 📊 Training Config

| Parameter    | Value                                          |
|---|---|
| Image Size   | 224×224                                        |
| Batch Size   | 16                                             |
| Epochs       | 20 (with early stopping)                       |
| Optimizer    | Adam (lr=3e-4)                                 |
| Loss         | Categorical Crossentropy (label smoothing=0.1) |
| Augmentation | Flip, Rotation, Zoom, Contrast                 |

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
- Base model: [EfficientNetB0](https://keras.io/api/applications/efficientnet/) via Keras Applications
