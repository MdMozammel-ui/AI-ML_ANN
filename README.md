# Neural Network Assignment — Shallow vs. Deep NN (Pima Indians Diabetes)

This repository contains a fully automated PyTorch workflow that builds, tunes, and compares a **Shallow Neural Network** (1 hidden layer) and a **Deep Neural Network** (4 hidden layers) on the [Pima Indians Diabetes dataset](diabetes.csv) — a binary classification task (diabetic vs. non-diabetic).

## 🔗 Links

| Resource | Link |
|---|---|
| GitHub Repository | `<ADD YOUR REPO URL HERE>` |
| Google Colab Notebook | `<ADD YOUR COLAB URL HERE>` |
| Dataset (raw CSV) | `<ADD YOUR RAW GITHUB CSV URL HERE>` |

> ⚠️ Fill in the three links above before final submission — the PDF you submit must include them.

## 📁 Repository Contents

```
.
├── diabetes.csv          # Raw dataset (Pima Indians Diabetes)
├── 220113_ANN.ipynb      # Full Colab notebook — run via Runtime → Run all
└── README.md
```

## ⚙️ How It Runs

The notebook fetches the dataset directly from this repo's raw GitHub URL — **no manual uploads required**. Open it in Colab and select `Runtime → Run all`; it executes start to finish.

## 🧪 Pipeline

1. **Data Preprocessing**
   - Biologically impossible zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing and imputed.
   - Features are standardized with `StandardScaler`.
   - Stratified 80/20 train/test split.
2. **Shallow Neural Network** — exactly 1 hidden layer (`8 → 16 → 1`, ReLU + Sigmoid), tuned hidden units/batch size/activation via a validation split.
3. **Deep Neural Network** — 4 hidden layers (`8 → 64 → 32 → 16 → 8 → 1`) with Dropout (0.3, 0.3, 0.2) and L2 weight decay to combat overfitting; learning rate, optimizer (Adam), and epoch count tuned.
4. **Evaluation** on the held-out test set: training history curves, confusion matrices, ROC curves, and a grouped bar chart across Accuracy, Precision, Recall, F1-Score, and AUC — all rendered as side-by-side (Shallow vs. Deep) comparisons.

## 📊 Test Set Results

| Metric | Shallow NN | Deep NN |
|---|---|---|
| Accuracy | 0.7403 | 0.7597 |
| Precision | 0.6400 | 0.6735 |
| Recall | 0.5926 | 0.6111 |
| F1-Score | 0.6154 | 0.6408 |
| AUC | 0.8202 | 0.8256 |

## 🧠 Network Topology

**Shallow NN**
```
Layer 1 (Input → Hidden): Linear(8 → 16) | ReLU
Layer 2 (Hidden → Output): Linear(16 → 1) | Sigmoid
```

**Deep NN**
```
Layer 1: Linear(8 → 64)  | ReLU | Dropout(0.3)
Layer 2: Linear(64 → 32) | ReLU | Dropout(0.3)
Layer 3: Linear(32 → 16) | ReLU | Dropout(0.2)
Layer 4: Linear(16 → 8)  | ReLU
Layer 5: Linear(8 → 1)   | Sigmoid | L2 (weight_decay=1e-4)
```

## 📝 Key Finding

The Deep NN edges out the Shallow NN on every metric (notably AUC: 0.826 vs. 0.820), but the margin is modest given it has roughly 4x the layers and built-in regularization (Dropout + L2). Training/validation curves for both models track closely, indicating the regularization in the Deep NN successfully controlled overfitting rather than the extra depth translating into a large performance gain — this low-dimensional tabular dataset doesn't require much representational capacity to find a good decision boundary.

## 🛠️ Tech Stack

- PyTorch (model building & training)
- scikit-learn (preprocessing, train/test split, metrics)
- pandas / numpy (data handling)
- matplotlib / seaborn (visualizations)
