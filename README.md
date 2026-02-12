
---

# 🚀 Cryptocurrency Closing Price Predictor

A production-ready Machine Learning application that predicts cryptocurrency closing prices using a **Random Forest K-Fold Ensemble**, deployed with **Streamlit** for an interactive analytical experience.

🔗 **Live App:**
[https://crypto-price-ml-dashboard-esvsawgjtfappe8tms4pguw.streamlit.app/](https://crypto-price-ml-dashboard-esvsawgjtfappe8tms4pguw.streamlit.app/)

---

## 📌 Overview

This project delivers a reliable and production-aligned cryptocurrency price prediction system.
The application mirrors the exact training pipeline used during model development to ensure **feature parity, reproducibility, and trust in predictions**.

It is designed for:

* Data Analysts
* Traders
* ML Engineers
* Financial Technology Enthusiasts

---

## ✨ Key Features

### ✅ Ensemble-Based Prediction

* Utilizes a **Random Forest K-Fold Ensemble**
* Loads serialized pipeline (`crypto_kfold_pipeline.pkl`)
* Averages predictions across folds for stability and robustness

### 📊 Intelligent Feature Engineering

* Log transformations
* Price spreads & volatility metrics
* Ratio-based financial indicators
* Statistical interactions

All features are reconstructed in real-time to match the original training configuration.

### 📈 Professional Analytical UI

* Clean Streamlit interface
* Input validation and sanitization
* Dynamic matplotlib visualizations
* Instant trend direction analysis
* Desktop and Streamlit Cloud optimized layout

### 📉 Market Insight

Alongside the predicted closing price, the app provides:

* Market Trend Direction (Bullish / Bearish)
* Visual comparison against opening price
* Immediate analytical interpretation

---

## 🏗 System Architecture

```
User Input
   ↓
Input Validation & Sanitization
   ↓
Feature Engineering (Reconstructed Signals)
   ↓
K-Fold Random Forest Inference
   ↓
Fold Averaging
   ↓
Trend Analysis + Visualization
```

### 1️⃣ Input Processing

* Converts inputs to finite float values
* Prevents division-by-zero errors
* Ensures feature completeness

### 2️⃣ Feature Reconstruction

The application regenerates engineered features to:

* Maintain exact training-time feature order
* Guarantee compatibility with serialized model

### 3️⃣ Ensemble Inference

* Each fold predicts independently
* Final output = Mean prediction across folds
* Improved generalization & reduced variance

---

## 🧠 Model & Data

* **Model Type:** Random Forest Regressor (K-Fold Ensemble)
* **Data Source:** Zindi Africa Cryptocurrency Dataset
* **Evaluation Strategy:** K-Fold Cross-Validation
* **Deployment:** Streamlit Cloud

The serialized model file is located at:

```
model/crypto_kfold_pipeline.pkl
```

It contains:

* Trained fold models
* Ordered feature list
* Complete inference-ready pipeline

---

## ⚙️ Installation & Local Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd crypto_ml
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🖥 Application Preview

### Input Interface

![Input Form](photo/Screenshot%202026-02-11%20142804.png)

### Prediction Output

![Prediction Results](photo/Screenshot%202026-02-11%20143146.png)

---

## 🚀 Deployment

The application is deployed on **Streamlit Cloud**.

To redeploy:

1. Push updates to GitHub
2. Ensure `model/crypto_kfold_pipeline.pkl` is included
3. Streamlit Cloud will auto-build from `requirements.txt`

---

## 🔒 Production Considerations

* Consistent feature ordering maintained
* Zero-division safeguards
* Missing engineered features defaulted safely
* Ensemble averaging for prediction stability
* Fully reproducible inference pipeline

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── model/
│   └── crypto_kfold_pipeline.pkl
├── photo/
│   └── screenshots
└── README.md
```

---

## 📬 Contact

**Segni Nadew**
Machine Learning & Data Science Enthusiast

If you’d like to collaborate, discuss improvements, or explore deployment strategies, feel free to open an issue or connect via GitHub.

---

## 🏆 Tech Stack

* Python
* Scikit-Learn
* Random Forest
* K-Fold Cross Validation
* Streamlit
* Matplotlib
* Zindi Competition Data

---

# 💡 Why This Project Stands Out

This is not just a notebook model — it is:

* A reproducible ML pipeline
* A deployed production-grade application
* A feature-consistent inference system
* A portfolio-ready end-to-end ML project

---


