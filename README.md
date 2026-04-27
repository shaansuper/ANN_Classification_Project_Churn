## 📌 Project Overview

Customer churn prediction helps businesses identify customers likely to leave. This project trains a binary classification ANN on a churn dataset and serves predictions through a user-friendly Streamlit frontend.

---

## 🚀 Live Demo

>
https://churn-customer-project.streamlit.app/
---

## 🗂️ Project Structure

├── app.py # Streamlit frontend application
├── model.h5 # Trained ANN model (saved Keras model)
├── notebook.ipynb # EDA, preprocessing & model training notebook
├── churn.csv # Dataset (e.g., Telco / Bank Customer Churn)
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── scaler.pkl # Saved preprocessing objects
├── requirements.txt # Python dependencies
├── logs/ # TensorBoard training logs
├── runtime.txt/ #Python Environment Version
└── README.md


---

## 🧰 Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Language     | Python 3.10+                      |
| Deep Learning| TensorFlow 2.15.0 / Keras         |
| Data Handling| Pandas, NumPy                     |
| ML Utilities | Scikit-learn                      |
| Monitoring   | TensorBoard                       |
| Frontend     | Streamlit                         |

---

## 📊 Dataset

- **Source:** [Kaggle — Bank Customer Churn Dataset](Provided in the Repository)
- **Target Column:** `Exited` (1 = Churned, 0 = Retained)
- **Features:** Geography, Gender, Age, Balance, Credit Score, Tenure, etc.

---

## 🏗️ Model Architecture

Input Layer → (n_features,)
Dense Layer 1 → 64 units, ReLU activation
Dense Layer 2 → 32 units, ReLU activation
Output Layer → 1 unit, Sigmoid activation


- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  
- **Callbacks:** EarlyStopping, TensorBoard

---

