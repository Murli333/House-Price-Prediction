# House-Price-Prediction
# 🏠 House Price Prediction using Multiple Linear Regression

## 📌 Overview

This project implements a **House Price Prediction system** using **Multiple Linear Regression trained via Gradient Descent**.
The model predicts house prices based on features like area, number of bedrooms, and bathrooms.

Unlike basic implementations, this project **builds the regression model from scratch**, including:

* Cost Function (Mean Squared Error)
* Gradient Descent optimization
* Parameter tuning

---

## ⚙️ Tech Stack

* **Python**
* **NumPy**
* **Pandas**
* **Pickle** (for model saving/loading)

---

## 📂 Project Structure

```
├── app.py               # Application interface (prediction)
├── train.py             # Model training using gradient descent
├── generate_data.py     # Synthetic dataset generation
├── model.pkl            # Saved trained model
├── requirements.txt     # Dependencies
```

---

## 🧠 Model Details

### 📊 Multiple Linear Regression

The model predicts price using:

[
Price = w_1 \cdot Area + w_2 \cdot Bedrooms + w_3 \cdot Bathrooms + b
]

---

### 📉 Cost Function (MSE)

[
J(w, b) = \frac{1}{2m} \sum (y_{pred} - y_{actual})^2
]

---

### 🔄 Gradient Descent Update Rule

Weights and bias are updated iteratively:

[
w = w - \alpha \cdot \frac{\partial J}{\partial w}
]
[
b = b - \alpha \cdot \frac{\partial J}{\partial b}
]

Where:

* ( \alpha ) = Learning Rate
* ( m ) = Number of training samples

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Murli333/House-Price-Prediction.git
cd House-Price-Prediction
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train the model

```bash
python train.py
```

---

### 4. Run the app

```bash
python app.py
```

---

## 📈 Features

* Custom implementation of Gradient Descent
* Synthetic dataset generation
* Model persistence using Pickle
* Simple prediction interface

---

## ⚠️ Limitations

* Uses synthetic data (not real-world dataset)
* No feature scaling (may affect accuracy)
* Limited features (can be extended)

---

## 🔮 Future Improvements

* Add feature scaling (Standardization/Normalization)
* Use real datasets (e.g., Kaggle Housing)
* Upgrade to advanced models (Random Forest, XGBoost)
* Deploy using Streamlit or Flask

---

## 💡 Key Learning Outcomes

* Deep understanding of Gradient Descent
* Implementation of cost functions from scratch
* End-to-end ML workflow (data → training → prediction → deployment)

---

## 📬 Author

Murli Mishra

---
