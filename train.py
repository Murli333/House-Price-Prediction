import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("house_prices.csv")

# Features & target
X = df.drop("Price", axis=1).values
y = df["Price"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Initialize parameters
n = X_train.shape[1]
w = np.zeros(n)
b = 0

# Cost function
def cost_function(X, y, w, b):
    m = X.shape[0]
    return np.sum((np.dot(X, w) + b - y) ** 2) / (2 * m)

# Gradient
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    errors = np.dot(X, w) + b - y
    dj_dw = np.dot(X.T, errors) / m
    dj_db = np.sum(errors) / m
    return dj_dw, dj_db

# Gradient descent
def gradient_descent(X, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        
        if i % 100 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i}: Cost = {cost:.2f}")
    
    return w, b

# Train model
w, b = gradient_descent(X_train, y_train, w, b, alpha=0.01, num_iters=1000)

print("✅ Training complete")

# Save model (VERY IMPORTANT: include scaling)
with open("model.pkl", "wb") as f:
    pickle.dump((w, b, mean, std), f)

print("✅ Model saved as model.pkl")