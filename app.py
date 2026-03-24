import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("DEBUG PATH:", BASE_DIR)

model_path = os.path.join(BASE_DIR, "model.pkl")
print("MODEL PATH:", model_path)
print("EXISTS?", os.path.exists(model_path))
import streamlit as st
import numpy as np

import pickle


with open(model_path, "rb") as f:
    w, b, mean, std = pickle.load(f)
    print(model_path, "loaded successfully")

st.title("🏠 House Price Predictor")

st.write("Enter house details:")

# Inputs
area = st.number_input("Area (sqft)", min_value=0.0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
age = st.number_input("Age of house", min_value=0)
location_score = st.slider("Location Score", 1, 10)
garage = st.selectbox("Garage", [0, 1])

# Prediction
if st.button("Predict Price"):
    features = np.array([area, bedrooms, bathrooms, age, location_score, garage])
    
    # APPLY SAME SCALING (CRITICAL)
    features = (features - mean) / std
    
    predicted_price = np.dot(features, w) + b
    
    st.success(f"🏠 Predicted Price: ₹{predicted_price:,.2f}")