import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load('gmm_model.pkl')

try:
    scaler = joblib.load('scaler.pkl')
    use_scaler = True
except:
    use_scaler = False

st.set_page_config(page_title="Customer Segmentation (GMM)", layout="centered")
st.title("🧠 Customer Segmentation with GMM")
st.write("Enter customer details below to predict the segment.")

# Inputs
age = st.number_input("Age", 10, 100, 30)
income = st.number_input("Annual Income", 1000, 200000, 50000)
score = st.slider("Spending Score", 1, 100, 50)

features = np.array([[age, income, score]])

if st.button("Predict Segment"):
    try:
        if use_scaler:
            features = scaler.transform(features)
        segment = model.predict(features)[0]
        st.success(f"🎯 Predicted Segment: {segment}")
    except Exception as e:
        st.error(f"Error: {e}")
