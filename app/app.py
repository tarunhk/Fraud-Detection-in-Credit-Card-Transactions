import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load model
model = joblib.load("models/xgb_model.pkl")

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Title
st.title("💳 Credit Card Fraud Detection System")

# Sidebar
st.sidebar.header("Input Transaction Features")

# Create inputs
input_data = []
for i in range(30):
    val = st.sidebar.number_input(f"Feature {i}", value=0.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_array)[0]
    prob = model.predict_proba(input_array)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Fraud Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Normal Transaction (Confidence: {prob:.2f})")

# Show graphs
st.subheader("Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    st.image("outputs/confusion_matrix.png")

with col2:
    st.write("ROC Curve")
    st.image("outputs/roc_curve.png")

st.subheader("Explainability")
st.image("outputs/shap_summary.png")
st.image("outputs/feature_importance.png")