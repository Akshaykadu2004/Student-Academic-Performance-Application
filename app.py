import streamlit as st
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error("❌ Model loading failed")
        st.exception(e)
        return None

model = load_model()

# Title
st.title("📊 Student Academic Performance Predictor")
st.write("Enter student details below:")

# 9 Feature Inputs (replace names if you know actual ones)
f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)
f5 = st.number_input("Feature 5", value=0.0)
f6 = st.number_input("Feature 6", value=0.0)
f7 = st.number_input("Feature 7", value=0.0)
f8 = st.number_input("Feature 8", value=0.0)
f9 = st.number_input("Feature 9", value=0.0)

# Predict button
if st.button("Predict"):
    if model is None:
        st.warning("⚠️ Model not loaded properly")
    else:
        try:
            features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9]])
            prediction = model.predict(features)

            st.success(f"🎯 Prediction: {prediction[0]}")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)
