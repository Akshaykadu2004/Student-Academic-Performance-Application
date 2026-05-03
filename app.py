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

# Inputs (replace with actual feature names if known)
study_hours = st.number_input("Study Hours", min_value=0.0, value=2.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, value=6.0)
previous_score = st.number_input("Previous Score", min_value=0.0, value=50.0)

# Prediction
if st.button("Predict"):
    if model is None:
        st.warning("⚠️ Model not loaded properly")
    else:
        try:
            features = np.array([[study_hours, attendance, sleep_hours, previous_score]])
            prediction = model.predict(features)

            st.success(f"🎯 Predicted Performance: {prediction[0]}")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)
