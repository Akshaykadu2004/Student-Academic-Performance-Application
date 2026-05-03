import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Load Model
# -------------------------------
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

# -------------------------------
# Title
# -------------------------------
st.title("Student Academic Performance Predictor")

st.write("Enter student details:")

# -------------------------------
# INPUTS (EDIT THESE if needed)
# -------------------------------
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    if model is None:
        st.warning("Model not loaded properly")
    else:
        try:
            features = np.array([[feature1, feature2, feature3, feature4]])
            prediction = model.predict(features)

            st.success(f"Prediction: {prediction[0]}")

        except Exception as e:
            st.error("Prediction failed")
            st.exception(e)
