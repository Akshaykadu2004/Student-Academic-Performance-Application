import streamlit as st
import numpy as np
import joblib

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

st.title("Student Academic Performance Predictor")
st.write("Enter student details:")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
    if model is None:
        st.warning("Model not loaded properly")
    else:
        features = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(features)
        st.success(f"Prediction: {prediction[0]}")
