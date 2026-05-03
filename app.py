import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title Section
# -------------------------------
st.markdown('<div class="title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter student details to predict performance</div>', unsafe_allow_html=True)

# -------------------------------
# Input Section (Direct typing)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    study_hours = st.text_input("📘 Study Hours")
    attendance = st.text_input("📊 Attendance (%)")

with col2:
    previous_score = st.text_input("📝 Previous Score")
    sleep_hours = st.text_input("😴 Sleep Hours")

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Performance"):
    try:
        # Convert inputs to float
        features = np.array([[
            float(study_hours),
            float(attendance),
            float(previous_score),
            float(sleep_hours)
        ]])

        prediction = model.predict(features)

        st.success(f"📊 Predicted Performance: {prediction[0]}")

    except:
        st.error("❌ Please enter valid numeric values in all fields")
