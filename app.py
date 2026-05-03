import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🎓 Student Performance Predictor")
st.write("Enter student details:")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.text_input("📘 Study Hours")
    attendance = st.text_input("📊 Attendance (%)")

with col2:
    previous_score = st.text_input("📝 Previous Score")
    sleep_hours = st.text_input("😴 Sleep Hours")

# -------------------------------
# Validation Function
# -------------------------------
def is_valid_number(value):
    try:
        float(value)
        return True
    except:
        return False

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Performance"):

    inputs = [study_hours, attendance, previous_score, sleep_hours]

    # Check empty fields
    if any(val.strip() == "" for val in inputs):
        st.warning("⚠️ Please fill all fields")
    
    # Check numeric values
    elif not all(is_valid_number(val) for val in inputs):
        st.error("❌ Only numeric values are allowed")
    
    else:
        features = np.array([[
            float(study_hours),
            float(attendance),
            float(previous_score),
            float(sleep_hours)
        ]])

        prediction = model.predict(features)
        st.success(f"📊 Predicted Performance: {prediction[0]}")
