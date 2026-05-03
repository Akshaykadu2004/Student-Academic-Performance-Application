import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="📊", layout="centered")

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
st.markdown("<h1 style='text-align: center;'>📊 Student Academic Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

st.write("### 🧾 Enter Student Details")

# Use columns for better UI
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Study Hours (per day)", 0, 12, 4)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
    sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 6)
    assignments = st.slider("📝 Assignments Completed (%)", 0, 100, 70)
    participation = st.slider("🙋 Class Participation (%)", 0, 100, 60)

with col2:
    previous_score = st.slider("📊 Previous Score (%)", 0, 100, 50)
    extra_study = st.slider("📖 Extra Study Hours", 0, 10, 2)
    internet_usage = st.slider("🌐 Internet Usage (hrs/day)", 0, 10, 3)
    health = st.slider("💪 Health Rating (1-10)", 1, 10, 7)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Performance"):
    if model is None:
        st.warning("⚠️ Model not loaded properly")
    else:
        try:
            # Arrange features (must match training order!)
            features = np.array([[
                study_hours,
                attendance,
                sleep_hours,
                assignments,
                participation,
                previous_score,
                extra_study,
                internet_usage,
                health
            ]])

            prediction = model.predict(features)[0]

            # Display result nicely
            st.markdown("## 🎯 Prediction Result")

            if prediction == 1:
                st.success("🏆 High Academic Performance")
            else:
                st.error("⚠️ Low Academic Performance")

            # Probability (if available)
            try:
                proba = model.predict_proba(features)
                st.info(f"📈 Confidence: {proba[0][1]*100:.2f}% chance of High Performance")
            except:
                st.info("ℹ️ Probability not available for this model")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)
