import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

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
st.markdown(
    "<h1 style='text-align: center;'>📊 Student Academic Performance Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

st.write("### 🧾 Enter Student Details")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Study Hours (per day)", 0, 12, 6)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
    sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 7)
    assignments = st.slider("📝 Assignments Completed (%)", 0, 100, 70)
    participation = st.slider("🙋 Class Participation (%)", 0, 100, 60)

with col2:
    previous_score = st.slider("📊 Previous Score (%)", 0, 100, 65)
    extra_study = st.slider("📖 Extra Study Hours", 0, 10, 3)
    internet_usage = st.slider("🌐 Internet Usage (hrs/day)", 0, 10, 4)
    health = st.slider("💪 Health Rating (1-10)", 1, 10, 7)

st.markdown("---")

# Prediction
if st.button("🚀 Predict Performance"):
    if model is None:
        st.warning("⚠️ Model not loaded properly")
    else:
        try:
            # Normalize inputs (IMPORTANT FIX)
            features = np.array([[
                study_hours / 12,
                attendance / 100,
                sleep_hours / 12,
                assignments / 100,
                participation / 100,
                previous_score / 100,
                extra_study / 10,
                internet_usage / 10,
                health / 10
            ]])

            prediction = model.predict(features)[0]

            st.markdown("## 🎯 Prediction Result")

            if prediction == 1:
                st.success("🏆 High Academic Performance")
            else:
                st.error("⚠️ Low Academic Performance")

            # Show probability if available
            try:
                proba = model.predict_proba(features)
                st.info(f"📈 Confidence: {proba[0][1]*100:.2f}% chance of High Performance")
            except:
                st.info("ℹ️ Probability not available for this model")

        except Exception as e:
            st.error("❌ Prediction failed")
            st.exception(e)
