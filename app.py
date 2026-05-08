import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load model and scaler ──
model = joblib.load('milk_model.pkl')
scaler = joblib.load('milk_scaler.pkl')

# ── Page config ──
st.set_page_config(
    page_title="Milk Adulteration Detector",
    page_icon="🥛",
    layout="centered"
)

# ── Header ──
st.title("🥛 Milk Adulteration Detector")
st.markdown("Enter the measured properties of the milk sample below.")
st.markdown("---")

# ── Input Section ──
st.subheader("📋 Enter Milk Sample Properties")

col1, col2 = st.columns(2)

with col1:
    ph = st.slider(
        "pH Level",
        min_value=3.0,
        max_value=10.0,
        value=6.6,
        step=0.1,
        help="Normal milk pH is between 6.4 to 6.8"
    )

    temperature = st.slider(
        "Temperature (°C)",
        min_value=34,
        max_value=90,
        value=35,
        step=1,
        help="Temperature at which sample was tested"
    )

    fat = st.selectbox(
        "Fat Present?",
        options=[1, 0],
        format_func=lambda x: "Yes ✅" if x == 1 else "No ❌",
        help="Is fat present in the milk sample?"
    )

with col2:
    taste = st.selectbox(
        "Taste",
        options=[1, 0],
        format_func=lambda x: "Good 👍" if x == 1 else "Bad 👎",
        help="Does the milk taste normal?"
    )

    odor = st.selectbox(
        "Odor",
        options=[0, 1],
        format_func=lambda x: "Normal ✅" if x == 0 else "Abnormal ⚠️",
        help="Does the milk smell normal?"
    )

    turbidity = st.selectbox(
        "Turbidity",
        options=[0, 1],
        format_func=lambda x: "Clear ✅" if x == 0 else "Cloudy ⚠️",
        help="Is the milk clear or cloudy?"
    )

colour = st.slider(
    "Colour Value",
    min_value=240,
    max_value=255,
    value=254,
    step=1,
    help="Visual colour value of the milk sample"
)

st.markdown("---")

# ── Predict Button ──
if st.button("🔍 Analyse Milk Sample", use_container_width=True):

    # Build input dataframe
    input_data = pd.DataFrame([{
        'pH': ph,
        'Temprature': temperature,
        'Taste': taste,
        'Odor': odor,
        'Fat': fat,
        'Turbidity': turbidity,
        'Colour': colour
    }])

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Confidence score
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_scaled)[0]
        confidence = round(max(proba) * 100, 2)
    else:
        confidence = None

    # ── Show Result ──
    st.markdown("## 🧪 Result")

    if prediction == 2:
        st.success("✅ PURE MILK — Safe to consume!")
        st.balloons()
    elif prediction == 1:
        st.warning("⚠️ AT RISK — Borderline quality. Lab verification recommended.")
    else:
        st.error("❌ ADULTERATED — Not safe to consume!")

    # Confidence
    if confidence:
        st.metric(label="Model Confidence", value=f"{confidence}%")

    # Input summary
    with st.expander("📊 See what you entered"):
        st.dataframe(input_data)

# ── Footer ──
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning | Random Forest Classifier | Dataset: Kaggle Milk Quality")