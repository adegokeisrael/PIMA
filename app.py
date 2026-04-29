"""
🧠 PIMA Diabetes Predictor — Streamlit App
Loads the trained ANN model and scaler saved from the notebook.
"""

import streamlit as st
import numpy as np
import pickle
import os

# ── TensorFlow / Keras import (graceful fallback message) ──────────────────
try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🧠 Diabetes ANN Predictor",
    page_icon="🧠",
    layout="centered",
)

# ══════════════════════════════════════════════════════════════════════════════
# Load model & scaler  (cached so they load only once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    """Load the Keras model and the fitted StandardScaler."""
    errors = []

    # ── Scaler ────────────────────────────────────────────────────────────
    scaler = None
    for scaler_path in ["scaler.pkl", "model_bundle.pkl"]:
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                obj = pickle.load(f)
            scaler = obj["scaler"] if isinstance(obj, dict) else obj
            break
    if scaler is None:
        errors.append("❌ `scaler.pkl` not found. Run the notebook first.")

    # ── Keras model ───────────────────────────────────────────────────────
    model = None
    if TF_AVAILABLE:
        for model_path in ["diabetes_ann_model.h5", "best_diabetes_model.keras"]:
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                break
        if model is None:
            errors.append("❌ `diabetes_ann_model.h5` not found. Run the notebook first.")
    else:
        errors.append("❌ TensorFlow is not installed.")

    return model, scaler, errors


model, scaler, load_errors = load_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 PIMA Diabetes ANN Predictor")
st.markdown(
    """
    This app uses a **deep Artificial Neural Network** trained on the
    [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
    to predict the probability of diabetes from 8 clinical features.
    """
)
st.divider()

# ── Show loading errors if any ────────────────────────────────────────────
if load_errors:
    for err in load_errors:
        st.error(err)
    st.info(
        "Run all cells in **brain-ann.ipynb** (including the new *Save Model* cell) "
        "to generate `diabetes_ann_model.h5` and `scaler.pkl`, "
        "then place them in the same folder as this `app.py`."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar — About
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Feedforward ANN (MLP)
        
        **Architecture:**
        ```
        Input(8)
          ↓
        Dense(64) + BN + ReLU + Dropout
          ↓
        Dense(32) + BN + ReLU + Dropout
          ↓
        Dense(16) + ReLU + Dropout
          ↓
        Dense(1) → Sigmoid
        ```
        **Framework:** TensorFlow / Keras  
        **Dataset:** PIMA Indians (768 samples)
        """
    )
    st.divider()
    st.markdown(
        "⚠️ *This tool is for educational purposes only and does not constitute "
        "medical advice.*"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Input form
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Enter Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration (2-hr oral glucose tolerance test)"
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0, max_value=150, value=70,
        help="Diastolic blood pressure"
    )
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin (μU/mL)",
        min_value=0, max_value=1000, value=80,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body mass index"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.000, max_value=3.000, value=0.470, step=0.001,
        format="%.3f",
        help="Genetic diabetes risk score based on family history"
    )
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=30, step=1
    )

# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

# Replace biologically implausible zeros with median (mirrors notebook preprocessing)
MEDIANS = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 23.0,
    "Insulin": 30.5,
    "BMI": 32.0,
}

raw_values = {
    "Pregnancies": pregnancies,
    "Glucose": glucose if glucose > 0 else MEDIANS["Glucose"],
    "BloodPressure": blood_pressure if blood_pressure > 0 else MEDIANS["BloodPressure"],
    "SkinThickness": skin_thickness if skin_thickness > 0 else MEDIANS["SkinThickness"],
    "Insulin": insulin if insulin > 0 else MEDIANS["Insulin"],
    "BMI": bmi if bmi > 0 else MEDIANS["BMI"],
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_array = np.array([[
        raw_values["Pregnancies"],
        raw_values["Glucose"],
        raw_values["BloodPressure"],
        raw_values["SkinThickness"],
        raw_values["Insulin"],
        raw_values["BMI"],
        raw_values["DiabetesPedigreeFunction"],
        raw_values["Age"],
    ]])

    # Scale
    input_scaled = scaler.transform(input_array)

    # Predict
    probability = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction  = int(probability >= 0.5)

    # ── Result display ────────────────────────────────────────────────────
    st.subheader("📊 Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("### 🔴 Diabetic")
        else:
            st.success("### 🟢 Non-Diabetic")

    with res_col2:
        st.metric(
            label="Probability of Diabetes",
            value=f"{probability * 100:.1f}%",
            delta=f"{'↑ Above' if probability >= 0.5 else '↓ Below'} 50% threshold",
        )

    # Probability bar
    st.progress(probability)

    # Confidence level
    if probability > 0.75 or probability < 0.25:
        conf_label, conf_color = "High confidence", "🟢"
    elif probability > 0.60 or probability < 0.40:
        conf_label, conf_color = "Medium confidence", "🟡"
    else:
        conf_label, conf_color = "Low confidence (near boundary)", "🔴"

    st.caption(f"{conf_color} Confidence: **{conf_label}**")

    # ── Input summary ──────────────────────────────────────────────────────
    with st.expander("🔎 Input values used for prediction"):
        import pandas as pd
        summary_df = pd.DataFrame(
            list(raw_values.items()), columns=["Feature", "Value"]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Disclaimer ─────────────────────────────────────────────────────────
    st.divider()
    st.warning(
        "⚠️ This prediction is generated by a machine learning model trained on a "
        "small dataset. It is **not a medical diagnosis**. Please consult a qualified "
        "healthcare professional for any health concerns."
    )
