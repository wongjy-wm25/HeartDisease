"""
Heart Disease Prediction - Streamlit App
=========================================
Loads the trained KNN and SVM models + scaler (produced by train_model.py)
and lets the user enter patient information to get predictions from both
models, following the system flowchart in Section 3.1 of the report:
    Enter patient data -> validate input -> apply saved scaler ->
    predict with both models -> display predicted class + probability
"""

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os

st.set_page_config(page_title="Heart Disease Prediction", page_icon="", layout="centered")

# -------------------------------------------------------------------
# Load saved models, scaler and feature list
# -------------------------------------------------------------------
# Directory this app.py file lives in (works no matter what the current
# working directory is when Streamlit Cloud runs the app).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    required_files = ['knn_model.joblib', 'svm_model.joblib',
                       'scaler.joblib', 'feature_columns.joblib']
    missing = [f for f in required_files
               if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        return None, None, None, None, missing

    knn_model = load(os.path.join(BASE_DIR, 'knn_model.joblib'))
    svm_model = load(os.path.join(BASE_DIR, 'svm_model.joblib'))
    scaler = load(os.path.join(BASE_DIR, 'scaler.joblib'))
    feature_columns = load(os.path.join(BASE_DIR, 'feature_columns.joblib'))
    return knn_model, svm_model, scaler, feature_columns, []


knn_model, svm_model, scaler, feature_columns, missing_files = load_artifacts()

# -------------------------------------------------------------------
# History of past predictions (kept in session_state so it survives
# across form submissions within the same browser session)
# -------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Heart Disease Prediction System")
st.write("Predicting heart disease using **K-Nearest Neighbors (KNN)** "
         "and **Support Vector Machine (SVM)**")

if missing_files:
    st.error(
        "Model files not found: " + ", ".join(missing_files) +
        "\n\nPlease run `train_model.py` first (it reads c:\\heart.csv, "
        "trains both models, and saves the .joblib files in this same "
        "folder as app.py)."
    )
    st.stop()

st.markdown("---")
st.subheader("Enter Patient Information")

# -------------------------------------------------------------------
# Patient input form (with input validation)
# -------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)],
                            format_func=lambda x: x[0])
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[(0, "0 - Typical angina"), (1, "1 - Atypical angina"),
                     (2, "2 - Non-anginal pain"), (3, "3 - Asymptomatic")],
            format_func=lambda x: x[1]
        )
        trestbps = st.number_input("Resting Blood Pressure (trestbps, mm Hg)",
                                    min_value=60, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (chol, mg/dl)",
                                min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)",
                            options=[("No", 0), ("Yes", 1)],
                            format_func=lambda x: x[0])
        restecg = st.selectbox(
            "Resting ECG Results (restecg)",
            options=[(0, "0 - Normal"), (1, "1 - ST-T abnormality"),
                     (2, "2 - Left ventricular hypertrophy")],
            format_func=lambda x: x[1]
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved (thalach)",
                                   min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise-Induced Angina (exang)",
                              options=[("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression (oldpeak)",
                                   min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment (slope)",
            options=[(0, "0 - Upsloping"), (1, "1 - Flat"), (2, "2 - Downsloping")],
            format_func=lambda x: x[1]
        )
        ca = st.selectbox("Number of Major Vessels (ca)",
                           options=[0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia (thal)",
            options=[(0, "0 - Unknown"), (1, "1 - Normal"),
                     (2, "2 - Fixed defect"), (3, "3 - Reversible defect")],
            format_func=lambda x: x[1]
        )

    submitted = st.form_submit_button("Predict")

# -------------------------------------------------------------------
# Input validation, scaling and prediction
# -------------------------------------------------------------------
if submitted:
    # Unpack tuple-valued selectbox choices
    sex_val = sex[1]
    cp_val = cp[0]
    fbs_val = fbs[1]
    restecg_val = restecg[0]
    exang_val = exang[1]
    slope_val = slope[0]
    thal_val = thal[0]

    input_values = {
        'age': age, 'sex': sex_val, 'cp': cp_val, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs_val, 'restecg': restecg_val,
        'thalach': thalach, 'exang': exang_val, 'oldpeak': oldpeak,
        'slope': slope_val, 'ca': ca, 'thal': thal_val
    }

    # Basic input validation
    errors = []
    if trestbps <= 0:
        errors.append("Resting blood pressure must be greater than 0.")
    if chol <= 0:
        errors.append("Cholesterol must be greater than 0.")
    if thalach <= 0:
        errors.append("Max heart rate must be greater than 0.")
    if oldpeak < 0:
        errors.append("ST depression (oldpeak) cannot be negative.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        # Build a single-row DataFrame in the correct feature order
        input_df = pd.DataFrame([[input_values[col] for col in feature_columns]],
                                 columns=feature_columns)

        # Apply the saved scaler (fit on training data)
        input_scaled = scaler.transform(input_df)

        # Predict with both models
        knn_pred = knn_model.predict(input_scaled)[0]
        knn_proba = knn_model.predict_proba(input_scaled)[0]

        svm_pred = svm_model.predict(input_scaled)[0]
        svm_proba = svm_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### KNN Prediction")
            label = "Heart Disease" if knn_pred == 1 else "No Heart Disease"
            if knn_pred == 1:
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")
            st.write(f"Probability of No Disease: {knn_proba[0]*100:.2f}%")
            st.write(f"Probability of Disease: {knn_proba[1]*100:.2f}%")

        with col2:
            st.markdown("### SVM Prediction")
            label = "Heart Disease" if svm_pred == 1 else "No Heart Disease"
            if svm_pred == 1:
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")
            st.write(f"Probability of No Disease: {svm_proba[0]*100:.2f}%")
            st.write(f"Probability of Disease: {svm_proba[1]*100:.2f}%")

        st.markdown("---")
        st.caption(
            "Disclaimer: This tool is for academic demonstration purposes only "
            "and is not a substitute for professional medical diagnosis."
        )

        # Save this prediction as a record in the history list
        record = dict(input_values)
        record["KNN Result"] = "Heart Disease" if knn_pred == 1 else "No Heart Disease"
        record["KNN Prob(Disease)"] = f"{knn_proba[1]*100:.2f}%"
        record["SVM Result"] = "Heart Disease" if svm_pred == 1 else "No Heart Disease"
        record["SVM Prob(Disease)"] = f"{svm_proba[1]*100:.2f}%"
        st.session_state.history.append(record)

# -------------------------------------------------------------------
# Display prediction history (one row per past prediction, newest first)
# -------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader(f"Prediction History ({len(st.session_state.history)} record(s))")

    history_df = pd.DataFrame(st.session_state.history[::-1])  # newest first
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.markdown("---")
st.caption("Project: Heart Disease Prediction Using Supervised Machine Learning "
           "(KNN & SVM) | BMCS2003")
