"""
Heart Disease Prediction - Streamlit App
=========================================
Loads the trained KNN and SVM models + scaler (produced by train_model.py)
and lets the user enter patient information to get predictions from both
models, following the system flowchart in Section 3.1 of the report:
    Enter patient data -> validate input -> apply saved scaler ->
    predict with both models -> display predicted class + probability

This version also keeps a running history of every prediction made during
the session, shown as a table below the form.
"""

import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os
from datetime import datetime

st.set_page_config(page_title="Heart Disease Prediction", page_icon="", layout="centered")

# -------------------------------------------------------------------
# Load saved models, scaler and feature list
# -------------------------------------------------------------------
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
# Session state: keep a list of past predictions for this session
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
st.caption("Most of these numbers come from a doctor's checkup, blood test, "
           "or ECG (heart electrical test) report. If you don't know a "
           "value, you can leave it at the default shown.")

# -------------------------------------------------------------------
# Patient input form (with input validation)
# -------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120,
                               value=50, step=1,
                               help="The patient's age in years.")
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)],
                            format_func=lambda x: x[0])
        cp = st.selectbox(
            "Chest Pain When It Happens (cp)",
            options=[(0, "0 - Typical chest pain (classic tight/pressure feeling)"),
                     (1, "1 - Unusual chest pain (doesn't feel like normal chest pain)"),
                     (2, "2 - Pain not related to the heart"),
                     (3, "3 - No chest pain symptoms")],
            format_func=lambda x: x[1],
            help="Describes the type of chest pain the patient feels, if any. "
                 "This is usually noted by a doctor during a checkup."
        )
        trestbps = st.number_input("Resting Blood Pressure (mmHg)",
                                    min_value=60, max_value=250, value=120,
                                    help="Blood pressure measured while the "
                                         "patient is calm and sitting still, "
                                         "not exercising. Found on a blood "
                                         "pressure reading, e.g. '120/80'.")
        chol = st.number_input("Cholesterol Level (mg/dl)",
                                min_value=100, max_value=600, value=200,
                                help="Total cholesterol level from a blood test.")
        fbs = st.selectbox("Fasting Blood Sugar Over 120 mg/dl?",
                            options=[("No", 0), ("Yes", 1)],
                            format_func=lambda x: x[0],
                            help="Was the patient's blood sugar high "
                                 "(above 120 mg/dl) after not eating for "
                                 "several hours? From a blood test.")
        restecg = st.selectbox(
            "Resting ECG Result",
            options=[(0, "0 - Normal"),
                     (1, "1 - Minor abnormality in heart's electrical signal"),
                     (2, "2 - Signs of thickened heart muscle")],
            format_func=lambda x: x[1],
            help="Result of a resting ECG (electrocardiogram), a machine "
                 "test that records the heart's electrical activity. Ask "
                 "your doctor or check the ECG report."
        )

    with col2:
        thalach = st.number_input("Highest Heart Rate Reached During Exercise (bpm)",
                                   min_value=60, max_value=250, value=150,
                                   help="The highest heart rate (beats per "
                                        "minute) the patient reached during "
                                        "an exercise/stress test.")
        exang = st.selectbox("Chest Pain Triggered by Exercise?",
                              options=[("No", 0), ("Yes", 1)],
                              format_func=lambda x: x[0],
                              help="Did exercise cause chest pain (angina)?")
        oldpeak = st.number_input("ST Depression Value (oldpeak)",
                                   min_value=0.0, max_value=10.0, value=1.0,
                                   step=0.1,
                                   help="A number from an exercise ECG test "
                                        "showing how much a certain part of "
                                        "the heart signal (the 'ST segment') "
                                        "dips during exercise. Higher usually "
                                        "means more strain on the heart. "
                                        "Found on the stress-test report.")
        slope = st.selectbox(
            "Shape of the Heart Signal During Peak Exercise",
            options=[(0, "0 - Rising (upsloping)"),
                     (1, "1 - Flat"),
                     (2, "2 - Falling (downsloping)")],
            format_func=lambda x: x[1],
            help="Describes the shape of the ST segment (part of the ECG "
                 "line) at the peak of exercise. From the exercise ECG report."
        )
        ca = st.selectbox("Number of Major Blood Vessels Showing Blockage (0-4)",
                           options=[0, 1, 2, 3, 4],
                           help="Number of major blood vessels near the "
                                "heart that show up as blocked/narrowed on "
                                "a special X-ray (fluoroscopy) test.")
        thal = st.selectbox(
            "Thalassemia Blood Test Result",
            options=[(0, "0 - Not tested / unknown"),
                     (1, "1 - Normal"),
                     (2, "2 - Fixed defect (permanent issue)"),
                     (3, "3 - Reversible defect (temporary issue)")],
            format_func=lambda x: x[1],
            help="Result of a thalassemia blood test, which checks for a "
                 "certain type of blood disorder that can affect heart "
                 "test readings."
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

        # ---------------------------------------------------------------
        # Save this prediction into the session history
        # ---------------------------------------------------------------
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": age,
            "Sex": sex[0],
            **{col: input_values[col] for col in
               ['trestbps', 'chol', 'thalach', 'oldpeak']},
            "KNN Result": "Disease" if knn_pred == 1 else "No Disease",
            "KNN Prob. of Disease (%)": round(knn_proba[1] * 100, 2),
            "SVM Result": "Disease" if svm_pred == 1 else "No Disease",
            "SVM Prob. of Disease (%)": round(svm_proba[1] * 100, 2),
        }
        st.session_state.history.append(record)

# -------------------------------------------------------------------
# Prediction history (accumulates across every Predict click)
# -------------------------------------------------------------------
st.markdown("---")
st.subheader("Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History as CSV", data=csv_data,
                            file_name="prediction_history.csv",
                            mime="text/csv")
    with col_b:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
else:
    st.caption("No predictions yet. Fill in the form above and click "
               "**Predict** — each result will be added to a record here.")

st.markdown("---")
st.caption("Project: Heart Disease Prediction Using Supervised Machine Learning "
           "(KNN & SVM) | BMCS2003")
