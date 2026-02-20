
import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
import os
sys.path.append(".")

from shap_utils import explain_patient_shap  # or shap.py if named that
from cdss import cdss_decision, cdss_batch_prediction

# -----------------------
# Load model & metadata
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "cdss_model.pkl") # to fetch the model path

model = jb.load(MODEL_PATH)

#model = jb.load("model/cdss_model.pkl")
feature_path = os.path.join(BASE_DIR, "model","feature_origin_map.pkl") 
feature_origin_map = jb.load(feature_path)
X_background_path = os.path.join(BASE_DIR, "model","X_background.pkl")
X_background = jb.load(X_background_path)

# ----------------------- HOSPITAL THEME STYLE ----------
st.set_page_config(
    page_title="CAD Clinical Decision Support System",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"]{
    background-color:#f6fbf9;  /* Soft hospital background */
}

/* Sidebar */
[data-testid="stSidebar"]{
    background-color:#e8f5f1;  /* Light greenish sidebar */
}

/* Headings */
h1, h2, h3 {
    color:#0f4c81;  /* Professional medical blue */
    font-weight:600;
}

/* Prediction result cards */
.result-box {
    padding:22px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
    text-align:center;
    margin-top:18px;
}

/* Low risk (hospital green) */
.low-risk {
    background:#e6f4ea;
    color:#1b7f3a;
    border:2px solid #b7e1c1;
}

/* High risk (medical alert red but soft) */
.high-risk {
    background:#fdeaea;
    color:#a61b1b;
    border:2px solid #f5b5b5;
}

/* Buttons */
.stButton>button {
    background-color:#2a7fba;
    color:white;
    border-radius:10px;
    height:3em;
    font-size:16px;
    border:none;
}

.stButton>button:hover {
    background-color:#1f6696;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER ----------
st.markdown("""
<h1 style='text-align: center;'>Coronary Artery Disease CDSS</h1>
<p style='text-align: center; font-size:16px; color:gray;'>
AI-Powered Clinical Risk Assessment (Logistic Regression + SHAP)
</p>
""", unsafe_allow_html=True)

st.divider()

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.header("System Options")

mode = st.sidebar.radio(
    "Select Prediction Mode",
    ["Single Patient", "Batch Prediction"]
)

show_shap = st.sidebar.checkbox(
    "Show Advanced SHAP Explanations (For Research  purposes Only)",
    value=False
)

# ---------- SINGLE PATIENT ----------
if mode == "Single Patient":
    st.subheader("Patient Assessment Form")

    with st.form("patient_form"):

        # -----------------------
        # DEMOGRAPHICS
        # -----------------------
        st.markdown("Demographic Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (Years)", 18, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 25.0)
            family_history = st.selectbox("Family History of CAD", ["Yes", "No"])
        with col3:
            systolic = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
            cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100.0, 400.0, 200.0)

        st.divider()

        # -----------------------
        # CLINICAL FACTORS
        # -----------------------
        st.markdown("Clinical Risk Factors")
        col4, col5, col6 = st.columns(3)
        with col4:
            diabetes = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
            hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        with col5:
            chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical"])
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        with col6:
            stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
            alcohol = st.selectbox("Alcohol Use", ["Low", "Moderate", "High"])

        st.divider()

        # -----------------------
        # SOCIAL / BEHAVIORAL
        # -----------------------
        st.markdown("Medication Adherance and Healthcare Access")
        col7, col8 = st.columns(2)
        with col7:
            medicare = st.selectbox("Medication Adherence", ["High", "Moderate", "Low"])
        with col8:
            healthcare = st.selectbox("Healthcare Access", ["Good", "Fair", "Poor"])

        submitted = st.form_submit_button("Assess CAD Risk", use_container_width=True)

    if submitted:

        patient = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "BMI": bmi,
            "Systolic_BP": systolic,
            "Total_Cholesterol": cholesterol,
            "Smoking_Status": smoking,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "Chest_Pain_Type": chest_pain,
            "Family_History_CAD": family_history,
            "Medication_Adherence": medicare,
            "Stress_Level": stress,
            "Alcohol_Use": alcohol,
            "Healthcare_Access": healthcare
        }])

        def clean_feature_name(name):
            return name.replace("numeric__", "").replace("nominal__", "").replace("_", " ")

        result = cdss_decision(patient_df=patient, model=model, feature_origin_map=feature_origin_map)

        st.divider()
        st.subheader("Risk Assessment Result")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("CAD Risk Probability", result["CAD_Risk_Probability"])
        with colB:
            st.metric("Risk Category", result["Risk_Category"])
        with colC:
            st.metric("CAD Phenotype", result["CAD_Phenotype"])

        st.markdown("## Key Contributing Factors")
        for feat, contrib in result["Top_Contributing_Features"]:
            st.write(f"- **{clean_feature_name(feat)}** => {contrib:+.3f}")

        if show_shap:
            shap_values, explainer, fig = explain_patient_shap(
                model, X_background, X_patient=patient, plot=True
            )
            st.subheader("SHAP Explanation")
            if fig:
                st.pyplot(fig)

# ---------- BATCH PREDICTION ----------
elif mode == "Batch Prediction":
    st.subheader("Batch CAD Risk Assessment")

    uploaded_file = st.file_uploader("Upload patient dataset (CSV) format", type=["csv"])
    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        st.markdown("Uploaded Data Preview:")
        st.dataframe(batch_data.head())

        if st.button("Run Batch Prediction", use_container_width=True):
            batch_results = cdss_batch_prediction(batch_data, model, feature_origin_map)
            st.success("Batch prediction completed")
            st.dataframe(batch_results)
            st.download_button(
                label="Download Results",
                data=batch_results.to_csv(index=False),
                file_name="cad_batch_predictions.csv",
                mime="text/csv"
            )

    st.info("This system is designed to assist clinicians in decision-making.\n It should not replace clinicians.")
    st.info("Developed by Ekpe Samuel Cyril \n Submitted to Department of Computer Science \n In Partial Fulfilment of the Requirements for the Award of Masters Degree in Computer Science"
