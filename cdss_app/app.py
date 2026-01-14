
import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
sys.path.append(".")

from shap_utils import explain_patient_shap  # or shap.py if named that
from cdss import cdss_decision, cdss_batch_prediction

# -----------------------
# Load model & metadata
# -----------------------
model = jb.load("model/cdss_model.pkl")
feature_origin_map = jb.load("model/feature_origin_map.pkl")
X_background = jb.load("model/X_background.pkl")

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="CDSS for CAD", layout="centered")

st.title("Coronary Artery Disease CDSS")
st.markdown("Clinical Decision Support System for CAD Risk Assessment")

# set sidebar options
st.sidebar.header("Options")


mode = st.radio(
    "Select Prediction Mode",
    ["Single Patient", "Batch Prediction"]
)

show_shap = st.sidebar.checkbox(
    "Show Advanced SHAP Explanations (For Research  purposes Only)",
    value=False
)
# UI by prediction mode choice


# Single patient assesment mode

if mode == "Single Patient":
    st.subheader("Patient Information")

    with st.form("patient_form"):

        col1, col2,col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 18, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            systolic = st.number_input("Systolic BP", 80, 220, 120)
            cholesterol = st.number_input("Total Cholesterol", 100.0, 400.0, 200.0)
           
            


        with col2:
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            diabetes = st.selectbox("Diabetes", ["Yes", "No"])
            hypertension = st.selectbox("Hypertension", ["Yes", "No"])
            chest_pain = st.selectbox("Chest Pain Type", ["Typical", "Atypical"])
           
            
            
        with col3:
            family_history = st.selectbox("Family History of CAD", ["Yes", "No"])
            medicare = st.selectbox("Medication Adherence Level",["High","Low","Moderate"])
            stress = st.selectbox("Stress Level",["Low","Moderate","High"])
            healthcare = st.selectbox("Healthcare Access",["Good","Fair","Poor"])
            alcohol = st.selectbox("Alcohol Use",["Low","Moderate","High"])
            


        submitted = st.form_submit_button("Assess CAD Risk")

    # -----------------------
    # Prediction
    # -----------------------
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
            return (
            name.replace("numeric__", "")
                .replace("nominal__", "")
                .replace("_", " ")
        )


        result = cdss_decision(
            patient_df=patient,
            model=model,
            feature_origin_map=feature_origin_map
        )

        st.subheader("CDSS Result")

        st.metric(
            label="CAD Risk Probability",
            value=result["CAD_Risk_Probability"]
        )

        st.write("**Risk Category:**", result["Risk_Category"])
        st.write("**CAD Phenotype:**", result["CAD_Phenotype"])

        st.subheader("Key Contributing Factors")
        
        for feat, contrib in result["Top_Contributing_Features"]:
            st.write(f"- **{clean_feature_name(feat)}** => {contrib:+.3f}")

  # OPTIONAL SHAP
    if show_shap:
        shap_values, explainer, fig = explain_patient_shap(
            model,
            X_background,   # background dataset
            X_patient=patient,
            plot=True
        )
        st.subheader("SHAP Explanation")
        if fig:
            st.pyplot(fig)
      

  # handling batch prediction
  
elif mode == "Batch Prediction":
    st.subheader("Batch CAD Risk Assessment")

    uploaded_file = st.file_uploader(
        "Upload patient data (CSV)",
        type=["csv"]
    )

    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(batch_data.head())

        if st.button("Run Batch Prediction"):
            batch_results = cdss_batch_prediction(
                batch_data,
                model,
                feature_origin_map
            )

            st.success("Batch prediction completed")
            st.dataframe(batch_results)

            st.download_button(
                label="Download Results",
                data=batch_results.to_csv(index=False),
                file_name="cad_batch_predictions.csv",
                mime="text/csv"
            )

      

    st.info(
        "This System is designed to assist clinicians in making decision!.\n It should not replace the clinicians"
    )
