
import streamlit as st
import joblib as jb
import pandas as pd
import sys
import os

sys.path.append(".")

from cdss import cdss_decision

# -----------------------
# Load model & metadata
# -----------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "cdss_model.pkl")

model = jb.load(MODEL_PATH)

#model = jb.load("model/cdss_model.pkl")
feature_path = os.path.join(BASE_DIR, "model","feature_origin_map.pkl")
feature_origin_map = jb.load(feature_path)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="CDSS for CAD", layout="centered")

st.title("Coronary Artery Disease CDSS")
st.markdown("Clinical Decision Support System for CAD Risk Assessment")

st.subheader("Patient Information")

with st.form("patient_form"):

    col1, col2 = st.columns(2)

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
        family_history = st.selectbox("Family History of CAD", ["Yes", "No"])

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
        "Family_History_CAD": family_history
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


    #for feat, contrib in result["Top_Contributing_Features"]:
      #  st.write(f"- **{feat}** → {contrib:+.3f}")
      

    st.info(
        "This System is designed to assist clinicians in making decision!.\n It should not replace the clinicians"
    )
