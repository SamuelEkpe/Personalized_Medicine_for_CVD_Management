
import numpy as np
import pandas as pd

def complete_patient_data(patient, feature_template):
    patient = patient.copy()
   
    for col in feature_template:
        if col not in patient.columns:
            patient[col] = np.nan
    return patient[feature_template]



def assign_phenotype(row, cad_status):
    if cad_status == "NO":
        return "No CAD"

    if (
         (row["Chest_Pain_Type"] == "Typical")
          or (row['Palpitations']=='Yes')
         or (row['Dizziness']== 'Yes')
     ):
     return "Symptomatic CAD"

    if row["Family_History_CAD"] == "Yes":
        return "Genetically predisposed CAD"

   
    return "Non-genetic, low-symptom CAD"


def explain_prediction(patient_df, model, feature_origin_map,top_k=5):

    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["model"]

    X = preprocessor.transform(patient_df)
    coefs = classifier.coef_[0]

    feature_names = preprocessor.get_feature_names_out()
    # Handle sparse OR dense
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    contributions = X_dense[0] * coefs

    explanation = []

    for fname, contrib in zip(feature_names, contributions):

        base_feature = feature_origin_map.get(fname)

        # Exclude if base feature missing or imputed
        if base_feature not in patient_df.columns:
            continue

        if pd.isna(patient_df[base_feature].iloc[0]):
            continue

        explanation.append((fname, contrib))

    explanation = sorted(
        explanation,
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_k]

    return explanation



def cdss_decision(patient_df, model, feature_origin_map,threshold=0.5):
    
   #  Run CAD CDSS decision for a single patient
    

    # Ensure required features exist
    feature_template = model.feature_names_in_


    patient_df = complete_patient_data(
        patient_df,
        feature_template
        
    )
    
    # Access pipeline components
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["model"]

    # Transform patient data
    X_patient = preprocessor.transform(patient_df)

    # Predict CAD risk
    risk_prob = classifier.predict_proba(X_patient)[0, 1]
    cad_status = "YES" if risk_prob >= threshold else "NO"
    risk_label = "High CAD Risk" if cad_status == "YES" else "Low CAD Risk"


    # Assign CAD phenotype (rule-based)
    phenotype = assign_phenotype(patient_df.iloc[0], cad_status)

    # Explain prediction 
    explanation = explain_prediction(
        patient_df=patient_df,
        model=model,
        feature_origin_map=feature_origin_map,
        top_k= 5
      )

    return {
        "CAD_Risk_Probability": round(risk_prob, 3),
        "Risk_Category": risk_label,
        "CAD_Phenotype": phenotype,
        "Top_Contributing_Features": explanation
    }

def cdss_batch_prediction(
    patients_df,
    model,
    threshold=0.5
):
    """
    Perform batch CAD risk prediction for multiple patients.
    Returns a DataFrame with risk probabilities and phenotypes.
    """

    results = []

    for idx in range(len(patients_df)):
        patient = patients_df.iloc[[idx]]  # keep as DataFrame

        output = cdss_decision(
            patient_df=patient,
            model=model,
            threshold=threshold
        )

        results.append({
            "Patient_ID": idx + 1,
            "CAD_Risk_Probability": output["CAD_Risk_Probability"],
            "Predicted_CAD_Status": output["Predicted_CAD_Status"],
            "Risk_Category": output["Risk_Category"],
            "CAD_Phenotype": output["CAD_Phenotype"]
        })

    return pd.DataFrame(results)

