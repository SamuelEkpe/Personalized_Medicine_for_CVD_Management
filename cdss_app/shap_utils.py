# shap_utils.py
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import re

def simplify_feature_name(name):
    """
    Removes transformer prefixes and one-hot category suffixes
    """
    name = re.sub(r"^(numeric|nominal|ordinal)__", "", name)
    parts = name.split("_")
    categories = {
        "yes", "no", "low", "moderate", "high",
        "current", "former", "never",
        "male", "female", "normal", "abnormal",
        "poor","average","good","primary","tertiary","secondary",
        "st_depression"
    }
    if parts[-1].lower() in categories:
        return "_".join(parts[:-1])
    return name


def explain_patient_shap(model, X_background, X_patient, plot=True):
    """
    Computes SHAP values for a single patient, handles missing features,
    aggregates one-hot encoded features, and optionally plots results.

    Parameters:
    - model: sklearn Pipeline with preprocessor and classifier
    - X_background: pd.DataFrame used as background dataset for SHAP
    - X_patient: pd.DataFrame, Series, or dict for patient features
    - plot: bool, whether to plot SHAP values

    Returns:
    - shap_values_agg: pd.Series of aggregated SHAP values for patient
    - explainer: SHAP explainer object
    """

    # Extract classifier and preprocessor
    classifier = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessing"]

    # Convert patient input to DataFrame
    if isinstance(X_patient, dict) or isinstance(X_patient, pd.Series):
        X_patient = pd.DataFrame([X_patient])

    # Ensure all columns are present
    expected_cols = preprocessor.feature_names_in_

    # Fill missing columns in background
    for col in expected_cols:
        if col not in X_background.columns:
            X_background[col] = np.nan
    X_background = X_background[expected_cols]

    # Fill missing columns in patient
    for col in expected_cols:
        if col not in X_patient.columns:
            X_patient[col] = np.nan
    X_patient = X_patient[expected_cols]

    # Transform data
    X_bg_transformed = preprocessor.transform(X_background)
    X_patient_transformed = preprocessor.transform(X_patient)

    # SHAP explainer
    explainer = shap.LinearExplainer(
        classifier,
        X_bg_transformed,
        feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(X_patient_transformed)

    # Convert to DataFrame
    feature_names = preprocessor.get_feature_names_out()
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Simplify feature names
    shap_df.columns = [simplify_feature_name(c) for c in shap_df.columns]

    # Aggregate one-hot encoded features
    shap_aggregated = shap_df.groupby(axis=1, level=0).sum()  # sum for single patient
    shap_values_agg = shap_aggregated.iloc[0].sort_values(ascending=False)

    # Optional plotting
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap_values_agg.sort_values().plot(kind="barh", color="skyblue", ax=ax)
        ax.set_xlabel("SHAP value")
        ax.set_title("Patient Feature Impact (SHAP)")
        plt.tight_layout()
        return shap_values_agg, explainer, fig

    return shap_values_agg, explainer, None
