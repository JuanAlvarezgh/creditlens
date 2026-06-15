import numpy as np
import pandas as pd
import shap

FEATURE_COLS = [
    "revolving_utilization",
    "age",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "times_90_days_late",
    "real_estate_loans",
    "times_60_89_days_late",
    "dependents",
    "total_late_payments",
    "dti",
]


def _classify_risk(probability: float) -> str:
    if probability < 0.3:
        return "Bajo"
    elif probability < 0.6:
        return "Medio"
    return "Alto"


def _add_derived_features(features: dict) -> dict:
    features["total_late_payments"] = (
        features.get("times_30_59_days_late", 0)
        + features.get("times_60_89_days_late", 0)
        + features.get("times_90_days_late", 0)
    )
    features["dti"] = features.get("debt_ratio", 0.0)
    return features


def score(features: dict, model, explainer: shap.TreeExplainer) -> dict:
    """
    Score a single credit application.

    Returns dict with probability_of_default, risk_level, shap_explanation.
    """
    enriched = _add_derived_features(dict(features))
    X = pd.DataFrame([enriched])[FEATURE_COLS].fillna(0)

    probability = float(model.predict_proba(X)[0, 1])
    risk_level = _classify_risk(probability)

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_row = shap_values[0]
    top3_idx = np.argsort(np.abs(shap_row))[::-1][:3]
    explanation = [{"feature": FEATURE_COLS[i], "impact": float(shap_row[i])} for i in top3_idx]

    return {
        "probability_of_default": probability,
        "risk_level": risk_level,
        "shap_explanation": explanation,
    }
