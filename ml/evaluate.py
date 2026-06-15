import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import roc_auc_score


def compute_ks_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Max separation between default and non-default score CDFs."""
    df = pd.DataFrame({"label": y_true, "score": y_pred_proba})
    defaults = df[df["label"] == 1]["score"].values
    non_defaults = df[df["label"] == 0]["score"].values
    thresholds = np.linspace(0, 1, 200)
    ks = max(abs(np.mean(defaults <= t) - np.mean(non_defaults <= t)) for t in thresholds)
    return float(ks)


def compute_gini(auc_roc: float) -> float:
    return 2 * auc_roc - 1


def compute_financial_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    auc = float(roc_auc_score(y_true, y_pred_proba))
    ks = compute_ks_statistic(y_true, y_pred_proba)
    gini = compute_gini(auc)
    logger.info(f"AUC-ROC={auc:.4f} | KS={ks:.4f} | Gini={gini:.4f}")
    return {"auc_roc": auc, "ks_statistic": ks, "gini": gini}


def get_shap_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Returns SHAP values for the positive class."""
    values = explainer.shap_values(X)
    if isinstance(values, list):
        return values[1]
    return values


def get_top_shap_features(
    shap_values: np.ndarray, feature_names: list[str], n: int = 3
) -> list[dict]:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    return [{"feature": feature_names[i], "impact": float(mean_abs[i])} for i in top_idx]
