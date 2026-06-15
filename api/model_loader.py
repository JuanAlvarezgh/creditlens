from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
import shap
from loguru import logger

from config import settings


def _build_explainer(model):
    """Pick the right SHAP explainer based on model type."""
    model_class = type(model).__name__
    if model_class in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)
    if model_class == "LogisticRegression":
        background = np.zeros((1, model.coef_.shape[1]))
        return shap.LinearExplainer(model, background)
    return shap.Explainer(model)


class ModelLoader:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.version_info: dict = {}

    def load(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient()

        versions = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
        if not versions:
            raise RuntimeError(
                f"No Production model for '{settings.mlflow_model_name}'. "
                "Run `python -m ml.train` first."
            )

        version = versions[0]
        model_uri = f"models:/{settings.mlflow_model_name}/Production"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.explainer = _build_explainer(self.model)

        run = client.get_run(version.run_id)
        registered_iso = datetime.fromtimestamp(version.creation_timestamp / 1000, tz=UTC).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        self.version_info = {
            "model_name": settings.mlflow_model_name,
            "version": version.version,
            "stage": version.current_stage,
            "algorithm": type(self.model).__name__,
            "auc_roc": run.data.metrics.get("auc_roc"),
            "ks_statistic": run.data.metrics.get("ks_statistic"),
            "gini": run.data.metrics.get("gini"),
            "registered_at": registered_iso,
        }
        logger.info(
            f"Loaded '{settings.mlflow_model_name}' v{version.version} "
            f"({type(self.model).__name__}) from Production"
        )


model_loader = ModelLoader()
