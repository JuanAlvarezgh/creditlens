import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import settings
from ml.evaluate import compute_financial_metrics

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
LABEL_COL = "label"

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "xgboost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    ),
    "xgboost_deep": XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    ),
    "lightgbm": LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
    ),
    "lightgbm_tuned": LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    ),
}


def load_features() -> pd.DataFrame:
    import psycopg2

    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )
    try:
        df = pd.read_sql_query("SELECT * FROM mart_credit_features", conn)
    finally:
        conn.close()
    logger.info(f"Loaded {len(df)} rows from mart_credit_features")
    return df


def train_and_register(experiment_name: str = "credit_risk") -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    df = load_features()
    X = df[FEATURE_COLS].fillna(0)
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_auc = 0.0
    best_run_id = None

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            metrics = compute_financial_metrics(y_test.values, y_pred)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=settings.mlflow_model_name,
            )
            logger.info(f"{name}: AUC={metrics['auc_roc']:.4f}")

            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_run_id = run.info.run_id

    _promote_best_model(best_run_id, best_auc)
    return best_run_id


def _promote_best_model(run_id: str, auc: float) -> None:
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
    best_version = next(v for v in versions if v.run_id == run_id)
    client.transition_model_version_stage(
        name=settings.mlflow_model_name,
        version=best_version.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(f"Promoted version {best_version.version} to Production (AUC={auc:.4f})")


if __name__ == "__main__":
    train_and_register()
