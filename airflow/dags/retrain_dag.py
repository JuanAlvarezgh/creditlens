from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_training():
    from ml.train import train_and_register

    return train_and_register()


def _check_drift(**context):
    import mlflow
    from loguru import logger

    from config import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    production = client.get_latest_versions(settings.mlflow_model_name, stages=["Production"])
    if not production:
        logger.warning("No Production model found for drift check")
        return

    current_auc = client.get_run(production[0].run_id).data.metrics.get("auc_roc", 1.0)
    archived = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
    archived = [v for v in archived if v.current_stage == "Archived"]

    if not archived:
        logger.info("No archived models — skipping drift comparison")
        return

    prev_auc = client.get_run(archived[-1].run_id).data.metrics.get("auc_roc", 1.0)
    drift = prev_auc - current_auc
    if drift > 0.02:
        logger.warning(
            f"DRIFT ALERT: AUC dropped {drift:.4f} "
            f"(prev={prev_auc:.4f}, current={current_auc:.4f})"
        )
    else:
        logger.info(f"No significant drift detected: delta={drift:.4f}")


with DAG(
    dag_id="creditlens_retrain",
    description="Weekly model retraining and drift monitoring",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
) as dag:
    run_training = PythonOperator(task_id="run_training", python_callable=_run_training)
    check_drift = PythonOperator(
        task_id="check_drift", python_callable=_check_drift, provide_context=True
    )
    run_training >> check_drift
