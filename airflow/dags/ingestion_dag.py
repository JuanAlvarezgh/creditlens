from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _run_producer():
    from ingestion.producer import produce_from_csv

    produce_from_csv(delay=0)


def _run_consumer():
    from ingestion.consumer import consume

    consume(max_messages=5000)


with DAG(
    dag_id="creditlens_ingestion",
    description="Daily ingestion of credit applications via Kafka",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
) as dag:
    produce = PythonOperator(task_id="produce_to_kafka", python_callable=_run_producer)
    consume_task = PythonOperator(task_id="consume_from_kafka", python_callable=_run_consumer)
    produce >> consume_task
