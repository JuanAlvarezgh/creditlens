def test_ingestion_dag_loads_without_error():
    from airflow.dags.ingestion_dag import dag

    assert dag.dag_id == "creditlens_ingestion"


def test_dbt_dag_loads_without_error():
    from airflow.dags.dbt_dag import dag

    assert dag.dag_id == "creditlens_dbt"


def test_retrain_dag_loads_without_error():
    from airflow.dags.retrain_dag import dag

    assert dag.dag_id == "creditlens_retrain"


def test_retrain_dag_check_drift_is_downstream_of_run_training():
    from airflow.dags.retrain_dag import dag

    run_training = dag.get_task("run_training")
    downstream_ids = [t.task_id for t in run_training.downstream_list]
    assert "check_drift" in downstream_ids
