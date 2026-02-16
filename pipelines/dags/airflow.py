"""
Clinical Trials Pipeline DAG
Orchestrates the clinical trial data fetch, processing, and upload workflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.clinical_trials_dag import fetch_trials, save_to_csv, upload_to_gcs, cleanup

default_args = {
    "owner": "triallink",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="clinical_trials_pipeline",
    default_args=default_args,
    description="Fetch diabetes clinical trials and upload to GCS",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["clinical-trials", "diabetes", "data-pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_trials",
        python_callable=fetch_trials,
    )

    t2 = PythonOperator(
        task_id="save_to_csv",
        python_callable=save_to_csv,
    )

    t3 = PythonOperator(
        task_id="upload_to_gcs",
        python_callable=upload_to_gcs,
    )

    t4 = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup,
    )

    t1 >> t2 >> t3 >> t4