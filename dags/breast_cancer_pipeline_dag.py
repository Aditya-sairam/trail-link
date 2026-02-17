from __future__ import annotations

from datetime import datetime
import pandas as pd
import json
import os
import subprocess

from src.pipelines.breast_cancer.quality import anomalies_found

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import ShortCircuitOperator

from airflow.operators.email import EmailOperator
from src.pipelines.breast_cancer.ingest import download_raw_trials_csv, enrich_trials_csv
from src.pipelines.breast_cancer.quality import run_quality_checks
from src.pipelines.breast_cancer.bias import generate_bias_report, save_bias_report


RAW_PATH = "data/breast_cancer/raw/breast_cancer_trials_raw.csv"
ENRICHED_PATH = "data/breast_cancer/processed/breast_cancer_trials_enriched.csv"
STATS_PATH = "data/breast_cancer/reports/stats.json"
ANOMALIES_PATH = "data/breast_cancer/reports/anomalies.json"
BIAS_PATH = "data/breast_cancer/reports/bias_report.json"


def task_fetch_raw() -> None:
    download_raw_trials_csv(
        raw_file_path=RAW_PATH,
        status="RECRUITING",
        page_size=1000,
        condition_query="breast cancer",
    )


def task_enrich() -> None:
    enrich_trials_csv(
        raw_file_path=RAW_PATH,
        enriched_file_path=ENRICHED_PATH,
    )


def task_quality() -> None:
    run_quality_checks(
        enriched_csv_path=ENRICHED_PATH,
        stats_path=STATS_PATH,
        anomalies_path=ANOMALIES_PATH,
    )


def task_bias() -> None:
    df = pd.read_csv(ENRICHED_PATH)
    report = generate_bias_report(df)
    save_bias_report(report, BIAS_PATH)

def task_check_anomalies_and_decide() -> bool:
    """
    Returns True if anomalies exist, else False.
    Airflow will push this return value to XCom.
    """
    with open(ANOMALIES_PATH, "r", encoding="utf-8") as f:
        anomalies = json.load(f)
    return anomalies_found(anomalies)

def task_dvc_version_data() -> None:
    """
    Version pipeline outputs using DVC inside the Airflow container.
    This runs:
      - dvc add (updates .dvc files if outputs changed)
      - dvc push (pushes artifacts to configured DVC remote)
    """
    repo_root = "/opt/airflow/repo"

    cmds = [
        ["dvc", "add", "data/breast_cancer/raw", "data/breast_cancer/processed", "data/breast_cancer/reports"],
        ["dvc", "push"],
    ]

    for cmd in cmds:
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

with DAG(
    dag_id="breast_cancer_data_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "breast_cancer"],
) as dag:

    fetch_raw = PythonOperator(
        task_id="fetch_raw",
        python_callable=task_fetch_raw,
    )

    enrich = PythonOperator(
        task_id="enrich",
        python_callable=task_enrich,
    )

    quality = PythonOperator(
        task_id="quality_checks",
        python_callable=task_quality,
    )

    bias = PythonOperator(
        task_id="bias_slicing",
        python_callable=task_bias,
    )

    check_anomalies = ShortCircuitOperator(
    task_id="check_anomalies",
    python_callable=task_check_anomalies_and_decide,
    )


    alert_email = EmailOperator(
    task_id="alert_on_anomalies",
    to="admin@example.com",
    subject="Breast Cancer Pipeline: Data Anomalies Detected",
    html_content="""
    <h3>Data anomalies were detected in the latest breast cancer pipeline run.</h3>
    <p>Please check the anomalies report at: <code>data/breast_cancer/reports/anomalies.json</code></p>
    """,
   )
    
    dvc_version = PythonOperator(
    task_id="dvc_version_data",
    python_callable=task_dvc_version_data,
)


    fetch_raw >> enrich >> quality >> check_anomalies
    quality >> bias >> dvc_version
    check_anomalies >> alert_email



