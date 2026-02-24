# pipelines/dags/airflow.py

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.task_group import TaskGroup

from src.bias import generate_bias_report, save_bias_report
from src.conditions.registry import REGISTRY
from src.ingest import download_raw_trials_csv, enrich_trials_csv
from src.quality import anomalies_found, run_quality_checks
from src.schema import run_schema_checkpoint, RAW_REQUIRED_DEFAULT, PROCESSED_REQUIRED_DEFAULT
from src.stats import compute_stats
from src.validate import run_validation
from src.gcs_upload import upload_raw_to_gcs, upload_reports_to_gcs
from src.firestore_upload import upload_enriched_to_firestore


BASE = os.getenv("TRAILLINK_BASE", "/opt/airflow/repo")
BUCKET_NAME  = os.getenv("BUCKET_NAME")
PROJECT_ID   = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

CONDITIONS = ["diabetes", "breast_cancer"]
log = logging.getLogger(__name__)


def get_config_for_condition(condition: str) -> dict:
    if condition not in REGISTRY:
        raise ValueError(f"Unknown condition: '{condition}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[condition]


def abs_path(path: str) -> str:
    return path if path.startswith("/") else os.path.join(BASE, path)


def safe_path(p: str) -> str | None:
    return p if p and os.path.exists(p) else None


def get_xcom(ti, task_id: str, key: str, default=None):
    val = ti.xcom_pull(task_ids=task_id, key=key)
    return default if val is None else val


def check_gcp_config() -> bool:
    """
    Check if GCP configuration is available for upload tasks.
    Returns True if all required env vars are set and credentials exist.
    """
    has_bucket = BUCKET_NAME is not None
    has_project = PROJECT_ID is not None
    has_firestore = FIRESTORE_DB is not None
    has_creds = GOOGLE_CREDS is not None and os.path.exists(GOOGLE_CREDS)
    
    gcp_configured = has_bucket and has_project and has_firestore and has_creds
    
    if not gcp_configured:
        log.warning("⚠️ GCP configuration incomplete - upload tasks will be skipped")
        log.warning(f"  RAW_CLINICAL_TRIALS_STORAGE: {'✓' if has_bucket else '✗ Not set'}")
        log.warning(f"  GCP_PROJECT_ID: {'✓' if has_project else '✗ Not set'}")
        log.warning(f"  CLINICAL_TRIALS_FIRESTORE: {'✓' if has_firestore else '✗ Not set'}")
        log.warning(f"  GOOGLE_APPLICATION_CREDENTIALS: {'✓' if has_creds else '✗ Not found'}")
        log.info("💡 Pipeline will run locally. Use 'docker cp' to retrieve output files.")
    else:
        log.info("✅ GCP configuration complete - uploads to GCS and Firestore enabled")
    
    return gcp_configured


def task_check_gcp_config(**context) -> bool:
    """
    ShortCircuit operator to check if GCP upload should proceed.
    Returns False to skip downstream upload tasks if config missing.
    """
    return check_gcp_config()


# Create task factories that take condition as parameter
def create_fetch_raw_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        download_raw_trials_csv(
            raw_file_path=abs_path(config["raw_path"]),
            condition_query=config["query"],
        )
    return task


def create_schema_raw_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        schema_dir = abs_path(config["schema_dir"])
        os.makedirs(schema_dir, exist_ok=True)

        csv_path = abs_path(config["raw_path"])
        baseline_schema_path = abs_path(config["raw_schema_path"])
        report_path = os.path.join(abs_path(config["reports_dir"]), "schema_raw_report.json")

        run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_schema_path,
            report_path=report_path,
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="warn", 
            allow_new_columns=True,
        )
    return task


def create_enrich_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        enrich_trials_csv(
            raw_file_path=abs_path(config["raw_path"]),
            enriched_file_path=abs_path(config["enriched_path"]),
            disease=config["disease"],
            classifier=config["classifier"],
        )
    return task


def create_schema_processed_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        schema_dir = abs_path(config["schema_dir"])
        os.makedirs(schema_dir, exist_ok=True)

        csv_path = abs_path(config["enriched_path"])
        baseline_schema_path = abs_path(config["processed_schema_path"])
        report_path = os.path.join(abs_path(config["reports_dir"]), "schema_processed_report.json")

        run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_schema_path,
            report_path=report_path,
            required_columns=PROCESSED_REQUIRED_DEFAULT,
            mode="enforce",
            allow_new_columns=True,
        )
    return task


def create_validate_task(condition: str):
    def task(**context) -> bool:
        config = get_config_for_condition(condition)
        return run_validation(enriched_file_path=abs_path(config["enriched_path"]))
    return task


def create_quality_task(condition: str):
    def task(**context) -> bool:
        config = get_config_for_condition(condition)
        reports_dir = abs_path(config["reports_dir"])

        anomalies = run_quality_checks(
            enriched_file_path=abs_path(config["enriched_path"]),
            stats_path=os.path.join(reports_dir, "quality_stats.json"),
            anomalies_path=os.path.join(reports_dir, "anomalies.json"),
        )

        return not anomalies_found(anomalies)
    return task


def create_stats_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        reports_dir = abs_path(config["reports_dir"])

        stats = compute_stats(
            enriched_file_path=abs_path(config["enriched_path"]),
            stats_path=os.path.join(reports_dir, "stats.json"),
        )

        log.info(f"[{condition}] Stats computed: total_trials={stats.get('total_trials')}")
        context["ti"].xcom_push(key=f"{condition}_total_trials", value=stats.get("total_trials"))
    return task


def create_anomaly_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        anomalies_path = os.path.join(abs_path(config["reports_dir"]), "anomalies.json")

        log.info(f"[{condition}] Checking for anomalies at {anomalies_path}...")
        if os.path.exists(anomalies_path):
            with open(anomalies_path) as f:
                anomalies = json.load(f)
            context["ti"].xcom_push(key=f"{condition}_anomalies_found", value=anomalies_found(anomalies))
        else:
            context["ti"].xcom_push(key=f"{condition}_anomalies_found", value=None)
    return task


def create_bias_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        reports_dir = abs_path(config["reports_dir"])
        enriched = abs_path(config["enriched_path"])

        log.info(f"[{condition}] Generating bias report for {enriched}...")
        if not os.path.exists(enriched):
            context["ti"].xcom_push(key=f"{condition}_bias_level", value=None)
            return

        df = pd.read_csv(enriched)
        report = generate_bias_report(df)
        save_bias_report(report, bias_path=os.path.join(reports_dir, "bias_report.json"))
        context["ti"].xcom_push(key=f"{condition}_bias_level", value=report.get("bias_level"))
    return task


def create_save_reports_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        reports_dir = abs_path(config["reports_dir"])
        os.makedirs(reports_dir, exist_ok=True)

        ti = context["ti"]

        total = ti.xcom_pull(key=f"{condition}_total_trials")
        bias_level = ti.xcom_pull(key=f"{condition}_bias_level")

        stats_path = os.path.join(reports_dir, "stats.json")
        anomalies_path = os.path.join(reports_dir, "anomalies.json")
        bias_path = os.path.join(reports_dir, "bias_report.json")
        schema_raw_report = os.path.join(reports_dir, "schema_raw_report.json")
        schema_processed_report = os.path.join(reports_dir, "schema_processed_report.json")

        summary = {
            "pipeline_run_date": datetime.now().isoformat(),
            "condition": condition,
            "total_trials": total,
            "bias_level": bias_level,
            "gcp_uploads_enabled": check_gcp_config(),
            "reports": {
                "stats": safe_path(stats_path),
                "anomalies": safe_path(anomalies_path),
                "bias": safe_path(bias_path),
                "schema_raw_report": safe_path(schema_raw_report),
                "schema_processed_report": safe_path(schema_processed_report),
                "raw_schema_baseline": safe_path(abs_path(config["raw_schema_path"])),
                "processed_schema_baseline": safe_path(abs_path(config["processed_schema_path"])),
            },
            "notes": {
                "validate_short_circuit": ti.xcom_pull(task_ids=f"{condition}_validate") is False,
                "quality_short_circuit": ti.xcom_pull(task_ids=f"{condition}_quality") is False,
            },
        }

        with open(os.path.join(reports_dir, f"pipeline_summary_{condition}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        log.info(f"[{condition}] ✓ Summary saved | trials={total} | bias={bias_level}")
    return task


def create_upload_raw_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        log.info(f"[{condition}] Uploading raw data to GCS bucket!!")
        upload_raw_to_gcs(
            raw_file_path=abs_path(config["raw_path"]),
            condition=config["disease"],
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
        )
        log.info(f"[{condition}] Successfully uploaded raw data to GCS bucket!!")
    return task


def create_upload_reports_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        log.info(f"[{condition}] Uploading reports to GCS bucket!!")
        upload_reports_to_gcs(
            reports_dir=abs_path(config["reports_dir"]),
            condition=config["disease"],
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
        )
        log.info(f"[{condition}] Successfully uploaded reports to GCS bucket!!")
    return task


def create_upload_firestore_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        log.info(f"[{condition}] Uploading processed data to Firestore DB!!")
        upload_enriched_to_firestore(
            enriched_file_path=abs_path(config["enriched_path"]),
            condition=config["disease"],
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
        )
        log.info(f"[{condition}] Successfully uploaded processed data to Firestore DB!!")
    return task


with DAG(
    dag_id="clinical_trials_data_pipeline",
    description="Parallel clinical trials pipeline for all conditions",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "clinical-trials", "parallel"],
    default_args={
        "owner": "sanika",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "email_on_failure": False,
    },
) as dag:
    
    # Create parallel task groups for each condition
    for condition in CONDITIONS:
        with TaskGroup(group_id=f"{condition}_pipeline") as condition_group:
            
            fetch_raw = PythonOperator(
                task_id="fetch_raw",
                python_callable=create_fetch_raw_task(condition),
                execution_timeout=timedelta(minutes=30),
            )

            schema_raw = PythonOperator(
                task_id="schema_raw",
                python_callable=create_schema_raw_task(condition),
                execution_timeout=timedelta(minutes=5),
            )

            enrich = PythonOperator(
                task_id="enrich",
                python_callable=create_enrich_task(condition),
                execution_timeout=timedelta(minutes=5),
            )

            schema_processed = PythonOperator(
                task_id="schema_processed",
                python_callable=create_schema_processed_task(condition),
                execution_timeout=timedelta(minutes=5),
            )

            validate = ShortCircuitOperator(
                task_id="validate",
                python_callable=create_validate_task(condition),
                execution_timeout=timedelta(minutes=5),
                ignore_downstream_trigger_rules=False,
            )

            quality = ShortCircuitOperator(
                task_id="quality",
                python_callable=create_quality_task(condition),
                execution_timeout=timedelta(minutes=10),
                ignore_downstream_trigger_rules=False,
            )

            stats = PythonOperator(
                task_id="stats",
                python_callable=create_stats_task(condition),
                execution_timeout=timedelta(minutes=5),
                trigger_rule="all_done",
            )

            anomaly = PythonOperator(
                task_id="anomaly",
                python_callable=create_anomaly_task(condition),
                execution_timeout=timedelta(minutes=5),
                trigger_rule="all_done",
            )

            bias = PythonOperator(
                task_id="bias",
                python_callable=create_bias_task(condition),
                execution_timeout=timedelta(minutes=5),
                trigger_rule="all_done",
            )

            save_reports = PythonOperator(
                task_id="save_reports",
                python_callable=create_save_reports_task(condition),
                execution_timeout=timedelta(minutes=5),
                trigger_rule="all_done",
            )

            # Check GCP configuration
            check_gcp = ShortCircuitOperator(
                task_id="check_gcp_config",
                python_callable=task_check_gcp_config,
                execution_timeout=timedelta(minutes=1),
            )

            upload_raw_files_gcs = PythonOperator(
                task_id="upload_raw_files_gcs",
                python_callable=create_upload_raw_task(condition),
                execution_timeout=timedelta(minutes=10),
            )

            upload_reports_gcs = PythonOperator(
                task_id="upload_reports_gcs",
                python_callable=create_upload_reports_task(condition),
                execution_timeout=timedelta(minutes=10),
            )

            upload_firestore = PythonOperator(
                task_id="upload_firestore",
                python_callable=create_upload_firestore_task(condition),
                execution_timeout=timedelta(minutes=15),
            )

            # Pipeline flow for this condition
            fetch_raw >> schema_raw >> enrich >> schema_processed >> validate >> quality >> [stats, anomaly, bias]
            [stats, anomaly, bias] >> save_reports
            save_reports >> check_gcp
            check_gcp >> upload_raw_files_gcs
            check_gcp >> upload_reports_gcs
            check_gcp >> upload_firestore