from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from src.embed import embed_conditions
from src.bias import generate_bias_report, save_bias_report
from src.conditions.registry import REGISTRY
from src.firestore_upload import upload_enriched_to_firestore
from src.gcs_upload import upload_raw_to_gcs, upload_reports_to_gcs
from src.ingest import download_raw_trials_csv, enrich_trials_csv
from src.quality import anomalies_found, run_quality_checks
from src.schema import PROCESSED_REQUIRED_DEFAULT, RAW_REQUIRED_DEFAULT, run_schema_checkpoint
from src.stats import compute_stats
from src.validate import run_validation


BASE = os.getenv("TRAILLINK_BASE", "/opt/airflow/repo")
BUCKET_NAME = os.getenv("BUCKET_NAME") or os.getenv("CLINICAL_TRIALS_BUCKET")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB") or os.getenv("FIRESTORE_DATABASE")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

CONDITIONS = ["diabetes"]
log = logging.getLogger(__name__)


def get_config_for_condition(condition: str) -> dict:
    if condition not in REGISTRY:
        raise ValueError(f"Unknown condition: '{condition}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[condition]


def abs_path(path: str) -> str:
    return path if path.startswith("/") else os.path.join(BASE, path)


def safe_path(p: str) -> str | None:
    return p if p and os.path.exists(p) else None


def check_gcp_config() -> bool:
    has_bucket = BUCKET_NAME is not None
    has_project = PROJECT_ID is not None
    has_firestore = FIRESTORE_DB is not None
    has_creds = GOOGLE_CREDS is not None and os.path.exists(GOOGLE_CREDS)

    gcp_configured = has_bucket and has_project and has_firestore 
    if not gcp_configured:
        log.warning("GCP configuration incomplete - upload tasks will be skipped")
        log.warning(f"  BUCKET_NAME / CLINICAL_TRIALS_BUCKET: {'yes' if has_bucket else 'missing'}")
        log.warning(f"  GCP_PROJECT_ID: {'yes' if has_project else 'missing'}")
        log.warning(f"  FIRESTORE_DB / FIRESTORE_DATABASE: {'yes' if has_firestore else 'missing'}")
        log.warning(f"  GOOGLE_APPLICATION_CREDENTIALS: {'yes' if has_creds else 'missing'}")
    return gcp_configured


def task_check_gcp_config(**context) -> bool:
    return check_gcp_config()


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
        os.makedirs(abs_path(config["schema_dir"]), exist_ok=True)
        run_schema_checkpoint(
            csv_path=abs_path(config["raw_path"]),
            baseline_schema_path=abs_path(config["raw_schema_path"]),
            report_path=os.path.join(abs_path(config["reports_dir"]), "schema_raw_report.json"),
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
        os.makedirs(abs_path(config["schema_dir"]), exist_ok=True)
        run_schema_checkpoint(
            csv_path=abs_path(config["enriched_path"]),
            baseline_schema_path=abs_path(config["processed_schema_path"]),
            report_path=os.path.join(abs_path(config["reports_dir"]), "schema_processed_report.json"),
            required_columns=PROCESSED_REQUIRED_DEFAULT,
            mode="warn",
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
        context["ti"].xcom_push(key=f"{condition}_total_trials", value=stats.get("total_trials"))
        log.info(f"[{condition}] Stats computed: total_trials={stats.get('total_trials')}")
    return task


def create_anomaly_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        anomalies_path = os.path.join(abs_path(config["reports_dir"]), "anomalies.json")
        if os.path.exists(anomalies_path):
            with open(anomalies_path) as f:
                anomalies = json.load(f)
            context["ti"].xcom_push(
                key=f"{condition}_anomalies_found",
                value=anomalies_found(anomalies),
            )
        else:
            context["ti"].xcom_push(key=f"{condition}_anomalies_found", value=None)
    return task


def create_notify_anomaly_email_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        anomalies_path = os.path.join(abs_path(config["reports_dir"]), "anomalies.json")

        if not os.path.exists(anomalies_path):
            log.info(f"[{condition}] No anomalies file found, skipping email")
            return

        with open(anomalies_path) as f:
            anomalies = json.load(f)

        has_anomaly = anomalies_found(anomalies)
        force_send = False

        if not has_anomaly and not force_send:
            log.info(f"[{condition}] No anomaly detected, skipping email")
            return

        msg = EmailMessage()
        msg["Subject"] = f"[TrialLink][TEST] Anomaly alert - {condition}"
        msg["From"] = "triallink@local.test"
        msg["To"] = "you@example.com"
        msg.set_content(
            f"Condition: {condition}\n"
            f"has_anomaly={has_anomaly}\n\n"
            f"anomalies:\n{json.dumps(anomalies, indent=2)}"
        )

        with smtplib.SMTP("host.docker.internal", 1025, timeout=10) as smtp:
            smtp.send_message(msg)

        log.info(f"[{condition}] Test anomaly email sent to MailHog")
    return task


def embed_trials(**context):
    """
    Embed all unembedded clinical trials from Firestore and upsert
    their vectors into Vertex AI Vector Search index.

    Runs AFTER all conditions have been uploaded to Firestore.
    Only processes trials where embedded != True, so weekly runs
    only embed the NEW trials added that week.
    """
    results = embed_conditions(
        conditions=CONDITIONS,
        project_id=PROJECT_ID,
        force_reembed=False,
    )
    log.info(f"Embedding results: {results}")
    context["ti"].xcom_push(key="embedding_results", value=results)


def create_bias_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        reports_dir = abs_path(config["reports_dir"])
        enriched = abs_path(config["enriched_path"])
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

        summary = {
            "pipeline_run_date": datetime.now().isoformat(),
            "condition": condition,
            "total_trials": ti.xcom_pull(key=f"{condition}_total_trials"),
            "bias_level": ti.xcom_pull(key=f"{condition}_bias_level"),
            "gcp_uploads_enabled": check_gcp_config(),
            "reports": {
                "stats": safe_path(os.path.join(reports_dir, "stats.json")),
                "anomalies": safe_path(os.path.join(reports_dir, "anomalies.json")),
                "bias": safe_path(os.path.join(reports_dir, "bias_report.json")),
                "schema_raw_report": safe_path(os.path.join(reports_dir, "schema_raw_report.json")),
                "schema_processed_report": safe_path(os.path.join(reports_dir, "schema_processed_report.json")),
                "raw_schema_baseline": safe_path(abs_path(config["raw_schema_path"])),
                "processed_schema_baseline": safe_path(abs_path(config["processed_schema_path"])),
            },
            "notes": {
                "validate_short_circuit": ti.xcom_pull(task_ids=f"{condition}_pipeline.validate") is False,
                "quality_short_circuit": ti.xcom_pull(task_ids=f"{condition}_pipeline.quality") is False,
            },
        }

        with open(os.path.join(reports_dir, f"pipeline_summary_{condition}.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return task


def create_upload_raw_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        upload_raw_to_gcs(
            raw_file_path=abs_path(config["raw_path"]),
            condition=config["disease"],
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
        )
    return task


def create_upload_reports_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        upload_reports_to_gcs(
            reports_dir=abs_path(config["reports_dir"]),
            condition=config["disease"],
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
        )
    return task


def create_upload_firestore_task(condition: str):
    def task(**context):
        config = get_config_for_condition(condition)
        upload_enriched_to_firestore(
            enriched_file_path=abs_path(config["enriched_path"]),
            condition=config["disease"],
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
        )
    return task


with DAG(
    dag_id="clinical_trials_data_pipeline",
    description="Parallel clinical trials pipeline for all conditions",
    schedule_interval=None,
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

    # ── Per-condition pipelines ───────────────────────────────────────────────
    # Track the final upload_firestore task per condition so embed_trials
    # can depend on ALL conditions being uploaded before it runs
    firestore_upload_tasks = []

    for condition in CONDITIONS:
        with TaskGroup(group_id=f"{condition}_pipeline"):
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
            notify_anomaly_email = PythonOperator(
                task_id="notify_anomaly_email",
                python_callable=create_notify_anomaly_email_task(condition),
                execution_timeout=timedelta(minutes=2),
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
            embed = PythonOperator(
                task_id="embed_trials",
                python_callable=embed_trials,
                execution_timeout=timedelta(minutes=60),  # embedding can take a while
                trigger_rule="all_done",                  # run even if some conditions had issues
            )
            # ── Task dependencies ─────────────────────────────────────────────
            fetch_raw >> schema_raw >> enrich >> schema_processed >> validate >> quality >> [stats, anomaly, bias]
            anomaly >> notify_anomaly_email
            [stats, bias, notify_anomaly_email] >> save_reports
            save_reports >> check_gcp
            check_gcp >> upload_raw_files_gcs
            check_gcp >> upload_reports_gcs
            check_gcp >> upload_firestore

            # Track firestore upload task for embed dependency
            firestore_upload_tasks.append(upload_firestore)

            # ── Embed trials — runs AFTER all conditions are uploaded to Firestore ────
            # This ensures all new trials are in Firestore before we try to embed them
            

            # All firestore uploads must complete before embedding starts
            firestore_upload_tasks >> embed