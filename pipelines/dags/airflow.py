# pipelines/dags/airflow.py

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from src.bias import generate_bias_report, save_bias_report
from src.conditions.registry import REGISTRY
from src.ingest import download_raw_trials_csv, enrich_trials_csv
from src.quality import anomalies_found, run_quality_checks
from src.schema import run_schema_checkpoint, RAW_REQUIRED_DEFAULT, PROCESSED_REQUIRED_DEFAULT
from src.stats import compute_stats
from src.validate import run_validation
from src.gcs_upload import upload_raw_to_gcs
from src.firestore_upload import upload_enriched_to_firestore


BASE = os.getenv("TRAILLINK_BASE", "/opt/airflow/repo")
BUCKET_NAME  = os.getenv("BUCKET_NAME")
PROJECT_ID   = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

log = logging.getLogger(__name__)


def get_config(context: dict) -> dict:
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}
    condition = conf.get("condition", "diabetes")

    if condition not in REGISTRY:
        raise ValueError(f"Unknown condition: '{condition}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[condition]


def abs_path(path: str) -> str:
    return path if path.startswith("/") else os.path.join(BASE, path)


def delta_csv_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_delta{ext or '.csv'}"


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


def task_fetch_raw(**context):
    config = get_config(context)
    meta = download_raw_trials_csv(
        raw_file_path=abs_path(config["raw_path"]),
        condition_query=config["query"],
    )
    ti = context["ti"]
    for key, value in meta.items():
        ti.xcom_push(key=key, value=value)
    ti.xcom_push(key="has_delta_data", value=bool(meta.get("delta_rows", 0)))


def task_has_delta(**context) -> bool:
    delta_rows = context["ti"].xcom_pull(task_ids="task_fetch_raw", key="delta_rows") or 0
    if int(delta_rows) <= 0:
        log.info("No new/updated trials fetched. Skipping downstream delta processing.")
        return False
    log.info(f"Delta processing enabled for {delta_rows} rows.")
    return True


def task_schema_raw(**context):
    config = get_config(context)
    ti = context["ti"]

    schema_dir = abs_path(config["schema_dir"])
    os.makedirs(schema_dir, exist_ok=True)

    csv_path = ti.xcom_pull(task_ids="task_fetch_raw", key="raw_delta_path") or abs_path(delta_csv_path(config["raw_path"]))
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


def task_enrich(**context):
    config = get_config(context)
    ti = context["ti"]
    raw_delta_path = ti.xcom_pull(task_ids="task_fetch_raw", key="raw_delta_path") or abs_path(delta_csv_path(config["raw_path"]))
    enriched_delta_path = abs_path(delta_csv_path(config["enriched_path"]))
    ti.xcom_push(key="enriched_delta_path", value=enriched_delta_path)

    enrich_trials_csv(
        raw_file_path=raw_delta_path,
        enriched_file_path=enriched_delta_path,
        disease=config["disease"],
        classifier=config["classifier"],
        full_enriched_file_path=abs_path(config["enriched_path"]),
    )


def task_schema_processed(**context):
    config = get_config(context)
    ti = context["ti"]

    schema_dir = abs_path(config["schema_dir"])
    os.makedirs(schema_dir, exist_ok=True)

    csv_path = ti.xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))
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


def task_validate(**context) -> bool:
    config = get_config(context)
    delta_path = context["ti"].xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))
    return run_validation(enriched_file_path=delta_path, min_rows=1)


def task_quality(**context) -> bool:
    config = get_config(context)
    reports_dir = abs_path(config["reports_dir"])
    delta_path = context["ti"].xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))

    anomalies = run_quality_checks(
        enriched_file_path=delta_path,
        stats_path=os.path.join(reports_dir, "quality_stats.json"),
        anomalies_path=os.path.join(reports_dir, "anomalies.json"),
    )

    return not anomalies_found(anomalies)


def task_stats_fn(**context):
    config = get_config(context)
    reports_dir = abs_path(config["reports_dir"])
    delta_path = context["ti"].xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))

    stats = compute_stats(
        enriched_file_path=delta_path,
        stats_path=os.path.join(reports_dir, "stats.json"),
    )

    log.info(f"Stats computed: total_trials={stats.get('total_trials')}")
    context["ti"].xcom_push(key="total_trials", value=stats.get("total_trials"))


def task_anomaly_fn(**context):
    config = get_config(context)
    anomalies_path = os.path.join(abs_path(config["reports_dir"]), "anomalies.json")

    log.info(f"Checking for anomalies at {anomalies_path}...")
    if os.path.exists(anomalies_path):
        with open(anomalies_path) as f:
            anomalies = json.load(f)
        context["ti"].xcom_push(key="anomalies_found", value=anomalies_found(anomalies))
    else:
        context["ti"].xcom_push(key="anomalies_found", value=None)


def task_bias_fn(**context):
    config = get_config(context)
    reports_dir = abs_path(config["reports_dir"])
    enriched = context["ti"].xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))

    log.info(f"Generating bias report for {enriched}...")
    if not os.path.exists(enriched):
        context["ti"].xcom_push(key="bias_level", value=None)
        return

    df = pd.read_csv(enriched)
    report = generate_bias_report(df)
    save_bias_report(report, bias_path=os.path.join(reports_dir, "bias_report.json"))
    context["ti"].xcom_push(key="bias_level", value=report.get("bias_level"))


def task_save_reports(**context):
    config = get_config(context)
    reports_dir = abs_path(config["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)

    ti = context["ti"]

    total = get_xcom(ti, "task_stats", "total_trials", default=None)
    bias_level = get_xcom(ti, "task_bias", "bias_level", default=None)

    delta_rows = get_xcom(ti, "task_fetch_raw", "delta_rows", default=0)
    fetched_rows = get_xcom(ti, "task_fetch_raw", "fetched_rows", default=0)
    snapshot_rows = get_xcom(ti, "task_fetch_raw", "snapshot_rows", default=None)
    since_date = get_xcom(ti, "task_fetch_raw", "since_date", default=None)
    raw_delta_path = get_xcom(ti, "task_fetch_raw", "raw_delta_path", default=abs_path(delta_csv_path(config["raw_path"])))
    enriched_delta_path = get_xcom(ti, "task_enrich", "enriched_delta_path", default=abs_path(delta_csv_path(config["enriched_path"])))

    stats_path = os.path.join(reports_dir, "stats.json")
    anomalies_path = os.path.join(reports_dir, "anomalies.json")
    bias_path = os.path.join(reports_dir, "bias_report.json")
    schema_raw_report = os.path.join(reports_dir, "schema_raw_report.json")
    schema_processed_report = os.path.join(reports_dir, "schema_processed_report.json")

    summary = {
        "pipeline_run_date": datetime.now().isoformat(),
        "condition": config["disease"],
        "total_trials": total,
        "delta_rows_processed": delta_rows,
        "incremental_fetch": {
            "enabled": True,
            "since_date": since_date,
            "fetched_rows_from_api": fetched_rows,
            "delta_rows": delta_rows,
            "snapshot_rows": snapshot_rows,
        },
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
        "data_files": {
            "raw_snapshot": safe_path(abs_path(config["raw_path"])),
            "raw_delta": safe_path(raw_delta_path),
            "enriched_snapshot": safe_path(abs_path(config["enriched_path"])),
            "enriched_delta": safe_path(enriched_delta_path),
        },
        "notes": {
            "validate_short_circuit": ti.xcom_pull(task_ids="task_validate") is False,
            "quality_short_circuit": ti.xcom_pull(task_ids="task_quality") is False,
        },
    }

    with open(os.path.join(reports_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"✓ Summary saved | condition={config['disease']} | trials={total} | bias={bias_level}")


def task_upload_gcs(**context):
    config = get_config(context)
    raw_delta_path = context["ti"].xcom_pull(task_ids="task_fetch_raw", key="raw_delta_path") or abs_path(delta_csv_path(config["raw_path"]))
    log.info("Uploading raw data in json format to GCS bucket!!")
    upload_raw_to_gcs(
        raw_file_path=raw_delta_path,
        condition=config["disease"],
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
    )
    log.info("Successfully Uploaded raw data in json format to GCS bucket!!")


def task_upload_firestore(**context):
    config = get_config(context)
    enriched_delta_path = context["ti"].xcom_pull(task_ids="task_enrich", key="enriched_delta_path") or abs_path(delta_csv_path(config["enriched_path"]))
    log.info("Uploading processed data in json format to FireStore DB!!")
    upload_enriched_to_firestore(
        enriched_file_path=enriched_delta_path,
        condition=config["disease"],
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
    )
    log.info("Successfully Uploaded processed data in json format to FireStore DB!!")


with DAG(
    dag_id="clinical_trials_data_pipeline",
    description="Generalised clinical trials pipeline (incremental + delta), trigger condition via conf",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    params={"condition": "breast_cancer"},
    tags=["mlops", "clinical-trials", "generalised"],
    default_args={
        "owner": "sanika",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "email_on_failure": False,
    },
) as dag:
    
    fetch_raw = PythonOperator(
        task_id="task_fetch_raw",
        python_callable=task_fetch_raw,
        execution_timeout=timedelta(minutes=30),
    )

    has_new_data = ShortCircuitOperator(
        task_id="task_has_new_data",
        python_callable=task_has_delta,
        execution_timeout=timedelta(minutes=2),
        ignore_downstream_trigger_rules=False,
    )

    schema_raw = PythonOperator(
        task_id="task_schema_raw",
        python_callable=task_schema_raw,
        execution_timeout=timedelta(minutes=5),
    )

    enrich = PythonOperator(
        task_id="task_enrich",
        python_callable=task_enrich,
        execution_timeout=timedelta(minutes=5),
    )

    schema_processed = PythonOperator(
        task_id="task_schema_processed",
        python_callable=task_schema_processed,
        execution_timeout=timedelta(minutes=5),
    )

    validate = ShortCircuitOperator(
        task_id="task_validate",
        python_callable=task_validate,
        execution_timeout=timedelta(minutes=5),
        ignore_downstream_trigger_rules=False,
    )

    quality = ShortCircuitOperator(
        task_id="task_quality",
        python_callable=task_quality,
        execution_timeout=timedelta(minutes=10),
        ignore_downstream_trigger_rules=False,
    )

    stats = PythonOperator(
        task_id="task_stats",
        python_callable=task_stats_fn,
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    anomaly = PythonOperator(
        task_id="task_anomaly",
        python_callable=task_anomaly_fn,
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    bias = PythonOperator(
        task_id="task_bias",
        python_callable=task_bias_fn,
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    save_reports = PythonOperator(
        task_id="task_save_reports",
        python_callable=task_save_reports,
        execution_timeout=timedelta(minutes=5),
        trigger_rule="all_done",
    )

    # Check GCP configuration before upload tasks
    check_gcp = ShortCircuitOperator(
        task_id="task_check_gcp_config",
        python_callable=task_check_gcp_config,
        execution_timeout=timedelta(minutes=1),
    )

    upload_gcs = PythonOperator(
        task_id="task_upload_gcs",
        python_callable=task_upload_gcs,
        execution_timeout=timedelta(minutes=10),
    )

    upload_firestore = PythonOperator(
        task_id="task_upload_firestore",
        python_callable=task_upload_firestore,
        execution_timeout=timedelta(minutes=15),
    )

    # Pipeline flow - main processing (always runs)
    fetch_raw >> has_new_data >> schema_raw >> enrich >> schema_processed >> validate >> quality >> [stats, anomaly, bias]
    
    # Wait for all analysis tasks to complete
    [stats, anomaly, bias] >> save_reports
    
    # Check GCP config AFTER save_reports completes
    save_reports >> check_gcp
    
    # Upload tasks only run if check_gcp returns True
    check_gcp >> upload_gcs
    check_gcp >> upload_firestore
