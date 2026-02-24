# pipelines/dags/airflow.py

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from src.bias import generate_bias_report, save_bias_report
from src.conditions.registry import REGISTRY
from src.quality import anomalies_found, run_quality_checks
from src.schema import run_schema_checkpoint, RAW_REQUIRED_DEFAULT, PROCESSED_REQUIRED_DEFAULT
from src.stats import compute_stats
from src.validate import run_validation
from src.gcs_upload import upload_raw_to_gcs
from src.ingest import (
    download_raw_trials_csv,
    enrich_trials_csv,
    get_latest_update_date,
    get_recent_nct_ids_since,
)
from src.firestore_upload import (
    upload_enriched_to_firestore,
    get_pipeline_watermark,
    set_pipeline_watermark,
    missing_nct_ids_in_firestore,
)

BASE = os.getenv("TRAILLINK_BASE", "/opt/airflow/repo")
BUCKET_NAME = os.getenv("BUCKET_NAME")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Runs automatically for BOTH diseases, in parallel, every schedule run
CONDITIONS = ["diabetes", "breast_cancer"]

log = logging.getLogger(__name__)


def get_config_for(condition: str) -> dict:
    if condition not in REGISTRY:
        raise ValueError(f"Unknown condition: '{condition}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[condition]


def parse_partial_date(date_str: str) -> Optional[datetime]:
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    fmts = ["%Y-%m-%d", "%Y-%m", "%Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def abs_path(path: str) -> str:
    return path if path.startswith("/") else os.path.join(BASE, path)


def safe_path(p: str) -> str | None:
    return p if p and os.path.exists(p) else None


def get_xcom(ti, task_id: str, key: str, default=None):
    val = ti.xcom_pull(task_ids=task_id, key=key)
    return default if val is None else val


def check_gcp_config() -> bool:
    has_bucket = BUCKET_NAME is not None
    has_project = PROJECT_ID is not None
    has_firestore = FIRESTORE_DB is not None
    has_creds = GOOGLE_CREDS is not None and os.path.exists(GOOGLE_CREDS)

    gcp_configured = has_bucket and has_project and has_firestore and has_creds

    if not gcp_configured:
        log.warning("⚠️ GCP configuration incomplete - upload tasks will be skipped")
        log.warning(f"  BUCKET_NAME: {'✓' if has_bucket else '✗ Not set'}")
        log.warning(f"  GCP_PROJECT_ID: {'✓' if has_project else '✗ Not set'}")
        log.warning(f"  FIRESTORE_DB: {'✓' if has_firestore else '✗ Not set'}")
        log.warning(f"  GOOGLE_APPLICATION_CREDENTIALS: {'✓' if has_creds else '✗ Not found'}")
        log.info("💡 Pipeline will run locally. Use 'docker cp' to retrieve output files.")
    else:
        log.info("✅ GCP configuration complete - uploads to GCS and Firestore enabled")

    return gcp_configured


def task_check_gcp_config(**context) -> bool:
    return check_gcp_config()


def task_check_updates(*, condition: str, **context) -> bool:
    """
    Short-circuit: returns False if no new updates are available.
    Stores the latest api update date in XCom for later use.

    If Firestore is not configured, we still proceed (local run),
    and we will not attempt watermark comparisons.
    """
    config = get_config_for(condition)

    latest_api_update = get_latest_update_date(
        condition_query=config["query"],
        status="RECRUITING",
        page_size=100,
    )

    context["ti"].xcom_push(key="latest_api_update", value=latest_api_update)

    if not latest_api_update:
        log.info(f"[{condition}] No latest update date found from API. Proceeding with pipeline.")
        return True

    if not PROJECT_ID or not FIRESTORE_DB:
        log.info(f"[{condition}] Firestore not configured. Proceeding with pipeline (no watermark check).")
        return True

    last_watermark = get_pipeline_watermark(
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
        condition=condition,
    )

    if not last_watermark:
        log.info(f"[{condition}] No watermark found. Proceeding with pipeline.")
        return True

    latest_dt = parse_partial_date(str(latest_api_update))
    wm_dt = parse_partial_date(str(last_watermark))

    if not latest_dt or not wm_dt:
        log.info(f"[{condition}] Could not parse dates reliably. Proceeding with pipeline.")
        return True

    if latest_dt > wm_dt:
        log.info(f"[{condition}] New updates detected. latest_api_update={latest_api_update} > watermark={last_watermark}")
        return True

    log.info(f"[{condition}] No new updates. latest_api_update={latest_api_update} <= watermark={last_watermark}. Skipping run.")
    return False


def task_check_firestore_new_trials(*, condition: str, **context) -> bool:
    """
    Short-circuit:
    - Use watermark (stored in Firestore) to fetch candidate updated NCT IDs from API
    - Check whether each candidate exists in Firestore
    - If ALL candidates already exist -> skip pipeline branch
    - If ANY is missing -> proceed

    If Firestore isn't configured, proceed (local run).
    """
    config = get_config_for(condition)

    if not PROJECT_ID or not FIRESTORE_DB:
        log.info(f"[{condition}] Firestore config missing (PROJECT_ID/FIRESTORE_DB). Proceeding with pipeline.")
        return True

    watermark = get_pipeline_watermark(
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
        condition=condition,
    )

    if not watermark:
        log.info(f"[{condition}] No watermark found. Treat as first run, proceed.")
        return True

    candidate_ids = get_recent_nct_ids_since(
        condition_query=config["query"],
        status="RECRUITING",
        since_date=str(watermark),
        page_size=100,
        max_ids=500,
    )
    context["ti"].xcom_push(key="candidate_nct_ids", value=candidate_ids)

    if not candidate_ids:
        log.info(f"[{condition}] No candidate NCT IDs newer than watermark={watermark}. Skipping pipeline branch.")
        return False

    missing = missing_nct_ids_in_firestore(
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
        condition=condition,
        nct_ids=candidate_ids,
    )
    context["ti"].xcom_push(key="missing_nct_ids", value=missing)

    if not missing:
        log.info(f"[{condition}] All {len(candidate_ids)} candidate NCT IDs already exist in Firestore. Skipping branch.")
        return False

    log.info(f"[{condition}] {len(missing)} candidate NCT IDs missing in Firestore. Proceeding with branch.")
    return True


def task_fetch_raw(*, condition: str, **context):
    config = get_config_for(condition)
    download_raw_trials_csv(
        raw_file_path=abs_path(config["raw_path"]),
        condition_query=config["query"],
    )


def task_schema_raw(*, condition: str, **context):
    config = get_config_for(condition)

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


def task_enrich(*, condition: str, **context):
    config = get_config_for(condition)
    enrich_trials_csv(
        raw_file_path=abs_path(config["raw_path"]),
        enriched_file_path=abs_path(config["enriched_path"]),
        disease=config["disease"],
        classifier=config["classifier"],
    )


def task_schema_processed(*, condition: str, **context):
    config = get_config_for(condition)

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


def task_validate(*, condition: str, **context) -> bool:
    config = get_config_for(condition)
    return run_validation(enriched_file_path=abs_path(config["enriched_path"]))


def task_quality(*, condition: str, **context) -> bool:
    config = get_config_for(condition)
    reports_dir = abs_path(config["reports_dir"])

    anomalies = run_quality_checks(
        enriched_file_path=abs_path(config["enriched_path"]),
        stats_path=os.path.join(reports_dir, "quality_stats.json"),
        anomalies_path=os.path.join(reports_dir, "anomalies.json"),
    )

    return not anomalies_found(anomalies)


def task_stats_fn(*, condition: str, **context):
    config = get_config_for(condition)
    reports_dir = abs_path(config["reports_dir"])

    stats = compute_stats(
        enriched_file_path=abs_path(config["enriched_path"]),
        stats_path=os.path.join(reports_dir, "stats.json"),
    )

    log.info(f"[{condition}] Stats computed: total_trials={stats.get('total_trials')}")
    context["ti"].xcom_push(key="total_trials", value=stats.get("total_trials"))


def task_anomaly_fn(*, condition: str, **context):
    config = get_config_for(condition)
    anomalies_path = os.path.join(abs_path(config["reports_dir"]), "anomalies.json")

    log.info(f"[{condition}] Checking for anomalies at {anomalies_path}...")
    if os.path.exists(anomalies_path):
        with open(anomalies_path) as f:
            anomalies = json.load(f)
        context["ti"].xcom_push(key="anomalies_found", value=anomalies_found(anomalies))
    else:
        context["ti"].xcom_push(key="anomalies_found", value=None)


def task_bias_fn(*, condition: str, **context):
    config = get_config_for(condition)
    reports_dir = abs_path(config["reports_dir"])
    enriched = abs_path(config["enriched_path"])

    log.info(f"[{condition}] Generating bias report for {enriched}...")
    if not os.path.exists(enriched):
        context["ti"].xcom_push(key="bias_level", value=None)
        return

    df = pd.read_csv(enriched)
    report = generate_bias_report(df)
    save_bias_report(report, bias_path=os.path.join(reports_dir, "bias_report.json"))
    context["ti"].xcom_push(key="bias_level", value=report.get("bias_level"))


def task_save_reports(*, condition: str, **context):
    config = get_config_for(condition)
    reports_dir = abs_path(config["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)

    ti = context["ti"]
    safe = condition.replace(" ", "_").lower()

    total = get_xcom(ti, f"task_stats__{safe}", "total_trials", default=None)
    bias_level = get_xcom(ti, f"task_bias__{safe}", "bias_level", default=None)

    stats_path = os.path.join(reports_dir, "stats.json")
    anomalies_path = os.path.join(reports_dir, "anomalies.json")
    bias_path = os.path.join(reports_dir, "bias_report.json")
    schema_raw_report = os.path.join(reports_dir, "schema_raw_report.json")
    schema_processed_report = os.path.join(reports_dir, "schema_processed_report.json")

    summary = {
        "pipeline_run_date": datetime.now().isoformat(),
        "condition": config["disease"],
        "total_trials": total,
        "bias_level": bias_level,
        "gcp_uploads_enabled": check_gcp_config(),
        "firestore_diff": {
            "candidate_nct_ids": ti.xcom_pull(task_ids=f"task_check_firestore_new_trials__{safe}", key="candidate_nct_ids"),
            "missing_nct_ids": ti.xcom_pull(task_ids=f"task_check_firestore_new_trials__{safe}", key="missing_nct_ids"),
        },
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
            "validate_short_circuit": ti.xcom_pull(task_ids=f"task_validate__{safe}") is False,
            "quality_short_circuit": ti.xcom_pull(task_ids=f"task_quality__{safe}") is False,
        },
    }

    with open(os.path.join(reports_dir, "pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"[{condition}] ✓ Summary saved | trials={total} | bias={bias_level}")


def task_upload_gcs(*, condition: str, **context):
    config = get_config_for(condition)
    log.info(f"[{condition}] Uploading raw data in json format to GCS bucket")
    upload_raw_to_gcs(
        raw_file_path=abs_path(config["raw_path"]),
        condition=config["disease"],
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
    )
    log.info(f"[{condition}] Successfully uploaded raw data to GCS")


def task_upload_firestore(*, condition: str, **context):
    config = get_config_for(condition)
    log.info(f"[{condition}] Uploading processed data to Firestore")
    upload_enriched_to_firestore(
        enriched_file_path=abs_path(config["enriched_path"]),
        condition=config["disease"],
        project_id=PROJECT_ID,
        database=FIRESTORE_DB,
    )
    log.info(f"[{condition}] Successfully uploaded processed data to Firestore")


def task_update_watermark(*, condition: str, **context):
    """
    Update watermark at the very end of a successful run.
    Uses the latest_api_update captured at task_check_updates time.
    """
    if not PROJECT_ID or not FIRESTORE_DB:
        log.info(f"[{condition}] Firestore not configured, skipping watermark update.")
        return

    ti = context["ti"]
    safe = condition.replace(" ", "_").lower()
    latest_api_update = ti.xcom_pull(task_ids=f"task_check_updates__{safe}", key="latest_api_update")

    if latest_api_update:
        set_pipeline_watermark(
            project_id=PROJECT_ID,
            database=FIRESTORE_DB,
            condition=condition,
            last_successful_update=str(latest_api_update),
        )
        log.info(f"[{condition}] Watermark updated to {latest_api_update}")
        return

    log.warning(f"[{condition}] No latest_api_update found in XCom, watermark not updated.")


with DAG(
    dag_id="clinical_trials_data_pipeline",
    description="Clinical trials pipeline for multiple conditions (auto, parallel)",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "clinical-trials", "multi-condition"],
    default_args={
        "owner": "sanika",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "email_on_failure": False,
    },
) as dag:

    # One shared gate for uploads. Branches will still run locally even if this is False.
    check_gcp = ShortCircuitOperator(
        task_id="task_check_gcp_config",
        python_callable=task_check_gcp_config,
        execution_timeout=timedelta(minutes=1),
    )

    # Build two parallel disease branches
    for condition in CONDITIONS:
        safe = condition.replace(" ", "_").lower()

        check_updates = ShortCircuitOperator(
            task_id=f"task_check_updates__{safe}",
            python_callable=task_check_updates,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            ignore_downstream_trigger_rules=False,
        )

        check_firestore_new = ShortCircuitOperator(
            task_id=f"task_check_firestore_new_trials__{safe}",
            python_callable=task_check_firestore_new_trials,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            ignore_downstream_trigger_rules=False,
        )

        fetch_raw = PythonOperator(
            task_id=f"task_fetch_raw__{safe}",
            python_callable=task_fetch_raw,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=30),
        )

        schema_raw = PythonOperator(
            task_id=f"task_schema_raw__{safe}",
            python_callable=task_schema_raw,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
        )

        enrich = PythonOperator(
            task_id=f"task_enrich__{safe}",
            python_callable=task_enrich,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
        )

        schema_processed = PythonOperator(
            task_id=f"task_schema_processed__{safe}",
            python_callable=task_schema_processed,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
        )

        validate = ShortCircuitOperator(
            task_id=f"task_validate__{safe}",
            python_callable=task_validate,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            ignore_downstream_trigger_rules=False,
        )

        quality = ShortCircuitOperator(
            task_id=f"task_quality__{safe}",
            python_callable=task_quality,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=10),
            ignore_downstream_trigger_rules=False,
        )

        stats = PythonOperator(
            task_id=f"task_stats__{safe}",
            python_callable=task_stats_fn,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            trigger_rule="all_done",
        )

        anomaly = PythonOperator(
            task_id=f"task_anomaly__{safe}",
            python_callable=task_anomaly_fn,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            trigger_rule="all_done",
        )

        bias = PythonOperator(
            task_id=f"task_bias__{safe}",
            python_callable=task_bias_fn,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            trigger_rule="all_done",
        )

        save_reports = PythonOperator(
            task_id=f"task_save_reports__{safe}",
            python_callable=task_save_reports,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=5),
            trigger_rule="all_done",
        )

        upload_gcs = PythonOperator(
            task_id=f"task_upload_gcs__{safe}",
            python_callable=task_upload_gcs,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=10),
        )

        upload_firestore = PythonOperator(
            task_id=f"task_upload_firestore__{safe}",
            python_callable=task_upload_firestore,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=15),
        )

        update_watermark = PythonOperator(
            task_id=f"task_update_watermark__{safe}",
            python_callable=task_update_watermark,
            op_kwargs={"condition": condition},
            execution_timeout=timedelta(minutes=2),
            trigger_rule="all_success",
        )

        # Wiring for this condition branch
        check_updates >> check_firestore_new >> fetch_raw
        fetch_raw >> schema_raw >> enrich >> schema_processed >> validate >> quality

        quality >> [stats, anomaly, bias]
        [stats, anomaly, bias] >> save_reports

        # Upload gate + watermark per condition
        save_reports >> check_gcp
        check_gcp >> [upload_gcs, upload_firestore] >> update_watermark