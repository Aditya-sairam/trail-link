"""
Alert Cloud Function
=====================
Triggered by GCS object notification when evaluate_rag.py uploads summary.json.

Reads the summary, writes custom metrics to Cloud Monitoring.
Cloud Monitoring Alert Policies watch these metrics and send
email/Slack notifications natively — no third-party service needed.

Trigger: GCS finalize event on eval_results/*/summary.json
"""

from __future__ import annotations

import os
import json
import logging
import functions_framework
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud import monitoring_v3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")

# ── Custom metric types — must be registered in Cloud Monitoring ───────────────
METRIC_AVG_SCORE        = "custom.googleapis.com/triallink/rag_avg_score"
METRIC_PATIENTS_EVAL    = "custom.googleapis.com/triallink/patients_evaluated"
METRIC_ELIGIBLE_PCT     = "custom.googleapis.com/triallink/eligible_percentage"
METRIC_BREACHES_COUNT   = "custom.googleapis.com/triallink/alert_breaches_count"
METRIC_NOT_ELIGIBLE_PCT = "custom.googleapis.com/triallink/not_eligible_percentage"


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

@functions_framework.cloud_event
def handle_eval_complete(cloud_event):
    """
    Triggered when a file is written to GCS eval bucket.
    Only processes files matching eval_results/*/summary.json
    """
    data        = cloud_event.data
    bucket_name = data["bucket"]
    file_path   = data["name"]

    logger.info(f"GCS event: gs://{bucket_name}/{file_path}")

    # Only process summary.json files
    if not file_path.endswith("summary.json"):
        logger.info(f"Ignoring: {file_path}")
        return

    # Read summary from GCS
    try:
        client  = storage.Client(project=GCP_PROJECT_ID)
        bucket  = client.bucket(bucket_name)
        blob    = bucket.blob(file_path)
        summary = json.loads(blob.download_as_text())
        logger.info(f"Read summary: {len(summary)} keys")
    except Exception as e:
        logger.error(f"Failed to read summary: {e}")
        return

    # Write metrics to Cloud Monitoring
    # Alert Policies in GCP Console watch these and send emails automatically
    write_custom_metrics(summary)

    # Log structured alert for Cloud Logging
    # (Cloud Logging can also trigger log-based alerts)
    log_alert_status(summary, bucket_name, file_path)


# ══════════════════════════════════════════════════════════════════════════════
# WRITE CUSTOM METRICS TO CLOUD MONITORING
# ══════════════════════════════════════════════════════════════════════════════

def write_custom_metrics(summary: dict) -> None:
    """
    Write evaluation metrics as custom time series to Cloud Monitoring.

    Once written, you set Alert Policies in GCP Console that watch
    these metrics and fire email/Slack notifications when thresholds breach.

    Metrics written:
      - triallink/rag_avg_score          → alert if < 3.0
      - triallink/patients_evaluated     → alert if < 3
      - triallink/eligible_percentage    → alert if < 30
      - triallink/not_eligible_percentage→ alert if > 70
      - triallink/alert_breaches_count   → alert if > 0
    """
    try:
        client  = monitoring_v3.MetricServiceClient()
        project = f"projects/{GCP_PROJECT_ID}"
        now     = datetime.now(timezone.utc)

        verdicts     = summary.get("verdict_distribution", {})
        total        = sum(verdicts.values()) or 1
        avg_score    = summary.get("average_overall_score") or 0
        evaluated    = summary["evaluation_run"]["patients_evaluated"]
        eligible_pct = round(verdicts.get("ELIGIBLE", 0) / total * 100, 1)
        not_elig_pct = round(verdicts.get("NOT ELIGIBLE", 0) / total * 100, 1)
        breaches     = len(summary.get("alert_breaches", []))

        metrics = {
            METRIC_AVG_SCORE        : float(avg_score),
            METRIC_PATIENTS_EVAL    : float(evaluated),
            METRIC_ELIGIBLE_PCT     : float(eligible_pct),
            METRIC_NOT_ELIGIBLE_PCT : float(not_elig_pct),
            METRIC_BREACHES_COUNT   : float(breaches),
        }

        time_series_list = []
        for metric_type, value in metrics.items():
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.resource.type = "global"
            series.resource.labels["project_id"] = GCP_PROJECT_ID

            point = monitoring_v3.Point()
            point.interval.end_time.seconds = int(now.timestamp())
            point.value.double_value = value
            series.points = [point]
            time_series_list.append(series)

        client.create_time_series(
            name=project,
            time_series=time_series_list
        )

        logger.info(f"Custom metrics written to Cloud Monitoring: {metrics}")

    except Exception as e:
        logger.error(f"Failed to write Cloud Monitoring metrics: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LOG STRUCTURED ALERT
# ══════════════════════════════════════════════════════════════════════════════

def log_alert_status(summary: dict, bucket: str, file_path: str) -> None:
    """
    Log structured JSON to Cloud Logging.
    Cloud Logging can be configured to forward CRITICAL logs
    to email/Slack via Log-Based Alerts in GCP Console.
    """
    alert_status = summary.get("alert_status", "OK")
    breaches     = summary.get("alert_breaches", [])

    log_entry = {
        "alert_type"        : "RAG_EVAL_COMPLETE",
        "alert_status"      : alert_status,
        "avg_overall_score" : summary.get("average_overall_score"),
        "patients_evaluated": summary["evaluation_run"]["patients_evaluated"],
        "verdict_distribution": summary.get("verdict_distribution", {}),
        "breaches_count"    : len(breaches),
        "breaches"          : breaches,
        "source_file"       : f"gs://{bucket}/{file_path}",
        "timestamp"         : summary["evaluation_run"]["timestamp"],
    }

    if alert_status == "OK":
        logger.info(f"RAG_EVAL_STATUS: {json.dumps(log_entry)}")
    else:
        # CRITICAL level — Cloud Logging log-based alerts can watch for this
        logger.critical(f"RAG_EVAL_ALERT: {json.dumps(log_entry)}")
        for b in breaches:
            logger.critical(
                f"THRESHOLD_BREACH severity={b['severity']} "
                f"metric={b['metric']} value={b['value']} "
                f"threshold={b['threshold']} message={b['message']}"
            )