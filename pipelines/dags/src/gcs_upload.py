"""
GCS Upload
==========
Converts raw clinical trials CSV to JSON and uploads to GCS bucket.

Usage:
    upload_raw_to_gcs(
        raw_file_path="/opt/airflow/repo/data/diabetes/raw/diabetes_trials_raw.csv",
        condition="diabetes",
        bucket_name="triallink-pipeline-data-trial-link",
        project_id="trial-link",
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import pandas as pd
from google.cloud import storage

log = logging.getLogger(__name__)


def upload_raw_to_gcs(
    raw_file_path: str,
    condition: str,
    bucket_name: str,
    project_id: str,
) -> str:
    """
    Read raw CSV, convert to JSON, upload to GCS.

    GCS path format:
        raw/{condition}/{YYYY-MM-DD}/trials_raw.json
    """
    if not bucket_name or not project_id:
        raise ValueError("bucket_name and project_id must be provided.")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Raw CSV not found at: {raw_file_path}")

    df = pd.read_csv(raw_file_path)
    records = df.to_dict(orient="records")

    clean_records = [
        {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in row.items()}
        for row in records
    ]

    json_bytes = json.dumps(clean_records, indent=2, default=str).encode("utf-8")

    run_date = datetime.now().strftime("%Y-%m-%d")
    gcs_path = f"raw/{condition}/{run_date}/trials_raw.json"

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(json_bytes, content_type="application/json")

    log.info(f"✓ Uploaded {len(clean_records)} raw records → gs://{bucket_name}/{gcs_path}")
    return gcs_path


def upload_reports_to_gcs(
    reports_dir: str,
    condition: str,
    bucket_name: str,
    project_id: str,
) -> list[str]:
    """
    Upload all JSON report files from reports_dir to GCS.

    GCS path format:
        reports/{condition}/{YYYY-MM-DD}/{report_name}.json
    """
    if not bucket_name or not project_id:
        raise ValueError("bucket_name and project_id must be provided.")

    if not os.path.exists(reports_dir):
        raise FileNotFoundError(f"Reports directory not found at: {reports_dir}")

    report_files = [f for f in os.listdir(reports_dir) if f.endswith(".json")]
    if not report_files:
        log.warning(f"No JSON report files found in: {reports_dir}")
        return []

    run_date = datetime.now().strftime("%Y-%m-%d")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    uploaded_paths: list[str] = []

    for report_file in report_files:
        local_path = os.path.join(reports_dir, report_file)
        with open(local_path, "r") as f:
            content = f.read()

        gcs_path = f"reports/{condition}/{run_date}/{report_file}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(content, content_type="application/json")

        log.info(f"✓ Uploaded report → gs://{bucket_name}/{gcs_path}")
        uploaded_paths.append(gcs_path)

    log.info(f"✓ Uploaded {len(uploaded_paths)} reports → gs://{bucket_name}/reports/{condition}/{run_date}/")
    return uploaded_paths