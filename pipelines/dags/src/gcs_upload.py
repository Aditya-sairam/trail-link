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

    Args:
        raw_file_path: Local path to the raw CSV file.
        condition: Disease condition (e.g. "diabetes", "breast_cancer").
        bucket_name: GCS bucket name.
        project_id: GCP project ID.

    Returns:
        Full GCS path of the uploaded file (e.g. "raw/diabetes/2026-02-22/trials_raw.json").
    """
    if not bucket_name or not project_id:
        raise ValueError("bucket_name and project_id must be provided.")

    if not os.path.exists(raw_file_path):
        raise FileNotFoundError(f"Raw CSV not found at: {raw_file_path}")

    # Read CSV and convert to JSON
    df = pd.read_csv(raw_file_path)
    records = df.to_dict(orient="records")

    # Clean NaN values — JSON doesn't handle them well
    clean_records = [
        {k: (None if (isinstance(v, float) and pd.isna(v)) else v) for k, v in row.items()}
        for row in records
    ]

    json_bytes = json.dumps(clean_records, indent=2, default=str).encode("utf-8")

    # Build GCS path and upload
    run_date = datetime.now().strftime("%Y-%m-%d")
    gcs_path = f"raw/{condition}/{run_date}/trials_raw.json"

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(json_bytes, content_type="application/json")

    log.info(f"✓ Uploaded {len(clean_records)} raw records → gs://{bucket_name}/{gcs_path}")
    return gcs_path