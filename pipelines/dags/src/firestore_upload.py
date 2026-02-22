"""
Firestore Upload
================
Uploads enriched clinical trials data to Firestore,
one document per trial, using NCT ID as the document ID.

If the same trial appears in a future pipeline run, it overwrites
the existing document but updates the pipeline_run_date field.

Usage:
    upload_enriched_to_firestore(
        enriched_file_path="/opt/airflow/repo/data/diabetes/processed/diabetes_trials_enriched.csv",
        condition="diabetes",
        project_id="trial-link",
        database="patient-db-sai",
    )
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
from google.cloud import firestore

log = logging.getLogger(__name__)


def upload_enriched_to_firestore(
    enriched_file_path: str,
    condition: str,
    project_id: str,
    database: str = "patient-db-sai",
) -> int:
    """
    Read enriched CSV and upload each row as a Firestore document.

    Firestore structure:
        Collection : clinical_trials_{condition}
        Document ID: {NCT_ID}  (e.g. NCT01234567)
        Fields     : all enriched CSV columns + condition + pipeline_run_date

    Args:
        enriched_file_path: Local path to the enriched/processed CSV file.
        condition: Disease condition (e.g. "diabetes", "breast_cancer").
        project_id: GCP project ID.
        database: Firestore database name.

    Returns:
        Number of documents uploaded.
    """
    if not project_id:
        raise ValueError("project_id must be provided.")

    if not os.path.exists(enriched_file_path):
        raise FileNotFoundError(f"Enriched CSV not found at: {enriched_file_path}")

    df = pd.read_csv(enriched_file_path)
    records = df.to_dict(orient="records")

    db = firestore.Client(project=project_id, database=database)
    collection = db.collection(f"clinical_trials_{condition}")
    run_date = datetime.now().strftime("%Y-%m-%d")
    uploaded = 0

    for row in records:
        # Clean NaN values — Firestore doesn't accept float NaN
        clean_row = {
            k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in row.items()
        }

        # Add metadata fields
        clean_row["condition"] = condition
        clean_row["pipeline_run_date"] = run_date

        # Use NCT ID as document ID if available, else auto-generate
        nct_id = (
            clean_row.get("NCTId")
            or clean_row.get("nct_id")
            or clean_row.get("id")
        )

        if nct_id:
            # .set() overwrites if document already exists
            collection.document(str(nct_id)).set(clean_row)
        else:
            # No NCT ID — let Firestore auto-generate an ID
            collection.add(clean_row)

        uploaded += 1

    log.info(f"✓ Uploaded {uploaded} documents → Firestore collection: clinical_trials_{condition}")
    return uploaded