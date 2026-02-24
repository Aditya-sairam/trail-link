"""
Firestore Upload
================
Uploads enriched clinical trials data to Firestore,
one document per trial, using NCT ID as the document ID.

If the same trial appears in a future pipeline run, it overwrites
the existing document but updates the pipeline_run_date field.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
from google.cloud import firestore

log = logging.getLogger(__name__)


def normalize_column_name(col: str) -> str:
    """
    Convert column names to snake_case.
    Example: "NCT Number" -> "nct_number"
    """
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def get_pipeline_watermark(
    project_id: str,
    database: str,
    condition: str,
) -> Optional[str]:
    """
    Reads the last_successful_update watermark from Firestore.
    Stored under collection 'pipeline_state', document = condition.
    """
    db = firestore.Client(project=project_id, database=database)
    doc = db.collection("pipeline_state").document(condition).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    return data.get("last_successful_update")


def set_pipeline_watermark(
    project_id: str,
    database: str,
    condition: str,
    last_successful_update: str,
) -> None:
    """
    Writes the watermark to Firestore.
    """
    db = firestore.Client(project=project_id, database=database)
    db.collection("pipeline_state").document(condition).set(
        {
            "last_successful_update": last_successful_update,
            "updated_at": datetime.now().isoformat(),
        },
        merge=True,
    )


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
        Document ID: {NCT_ID}
        Fields     : all enriched CSV columns (snake_case) + condition + pipeline_run_date
    """
    if not project_id:
        raise ValueError("project_id must be provided.")

    if not os.path.exists(enriched_file_path):
        raise FileNotFoundError(f"Enriched CSV not found at: {enriched_file_path}")

    df = pd.read_csv(enriched_file_path)
    df.columns = [normalize_column_name(col) for col in df.columns]
    records = df.to_dict(orient="records")

    db = firestore.Client(project=project_id, database=database)
    collection = db.collection(f"clinical_trials_{condition}")

    run_date = datetime.now().strftime("%Y-%m-%d")
    uploaded = 0
    skipped = 0

    for row in records:
        clean_row = {
            k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in row.items()
        }

        clean_row["condition"] = condition
        clean_row["pipeline_run_date"] = run_date

        nct_id = clean_row.get("nct_number") or clean_row.get("nctnumber")

        if nct_id:
            collection.document(str(nct_id)).set(clean_row)
            uploaded += 1
        else:
            log.warning(f"⚠️ Skipping trial without NCT ID: {clean_row.get('title', 'Unknown')}")
            skipped += 1

    log.info(f"✓ Uploaded {uploaded} documents to Firestore collection: clinical_trials_{condition}")
    if skipped > 0:
        log.warning(f"⚠️ Skipped {skipped} trials without NCT ID")

    return uploaded


def missing_nct_ids_in_firestore(
    *,
    project_id: str,
    database: str,
    condition: str,
    nct_ids: Iterable[str],
) -> List[str]:
    """
    Given a list of NCT IDs, return the subset that do NOT exist in Firestore.

    Collection: clinical_trials_{condition}
    Document ID: NCT ID (string)
    """
    ids = [str(x) for x in nct_ids if x]
    if not ids:
        return []

    db = firestore.Client(project=project_id, database=database)
    collection = db.collection(f"clinical_trials_{condition}")

    doc_refs = [collection.document(nct_id) for nct_id in ids]
    snaps = db.get_all(doc_refs)

    missing: List[str] = []
    for snap in snaps:
        if not snap.exists:
            missing.append(snap.id)

    return missing