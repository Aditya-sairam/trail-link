import json
import os
from typing import Any, Dict

import pandas as pd

from src.pipelines.breast_cancer.ingest import get_logger


# ClinicalTrials.gov CSV export uses "NCT Number" (not "NCTId")
NCT_ID_COL = "NCT Number"


def generate_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight stats snapshot saved every run.
    This becomes your time series of pipeline health over time.
    """
    stats: Dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "missing_by_col": df.isna().sum().to_dict(),
    }

    # Track unique trial IDs if the ID column exists
    if NCT_ID_COL in df.columns:
        stats["unique_nct_number"] = int(df[NCT_ID_COL].nunique())

    return stats


def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple anomaly rules that are:
    - easy to understand
    - easy to test
    - good enough for a pipeline-only submission
    """
    anomalies: Dict[str, Any] = {
        "missing_required_columns": [],
        "duplicate_nct_number_found": False,
        "high_missing_columns": [],
    }

    # Required columns for your pipeline outputs
    required_cols = [NCT_ID_COL]
    for c in required_cols:
        if c not in df.columns:
            anomalies["missing_required_columns"].append(c)

    # Duplicate ID check
    if NCT_ID_COL in df.columns:
        anomalies["duplicate_nct_number_found"] = bool(df[NCT_ID_COL].duplicated().any())

    # Flag columns with too much missingness (70%+)
    for c in df.columns:
        missing_frac = float(df[c].isna().mean())
        if missing_frac >= 0.70:
            anomalies["high_missing_columns"].append(
                {"col": c, "missing_frac": missing_frac}
            )

    return anomalies


def save_json(obj: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return out_path


def anomalies_found(anomalies: Dict[str, Any]) -> bool:
    """
    Used by Airflow to decide whether to alert.
    """
    return bool(
        anomalies["missing_required_columns"]
        or anomalies["duplicate_nct_number_found"]
        or anomalies["high_missing_columns"]
    )
# def anomalies_found(anomalies):
#     return True


def run_quality_checks(enriched_csv_path: str, stats_path: str, anomalies_path: str) -> None:
    """
    Loads enriched CSV and produces stats + anomaly reports.
    """
    logger = get_logger("breast_cancer")

    df = pd.read_csv(enriched_csv_path)

    stats = generate_stats(df)
    save_json(stats, stats_path)
    logger.info("Saved stats report: %s", stats_path)

    anomalies = detect_anomalies(df)
    save_json(anomalies, anomalies_path)
    logger.info("Saved anomalies report: %s", anomalies_path)

    if anomalies_found(anomalies):
        logger.warning("Anomalies detected. Check: %s", anomalies_path)
    else:
        logger.info("No anomalies detected.")
