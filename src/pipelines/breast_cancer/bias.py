import json
import os
from typing import Dict, Any, List

import pandas as pd

from src.pipelines.breast_cancer.ingest import get_logger


DEFAULT_SLICE_COLUMNS = ["Sex", "Phase", "StudyType", "OverallStatus"]


def _value_counts_with_pct(series: pd.Series) -> Dict[str, Any]:
    """
    Returns both counts and percentages for a categorical series.
    """
    s = series.fillna("MISSING").astype(str)
    counts = s.value_counts(dropna=False).to_dict()
    pct = (s.value_counts(normalize=True, dropna=False)).to_dict()
    return {"counts": counts, "pct": pct}


def _missingness_by_slice(df: pd.DataFrame, slice_col: str, sample_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    For each sampled column, compute missing fraction per slice group.
    This highlights if certain subgroups systematically have more missing data.
    """
    s = df[slice_col].fillna("MISSING").astype(str)

    out: Dict[str, Dict[str, float]] = {}
    for c in sample_cols:
        if c not in df.columns:
            continue
        out[c] = df.groupby(s)[c].apply(lambda x: float(x.isna().mean())).to_dict()

    return out


def generate_bias_report(df: pd.DataFrame, slice_columns: List[str] = None) -> Dict[str, Any]:
    """
    Bias report at data level (no model required):
    - representation per slice
    - missingness per slice (for a sample of important columns)
    """
    logger = get_logger("breast_cancer")

    slice_columns = slice_columns or DEFAULT_SLICE_COLUMNS

    # Pick some columns that are likely to exist and matter
    candidate_cols = [
        "NCTId",
        "Conditions",
        "BriefTitle",
        "OverallStatus",
        "Phase",
        "Sex",
        "MinimumAge",
        "MaximumAge",
        "EnrollmentCount",
        "StudyType",
    ]
    sample_cols = [c for c in candidate_cols if c in df.columns]

    report: Dict[str, Any] = {
        "dataset_rows": int(len(df)),
        "slice_columns_requested": slice_columns,
        "slice_columns_found": [c for c in slice_columns if c in df.columns],
        "slices": {},
        "notes": [
            "This is data-level slicing (representation + missingness) intended for pipeline-stage bias detection.",
            "When a model is added later, per-slice model metrics (accuracy, TPR/FPR, etc.) should be attached here too.",
        ],
    }

    for col in slice_columns:
        if col not in df.columns:
            report["slices"][col] = {"error": "column_not_found"}
            continue

        rep = _value_counts_with_pct(df[col])
        missingness = _missingness_by_slice(df, col, sample_cols)

        report["slices"][col] = {
            "representation": rep,
            "missingness_by_slice": missingness,
        }

        logger.info("Bias slicing complete for column: %s", col)

    return report


def save_bias_report(report: Dict[str, Any], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path
