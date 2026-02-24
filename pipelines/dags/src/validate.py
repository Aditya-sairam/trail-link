"""
Common validation for clinical trials enriched CSV.
Paths passed as arguments.
"""
import os
import logging
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "NCT Number", "Study Title", "Recruitment Status",
    "Conditions", "Interventions", "Sponsor", "Enrollment",
    "Sex", "Location Countries", "Location Cities",
    "disease", "disease_type",
]


def run_validation(enriched_file_path: str, min_rows: int = 10) -> bool:
    """Run all validation checks. Returns True if passes."""
    if not os.path.exists(enriched_file_path):
        log.error(f"❌ File not found: {enriched_file_path}")
        return False

    df = pd.read_csv(enriched_file_path)
    log.info(f"Loaded {len(df):,} rows for validation")

    errors: List[str] = []

    # Empty check
    if len(df) == 0:
        errors.append("Dataset is empty")
    elif len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} (minimum={min_rows})")

    # Schema check
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # NCT format check
    if "NCT Number" in df.columns:
        invalid = (~df["NCT Number"].astype(str).str.match(r"^NCT\d{8}$", na=False)).sum()
        if invalid / len(df) > 0.10:
            errors.append(f"{invalid} rows have invalid NCT Number format")

    # New columns check
    if "disease" in df.columns and df["disease"].isnull().sum() > 0:
        errors.append(f"disease column has nulls")
    if "disease_type" in df.columns:
        unknown = (df["disease_type"] == "Unknown").sum()
        if unknown > len(df) * 0.5:
            errors.append(f"disease_type is Unknown for {unknown/len(df)*100:.1f}% of rows")

    # Critical nulls check
    for col in ["NCT Number", "Study Title", "Conditions"]:
        if col in df.columns:
            pct = df[col].isnull().sum() / len(df) * 100
            if pct > 80:
                errors.append(f"CRITICAL: {col} is {pct:.1f}% null")

    if errors:
        log.error(f"❌ VALIDATION FAILED — {len(errors)} error(s):")
        for err in errors:
            log.error(f"   • {err}")
        return False

    log.info(f"✅ VALIDATION PASSED — {len(df):,} trials")
    return True
