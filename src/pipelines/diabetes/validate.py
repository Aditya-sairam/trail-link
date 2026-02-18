"""
Data validation for diabetes trials
Mirrors: src/pipelines/breast_cancer/validate.py

Validates that the enriched CSV has:
- Required columns present
- No completely empty files
- NCT Numbers in correct format
- disease_type column populated (our new column)
- data_source column populated (our new column)
"""
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


BASE          = "/opt/airflow/repo"
ENRICHED_PATH = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"

REQUIRED_COLUMNS = [
    "NCT Number",
    "Study Title",
    "Recruitment Status",
    "Conditions",
    "Brief Summary",
    "Interventions",
    "Sponsor",
    "Enrollment",
    "Age",
    "Sex",
    "Locations",
    "disease_type",    # our new column
    "data_source",     # our new column
]


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Check all required columns are present"""
    errors = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    return errors


def validate_not_empty(df: pd.DataFrame) -> list[str]:
    """Check dataframe is not empty"""
    errors = []
    if len(df) == 0:
        errors.append("Dataset is empty — no trials loaded")
    elif len(df) < 10:
        errors.append(f"Suspiciously few trials: only {len(df)} rows")
    return errors


def validate_nct_format(df: pd.DataFrame) -> list[str]:
    """Check NCT Numbers follow NCT########  format"""
    errors = []
    if "NCT Number" not in df.columns:
        return errors
    invalid = ~df["NCT Number"].astype(str).str.match(r"^NCT\d{8}$", na=False)
    count = invalid.sum()
    if count > 0:
        pct = count / len(df) * 100
        if pct > 10:
            errors.append(f"{count} ({pct:.1f}%) rows have invalid NCT Number format")
        else:
            logger.warning(f"  ⚠️ {count} rows with non-standard NCT format (acceptable)")
    return errors


def validate_new_columns(df: pd.DataFrame) -> list[str]:
    """
    Validate our two new enrichment columns are populated.
    disease_type and data_source must not be empty.
    """
    errors = []

    if "disease_type" in df.columns:
        null_count = df["disease_type"].isnull().sum()
        unknown_count = (df["disease_type"] == "Unknown").sum()
        if null_count > 0:
            errors.append(f"disease_type has {null_count} null values")
        if unknown_count > len(df) * 0.5:
            errors.append(
                f"disease_type is 'Unknown' for {unknown_count} ({unknown_count/len(df)*100:.1f}%) "
                f"trials — classification may be broken"
            )

    if "data_source" in df.columns:
        null_count = df["data_source"].isnull().sum()
        if null_count > 0:
            errors.append(f"data_source has {null_count} null values")

    return errors


def validate_critical_nulls(df: pd.DataFrame) -> list[str]:
    """Flag columns with >80% missing — these are critical failures"""
    errors = []
    critical_cols = ["NCT Number", "Study Title", "Conditions"]
    for col in critical_cols:
        if col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            if null_pct > 80:
                errors.append(f"CRITICAL: {col} is {null_pct:.1f}% null")
    return errors


def run_validation(enriched_file_path: str = ENRICHED_PATH) -> bool:
    """
    Run all validation checks.
    Returns True if data passes, False if critical errors found.
    Mirrors Vaishnavi's run_validation()
    """
    logger.info("=" * 60)
    logger.info("VALIDATION - Diabetes Trials")
    logger.info("=" * 60)

    # Check file exists
    if not os.path.exists(enriched_file_path):
        logger.error(f"❌ File not found: {enriched_file_path}")
        logger.error("   Run ingest step first.")
        return False

    df = pd.read_csv(enriched_file_path)
    logger.info(f"  Loaded {len(df):,} rows for validation")

    all_errors = []

    # Run all checks
    all_errors += validate_not_empty(df)
    all_errors += validate_schema(df)
    all_errors += validate_nct_format(df)
    all_errors += validate_new_columns(df)
    all_errors += validate_critical_nulls(df)

    # Report
    if all_errors:
        logger.error(f"\n❌ VALIDATION FAILED — {len(all_errors)} error(s):")
        for err in all_errors:
            logger.error(f"   • {err}")
        return False
    else:
        logger.info(f"\n✅ VALIDATION PASSED")
        logger.info(f"   {len(df):,} trials — all checks passed")
        return True