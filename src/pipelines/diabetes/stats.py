"""
Statistics generation for diabetes trials
Mirrors: src/pipelines/breast_cancer/stats.py
"""
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)


BASE          = "/opt/airflow/repo"
ENRICHED_PATH = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"
STATS_PATH    = f"{BASE}/data/diabetes/reports/stats.json"


def compute_stats(
    enriched_file_path: str = ENRICHED_PATH,
    stats_path: str = STATS_PATH,
) -> dict:
    """
    Compute summary statistics for diabetes trials dataset.
    Mirrors Vaishnavi's compute_stats()
    """
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    df = pd.read_csv(enriched_file_path)
    logger.info(f"Computing stats for {len(df):,} trials...")

    stats = {
        "total_trials": len(df),
        "columns": list(df.columns),
    }

    # ── Recruitment status breakdown ──
    if "Recruitment Status" in df.columns:
        stats["recruitment_status"] = df["Recruitment Status"].value_counts().to_dict()

    # ── Disease type breakdown (our new column) ──
    if "disease_type" in df.columns:
        stats["disease_type_distribution"] = df["disease_type"].value_counts().to_dict()

    # ── Sex breakdown ──
    if "Sex" in df.columns:
        stats["sex_distribution"] = df["Sex"].value_counts().to_dict()

    # ── Age groups ──
    if "Age" in df.columns:
        stats["age_distribution"] = df["Age"].astype(str).value_counts().head(10).to_dict()

    # ── Enrollment stats ──
    if "Enrollment" in df.columns:
        enrollment = pd.to_numeric(df["Enrollment"], errors="coerce").dropna()
        if len(enrollment) > 0:
            stats["enrollment"] = {
                "mean":   round(float(enrollment.mean()), 2),
                "median": round(float(enrollment.median()), 2),
                "min":    int(enrollment.min()),
                "max":    int(enrollment.max()),
                "total":  int(enrollment.sum()),
            }

    # ── Top sponsors ──
    if "Sponsor" in df.columns:
        stats["top_sponsors"] = df["Sponsor"].value_counts().head(10).to_dict()

    # ── Geographic distribution ──
    if "Locations" in df.columns:
        us_count = df["Locations"].astype(str).str.contains(
            "United States", case=False, na=False
        ).sum()
        stats["geography"] = {
            "us_trials":            int(us_count),
            "international_trials": int(len(df) - us_count),
            "us_percentage":        round(us_count / len(df) * 100, 2),
        }

    # ── Missing values per column ──
    missing = df.isnull().sum()
    stats["missing_values"] = {
        col: int(count)
        for col, count in missing[missing > 0].items()
    }

    # Save
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"✓ Stats saved → {stats_path}")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"DIABETES TRIALS STATS SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total trials     : {stats['total_trials']:,}")
    if "disease_type_distribution" in stats:
        logger.info(f"\nDisease Types:")
        for dtype, count in stats["disease_type_distribution"].items():
            logger.info(f"  {dtype:35} {count:,}")
    if "enrollment" in stats:
        logger.info(f"\nEnrollment:")
        logger.info(f"  Mean   : {stats['enrollment']['mean']:,.0f}")
        logger.info(f"  Median : {stats['enrollment']['median']:,.0f}")
        logger.info(f"  Max    : {stats['enrollment']['max']:,}")

    return stats