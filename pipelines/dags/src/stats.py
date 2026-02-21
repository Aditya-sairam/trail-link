"""
Common statistics generation for clinical trials.
Separate file — single responsibility.
Paths passed as arguments.
"""
import os
import json
import logging
from typing import Dict, Any

import pandas as pd

log = logging.getLogger(__name__)


def compute_stats(enriched_file_path: str, stats_path: str) -> Dict[str, Any]:
    """Compute and save summary statistics for any condition."""
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    df = pd.read_csv(enriched_file_path)
    log.info(f"Computing stats for {len(df):,} trials...")

    stats: Dict[str, Any] = {
        "total_trials":  len(df),
        "total_columns": len(df.columns),
        "columns":       df.columns.tolist(),
    }

    if "disease" in df.columns and len(df) > 0:
        stats["disease"] = df["disease"].iloc[0]

    if "disease_type" in df.columns:
        stats["disease_type_distribution"] = df["disease_type"].value_counts().to_dict()

    if "Recruitment Status" in df.columns:
        stats["recruitment_status"] = df["Recruitment Status"].value_counts().to_dict()

    if "Sex" in df.columns:
        stats["sex_distribution"] = df["Sex"].value_counts().to_dict()

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

    if "Sponsor" in df.columns:
        stats["top_sponsors"] = df["Sponsor"].value_counts().head(10).to_dict()

    if "Location Countries" in df.columns:
        us_count = df["Location Countries"].astype(str).str.contains(
            "United States", case=False, na=False
        ).sum()
        stats["geography"] = {
            "us_trials":            int(us_count),
            "international_trials": int(len(df) - us_count),
            "us_percentage":        round(us_count / len(df) * 100, 2),
            "top_countries":        _top_countries(df),
        }

    missing = df.isnull().sum()
    stats["missing_values"] = {col: int(v) for col, v in missing[missing > 0].items()}

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    log.info(f"✓ Stats → {stats_path}")
    return stats


def _top_countries(df: pd.DataFrame, n: int = 10) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cell in df["Location Countries"].dropna():
        for country in str(cell).split(";"):
            country = country.strip()
            if country:
                counts[country] = counts.get(country, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n])