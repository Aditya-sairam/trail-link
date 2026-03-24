"""
Common bias detection for clinical trials.
Takes df as parameter — testable and reusable.
Works for any condition.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


def _value_counts_with_pct(series: pd.Series) -> Dict[str, Any]:
    s      = series.fillna("MISSING").astype(str)
    counts = s.value_counts(dropna=False).to_dict()
    pct    = s.value_counts(normalize=True, dropna=False).to_dict()
    return {"counts": counts, "pct": pct}


def generate_bias_report(
    df: pd.DataFrame,
    slice_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate bias report through data slicing.
    Takes df as parameter — not a file path.
    Works for any condition.
    """
    slice_columns = slice_columns or ["Sex", "disease_type", "Location Countries"]
    total = len(df)

    report: Dict[str, Any] = {
        "total_trials":            total,
        "disease":                 df["disease"].iloc[0] if "disease" in df.columns and len(df) > 0 else "",
        "slices":                  {},
        "warnings":                [],
    }

    # ── Age bias ──
    age_col = next((c for c in ["Age", "stdAges"] if c in df.columns), None)
    if age_col:
        pediatric = int(df[age_col].astype(str).str.contains("Child|Pediatric", case=False, na=False).sum())
        adult     = int(df[age_col].astype(str).str.contains("Adult", case=False, na=False).sum())
        elderly   = int(df[age_col].astype(str).str.contains("Older Adult|65|Elder", case=False, na=False).sum())
        report["slices"]["age"] = {"pediatric": pediatric, "adult": adult, "elderly": elderly}
        if pediatric < total * 0.05:
            msg = f"LOW pediatric representation: {pediatric} ({pediatric/total*100:.1f}%)"
            log.warning(f"⚠️ {msg}")
            report["warnings"].append(msg)

    # ── Sex bias (exclude gestational diabetes) ──
    if "Sex" in df.columns:
        disease = df["disease"].iloc[0] if "disease" in df.columns and len(df) > 0 else ""
        df_sex  = df[df["disease_type"] != "Gestational Diabetes"] \
            if disease == "diabetes" and "disease_type" in df.columns else df

        report["slices"]["Sex"] = _value_counts_with_pct(df_sex["Sex"])
        counts = report["slices"]["Sex"]["counts"]
        male   = sum(v for k, v in counts.items() if "male" in str(k).lower() and "fe" not in str(k).lower())
        female = sum(v for k, v in counts.items() if "female" in str(k).lower())
        if male > 0 and female > 0:
            ratio = male / female
            if ratio > 2 or ratio < 0.5:
                msg = f"SEX IMBALANCE: Male={male}, Female={female} (ratio: {ratio:.2f})"
                log.warning(f"⚠️ {msg}")
                report["warnings"].append(msg)

    # ── Disease type distribution ──
    if "disease_type" in df.columns:
        report["slices"]["disease_type"] = _value_counts_with_pct(df["disease_type"])

    # ── Geographic bias — uses Location Countries column ──
    if "Location Countries" in df.columns:
        us_trials = int(df["Location Countries"].astype(str).str.contains(
            "United States", case=False, na=False
        ).sum())
        us_pct = round((us_trials / total * 100), 2) if total > 0 else 0
        report["slices"]["geography"] = {
            "us_trials":            us_trials,
            "international_trials": int(total - us_trials),
            "us_percentage":        us_pct,
        }
        if us_pct > 80:
            msg = f"GEOGRAPHIC BIAS: {us_pct:.1f}% US-only trials"
            log.warning(f"⚠️ {msg}")
            report["warnings"].append(msg)

    bias_score             = len(report["warnings"])
    report["bias_score"]   = bias_score
    report["bias_level"]   = "HIGH" if bias_score >= 2 else "MEDIUM" if bias_score == 1 else "LOW"

    log.info(f"Bias Score: {bias_score} | Level: {report['bias_level']}")
    return report


def save_bias_report(report: Dict[str, Any], bias_path: str) -> None:
    os.makedirs(os.path.dirname(bias_path), exist_ok=True)
    with open(bias_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"✓ Bias report → {bias_path}")