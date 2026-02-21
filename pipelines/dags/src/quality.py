"""
Common quality checks for clinical trials.
Separate generate_stats() and detect_anomalies() functions.
Paths passed as arguments.
"""
import os
import re
import json
import logging
from typing import Optional, Dict, Any

import pandas as pd

log = logging.getLogger(__name__)

# Columns we do NOT want to treat as "critical" even if they are highly missing
IGNORE_HIGH_MISSING = {"Phase"}


# ─── Text cleaning ────────────────────────────────────────────────────────────

def clean_whitespace(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def remove_html_entities(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&#39;": "'",
        "&quot;": '"',
        "&nbsp;": " ",
    }
    text = str(text)
    for entity, char in entities.items():
        text = text.replace(entity, char)
    return text


def remove_invalid_characters(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text).replace("\x00", "").replace("\ufffd", "")
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    return text


def normalize_medical_terminology(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text)
    replacements = {
        r"\bt1dm?\b": "Type 1 Diabetes Mellitus",
        r"\biddm\b": "Type 1 Diabetes Mellitus",
        r"\bt2dm?\b": "Type 2 Diabetes Mellitus",
        r"\bniddm\b": "Type 2 Diabetes Mellitus",
        r"\bgdm\b": "Gestational Diabetes Mellitus",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def remove_duplicate_words(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    words = str(text).split()
    if len(words) <= 1:
        return text
    cleaned = [words[0]]
    for word in words[1:]:
        if word.lower() != cleaned[-1].lower():
            cleaned.append(word)
    return " ".join(cleaned)


def apply_all_text_cleaning(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = clean_whitespace(text)
    text = remove_html_entities(text)
    text = remove_invalid_characters(text)
    text = normalize_medical_terminology(text)
    text = remove_duplicate_words(text)
    text = clean_whitespace(text)
    return text


# ─── Stats ────────────────────────────────────────────────────────────────────

def generate_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "missing_by_col": {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        },
    }

    if "NCT Number" in df.columns:
        stats["unique_nct_numbers"] = int(df["NCT Number"].nunique())

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
                "mean": round(float(enrollment.mean()), 2),
                "median": round(float(enrollment.median()), 2),
                "min": int(enrollment.min()),
                "max": int(enrollment.max()),
            }

    return stats


# ─── Anomaly detection ────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies: Dict[str, Any] = {
        "missing_required_columns": [],
        "duplicate_nct_found": False,
        "high_missing_columns": [],
        "enrollment_outliers": [],
        "invalid_nct_format": 0,
    }

    for col in ["NCT Number", "Study Title", "Conditions", "disease", "disease_type"]:
        if col not in df.columns:
            anomalies["missing_required_columns"].append(col)

    if "NCT Number" in df.columns:
        anomalies["duplicate_nct_found"] = bool(df["NCT Number"].duplicated().any())

    for col in df.columns:
        if col in IGNORE_HIGH_MISSING:
            continue
        missing_rate = float(df[col].isna().mean())
        if missing_rate >= 0.70:
            anomalies["high_missing_columns"].append({
                "col": col,
                "missing_pct": round(missing_rate * 100, 1),
            })

    if "Enrollment" in df.columns:
        enrollment = pd.to_numeric(df["Enrollment"], errors="coerce").dropna()
        if len(enrollment) > 0:
            q1, q3 = enrollment.quantile(0.25), enrollment.quantile(0.75)
            iqr = q3 - q1
            outliers = enrollment[
                (enrollment < q1 - 1.5 * iqr) | (enrollment > q3 + 1.5 * iqr)
            ]
            if len(outliers) > 0:
                anomalies["enrollment_outliers"] = outliers.head(5).tolist()

    if "NCT Number" in df.columns:
        anomalies["invalid_nct_format"] = int(
            (~df["NCT Number"].astype(str).str.match(r"^NCT\d{8}$", na=False)).sum()
        )

    return anomalies


def anomalies_found(anomalies: Dict[str, Any]) -> bool:
    return bool(
        anomalies["missing_required_columns"]
        or anomalies["duplicate_nct_found"]
        or anomalies["high_missing_columns"]
    )


# ─── Main run ─────────────────────────────────────────────────────────────────

def run_quality_checks(
    enriched_file_path: str,
    stats_path: str,
    anomalies_path: str,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    os.makedirs(os.path.dirname(anomalies_path), exist_ok=True)

    df = pd.read_csv(enriched_file_path)
    log.info(f"Loaded {len(df):,} rows")

    text_cols = ["Study Title", "Brief Summary", "Conditions", "Interventions", "Sponsor"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(apply_all_text_cleaning)

    df.to_csv(enriched_file_path, index=False)

    stats = generate_stats(df)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info(f"✓ Stats → {stats_path}")

    anomalies = detect_anomalies(df)
    with open(anomalies_path, "w") as f:
        json.dump(anomalies, f, indent=2, default=str)
    log.info(f"✓ Anomalies → {anomalies_path}")

    if anomalies_found(anomalies):
        log.warning("⚠️ Anomalies detected")
    else:
        log.info("✓ No critical anomalies")

    return anomalies