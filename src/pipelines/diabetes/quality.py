"""
Diabetes data quality checks + cleaning
Mirrors: src/pipelines/breast_cancer/quality.py
"""
import pandas as pd
import re
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BASE           = "/opt/airflow/repo"
ENRICHED_PATH  = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"
STATS_PATH     = f"{BASE}/data/diabetes/reports/stats.json"


# ─── TEXT CLEANING ────────────────────────────────────────────────────────────

def clean_whitespace(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\t', ' ')
    return text if text else None


def remove_html_entities(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    html_entities = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&#39;': "'", '&quot;': '"', '&nbsp;': ' ', '&apos;': "'",
    }
    text = str(text)
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    return text


def remove_invalid_characters(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text)
    text = text.replace('\x00', '').replace('\ufffd', '')
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text


def normalize_medical_terminology(text: Optional[str]) -> Optional[str]:
    if pd.isna(text) or text is None:
        return None
    text = str(text)
    terminology_map = {
        r'\bt1d\b': 'Type 1 Diabetes Mellitus',
        r'\bt1dm\b': 'Type 1 Diabetes Mellitus',
        r'\btype\s*i\s+diabetes\b': 'Type 1 Diabetes Mellitus',
        r'\biddm\b': 'Type 1 Diabetes Mellitus',
        r'\bjuvenile\s+diabetes\b': 'Type 1 Diabetes Mellitus',
        r'\bt2d\b': 'Type 2 Diabetes Mellitus',
        r'\bt2dm\b': 'Type 2 Diabetes Mellitus',
        r'\btype\s*ii\s+diabetes\b': 'Type 2 Diabetes Mellitus',
        r'\bniddm\b': 'Type 2 Diabetes Mellitus',
        r'\bgdm\b': 'Gestational Diabetes Mellitus',
    }
    for pattern, replacement in terminology_map.items():
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
    return ' '.join(cleaned)


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


# ─── DEDUPLICATION ───────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame, column: str = "NCT Number") -> pd.DataFrame:
    """Keep most complete row for each NCT Number"""
    initial = len(df)
    df["_score"] = df.count(axis=1)
    df_sorted = df.sort_values([column, "_score"], ascending=[True, False])
    df_clean = df_sorted.drop_duplicates(subset=[column], keep="first").copy()
    df_clean = df_clean.drop(columns=["_score"])
    logger.info(f"  Deduplication: {initial:,} → {len(df_clean):,} rows (removed {initial - len(df_clean):,})")
    return df_clean


# ─── MAIN QUALITY FUNCTIONS ───────────────────────────────────────────────────

def run_quality_checks(enriched_file_path: str = ENRICHED_PATH) -> dict:
    """
    Run all quality checks and clean the data.
    Mirrors Vaishnavi's run_quality_checks()
    Returns stats dict.
    """
    # ── Always create reports folder using absolute path ──
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)

    logger.info("Running quality checks on diabetes trials...")
    df = pd.read_csv(enriched_file_path)
    logger.info(f"  Loaded {len(df):,} rows")

    stats = {
        "total_rows": len(df),
        "issues": {}
    }

    text_cols = ["Study Title", "Conditions", "Brief Summary",
                 "Interventions", "Sponsor", "Locations"]

    # Count issues before cleaning
    extra_whitespace = 0
    null_values = 0

    for col in text_cols:
        if col not in df.columns:
            continue
        null_values += int(df[col].isnull().sum())
        for val in df[col].dropna():
            if re.search(r'\s{2,}', str(val)):
                extra_whitespace += 1

    stats["issues"] = {
        "null_values": null_values,
        "extra_whitespace": extra_whitespace,
    }

    # Clean all text columns
    logger.info("  Cleaning text fields...")
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(apply_all_text_cleaning)

    # Deduplicate
    logger.info("  Deduplicating...")
    df = remove_duplicates(df)

    # Save cleaned file back
    df.to_csv(enriched_file_path, index=False)

    stats["rows_after_cleaning"] = len(df)
    stats["duplicates_removed"] = stats["total_rows"] - len(df)

    # ── Save stats — folder guaranteed to exist from makedirs above ──
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"✓ Quality checks complete → {STATS_PATH}")
    return stats


def anomalies_found(stats: dict) -> bool:
    """
    Returns True if data quality is acceptable → pipeline continues.
    Returns False → ShortCircuitOperator skips downstream tasks.
    Mirrors Vaishnavi's anomalies_found()
    """
    issues = stats.get("issues", {})
    null_count = issues.get("null_values", 0)
    total = stats.get("total_rows", 1)

    null_pct = (null_count / total) * 100 if total > 0 else 0

    if null_pct > 80:
        logger.warning(f"  ⚠️ Too many nulls ({null_pct:.1f}%) — skipping downstream tasks")
        return False

    logger.info(f"  ✓ Data quality acceptable ({null_pct:.1f}% nulls) — continuing pipeline")
    return True