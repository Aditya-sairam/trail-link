"""
Bias detection for diabetes trials
Mirrors: src/pipelines/breast_cancer/bias.py
Uses Sanika's existing detect_bias logic
"""
import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)


BASE           = "/opt/airflow/repo"
ENRICHED_PATH  = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"
BIAS_PATH      = f"{BASE}/data/diabetes/reports/bias_report.json"



def generate_bias_report(enriched_file_path: str = ENRICHED_PATH) -> dict:
    """
    Detect bias through data slicing across age, sex, disease type, geography.
    Mirrors Vaishnavi's generate_bias_report()
    """
    logger.info("=" * 60)
    logger.info("BIAS DETECTION - Diabetes Trials")
    logger.info("=" * 60)

    df = pd.read_csv(enriched_file_path)
    total = len(df)

    bias_report = {
        "total_trials": total,
        "slices": {},
        "warnings": []
    }

    # ── 1. Age bias ──
    logger.info("\n[1/4] Age Bias...")
    if "Age" in df.columns:
        age_slices = {
            "pediatric":  df["Age"].astype(str).str.contains("Child|Pediatric", case=False, na=False).sum(),
            "adult":      df["Age"].astype(str).str.contains("Adult|18", case=False, na=False).sum(),
            "elderly":    df["Age"].astype(str).str.contains("65|Elder|Senior|Older", case=False, na=False).sum(),
            "all_ages":   df["Age"].astype(str).str.contains("All", case=False, na=False).sum(),
        }
        bias_report["slices"]["age"] = {k: int(v) for k, v in age_slices.items()}

        if age_slices["pediatric"] < total * 0.05:
            msg = f"LOW pediatric representation: {age_slices['pediatric']} trials ({age_slices['pediatric']/total*100:.1f}%)"
            logger.warning(f"  ⚠️ {msg}")
            bias_report["warnings"].append(msg)
        else:
            logger.info(f"  ✓ Pediatric: {age_slices['pediatric']} ({age_slices['pediatric']/total*100:.1f}%)")

    # ── 2. Sex bias ──
    logger.info("\n[2/4] Sex Bias...")
    if "Sex" in df.columns:
        sex_counts = df["Sex"].value_counts().to_dict()
        bias_report["slices"]["sex"] = {k: int(v) for k, v in sex_counts.items()}

        male   = sum(v for k, v in sex_counts.items() if "male" in str(k).lower() and "fe" not in str(k).lower())
        female = sum(v for k, v in sex_counts.items() if "female" in str(k).lower())

        if male > 0 and female > 0:
            ratio = male / female
            if ratio > 2 or ratio < 0.5:
                msg = f"SEX IMBALANCE: Male={male}, Female={female} (ratio: {ratio:.2f})"
                logger.warning(f"  ⚠️ {msg}")
                bias_report["warnings"].append(msg)
            else:
                logger.info(f"  ✓ Balanced: Male={male}, Female={female}")

    # ── 3. Disease type distribution ──
    logger.info("\n[3/4] Disease Type Distribution...")
    if "disease_type" in df.columns:
        type_dist = df["disease_type"].value_counts().to_dict()
        bias_report["slices"]["disease_type"] = {k: int(v) for k, v in type_dist.items()}
        for dtype, count in list(type_dist.items())[:5]:
            logger.info(f"  {dtype:35} {count:5,} ({count/total*100:.1f}%)")

    # ── 4. Geographic bias ──
    logger.info("\n[4/4] Geographic Bias...")
    if "Locations" in df.columns:
        us_trials = df["Locations"].astype(str).str.contains("United States", case=False, na=False).sum()
        us_pct = (us_trials / total * 100) if total > 0 else 0

        bias_report["slices"]["geography"] = {
            "us_trials": int(us_trials),
            "international_trials": int(total - us_trials),
            "us_percentage": round(us_pct, 2),
        }

        if us_pct > 80:
            msg = f"GEOGRAPHIC BIAS: {us_pct:.1f}% US-only trials"
            logger.warning(f"  ⚠️ {msg}")
            bias_report["warnings"].append(msg)
        else:
            logger.info(f"  ✓ Geographic balance: {us_pct:.1f}% US / {100-us_pct:.1f}% International")

    # ── Overall bias score ──
    bias_score = len(bias_report["warnings"])
    bias_report["overall_bias_score"] = bias_score
    bias_report["bias_level"] = "HIGH" if bias_score >= 2 else "MEDIUM" if bias_score == 1 else "LOW"

    logger.info(f"\nBias Score: {bias_score} | Level: {bias_report['bias_level']}")
    return bias_report


def save_bias_report(bias_report: dict, bias_path: str = BIAS_PATH) -> None:
    """
    Save bias report to JSON.
    Mirrors Vaishnavi's save_bias_report()
    """
    os.makedirs(os.path.dirname(bias_path), exist_ok=True)
    with open(bias_path, "w") as f:
        json.dump(bias_report, f, indent=2, default=str)
    logger.info(f"✓ Bias report saved → {bias_path}")