"""
Diabetes Clinical Trials Pipeline DAG
======================================
Downloads data directly from ClinicalTrials.gov API.
No local CSVs needed.

Flow:
  task_fetch_raw      ← downloads from API
        ↓
  task_enrich         ← adds disease_type + data_source columns
        ↓
  task_validate       ← ShortCircuit: stops if data invalid
        ↓
  task_quality        ← ShortCircuit: stops if data too poor
        ↓
    ┌── task_stats ────┐
    ├── task_anomaly ──┤  (parallel)
    └── task_bias ─────┘
        ↓
  task_save_reports
        ↓
  task_dvc_push
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
import subprocess

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from src.pipelines.diabetes.ingest   import download_raw_trials_csv, enrich_trials_csv
from src.pipelines.diabetes.quality  import run_quality_checks, anomalies_found
from src.pipelines.diabetes.validate import run_validation
from src.pipelines.diabetes.stats    import compute_stats
from src.pipelines.diabetes.bias     import generate_bias_report, save_bias_report

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE           = "/opt/airflow/repo"
RAW_PATH       = f"{BASE}/data/diabetes/raw/diabetes_trials_raw.csv"
ENRICHED_PATH  = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"
STATS_PATH     = f"{BASE}/data/diabetes/reports/stats.json"
ANOMALIES_PATH = f"{BASE}/data/diabetes/reports/anomalies.json"
BIAS_PATH      = f"{BASE}/data/diabetes/reports/bias_report.json"
SUMMARY_PATH   = f"{BASE}/data/diabetes/reports/pipeline_summary.json"

default_args = {
    "owner": "sanika",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# ─── Task functions ───────────────────────────────────────────────────────────

def task_fetch_raw() -> None:
    """Download ALL diabetes trials from ClinicalTrials.gov API"""
    download_raw_trials_csv(
        raw_file_path=RAW_PATH,
        status="RECRUITING",
        page_size=1000,
        condition_query="diabetes",
    )


def task_enrich() -> None:
    """Add disease_type and data_source columns"""
    enrich_trials_csv(
        raw_file_path=RAW_PATH,
        enriched_file_path=ENRICHED_PATH,
    )


def task_validate() -> bool:
    """Validate schema, NCT formats, new columns. Halts pipeline if invalid."""
    passed = run_validation(enriched_file_path=ENRICHED_PATH)
    if not passed:
        print("❌ Validation failed — halting pipeline")
    return passed


def task_quality() -> bool:
    """Clean text + deduplicate. Halts pipeline if data quality too poor."""
    stats = run_quality_checks(enriched_file_path=ENRICHED_PATH)
    return anomalies_found(stats)


def task_stats_fn(**context) -> None:
    """Compute summary statistics"""
    stats = compute_stats(
        enriched_file_path=ENRICHED_PATH,
        stats_path=STATS_PATH,
    )
    context["ti"].xcom_push(key="total_trials", value=stats.get("total_trials"))
    print(f"✓ Stats complete — {stats.get('total_trials'):,} trials")


def task_anomaly(**context) -> None:
    """Detect anomalies in cleaned data"""
    df = pd.read_csv(ENRICHED_PATH)

    anomalies = {
        "high_missing":    [],
        "outliers":        [],
        "duplicates":      [],
        "invalid_formats": [],
    }

    # High missing values >50%
    missing_pct = df.isnull().sum() / len(df) * 100
    for col, pct in missing_pct[missing_pct > 50].items():
        anomalies["high_missing"].append({"column": col, "missing_percentage": float(pct)})

    # Enrollment outliers
    if "Enrollment" in df.columns:
        enrollment = pd.to_numeric(df["Enrollment"], errors="coerce").dropna()
        if len(enrollment) > 0:
            q1, q3 = enrollment.quantile(0.25), enrollment.quantile(0.75)
            iqr = q3 - q1
            outliers = enrollment[
                (enrollment < q1 - 1.5 * iqr) | (enrollment > q3 + 1.5 * iqr)
            ]
            if len(outliers) > 0:
                anomalies["outliers"].append({
                    "field": "Enrollment",
                    "count": len(outliers),
                    "examples": outliers.head(5).tolist(),
                })

    # Duplicate NCT Numbers
    if "NCT Number" in df.columns:
        dupes = int(df["NCT Number"].duplicated().sum())
        if dupes > 0:
            anomalies["duplicates"].append({"column": "NCT Number", "count": dupes})

    # Invalid NCT format
    if "NCT Number" in df.columns:
        invalid = int((~df["NCT Number"].astype(str).str.match(r"^NCT\d{8}$", na=False)).sum())
        if invalid > 0:
            anomalies["invalid_formats"].append({"field": "NCT Number", "count": invalid})

    total_anomalies = sum(len(v) for v in anomalies.values())
    quality_score   = round(100 - (total_anomalies / len(df) * 100), 2) if len(df) > 0 else 0

    report = {
        "total_anomalies":    total_anomalies,
        "anomalies":          anomalies,
        "data_quality_score": quality_score,
    }

    os.makedirs(os.path.dirname(ANOMALIES_PATH), exist_ok=True)
    with open(ANOMALIES_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    context["ti"].xcom_push(key="data_quality_score", value=quality_score)
    print(f"✓ Anomaly detection complete — quality score: {quality_score:.1f}%")


def task_bias_fn(**context) -> None:
    """Detect bias across age, sex, geography, disease type"""
    report = generate_bias_report(enriched_file_path=ENRICHED_PATH)
    save_bias_report(report, bias_path=BIAS_PATH)
    context["ti"].xcom_push(key="bias_level", value=report["bias_level"])
    print(f"✓ Bias detection complete — level: {report['bias_level']}")


def task_save_reports(**context) -> None:
    """Consolidate all reports into pipeline summary"""
    ti = context["ti"]

    quality_score = ti.xcom_pull(task_ids="task_anomaly", key="data_quality_score")
    bias_level    = ti.xcom_pull(task_ids="task_bias",    key="bias_level")
    total_trials  = ti.xcom_pull(task_ids="task_stats",   key="total_trials")

    summary = {
        "pipeline_run_date":  datetime.now().isoformat(),
        "condition":          "diabetes",
        "total_trials":       total_trials,
        "data_quality_score": quality_score,
        "bias_level":         bias_level,
        "reports": {
            "stats":     STATS_PATH,
            "anomalies": ANOMALIES_PATH,
            "bias":      BIAS_PATH,
        },
    }

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved → {SUMMARY_PATH}")
    print(f"  Total trials  : {total_trials}")
    print(f"  Quality score : {quality_score}")
    print(f"  Bias level    : {bias_level}")


def task_dvc_push_fn() -> None:
    """Version data artifacts with DVC"""
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    if not os.path.exists(".dvc"):
        print("⚠️  DVC not initialized — skipping (non-fatal)")
        return

    remote_check = run(["dvc", "remote", "list"])
    if not remote_check.stdout.strip():
        print("⚠️  No DVC remote configured — skipping (non-fatal)")
        return

    for folder in ["data/diabetes/raw", "data/diabetes/processed", "data/diabetes/reports"]:
        if os.path.exists(folder):
            run(["dvc", "add", folder, "--no-commit"])
            print(f"  ✓ Tracked: {folder}")

    run(["dvc", "push"])
    print("✓ DVC push complete")


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="diabetes_pipeline",
    default_args=default_args,
    description="Diabetes clinical trials — fetch from API, validate, clean, stats, bias, DVC",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "clinical-trials", "diabetes", "sanika"],
) as dag:

    fetch_raw = PythonOperator(
        task_id="task_fetch_raw",
        python_callable=task_fetch_raw,
        execution_timeout=timedelta(minutes=30),
    )

    enrich = PythonOperator(
        task_id="task_enrich",
        python_callable=task_enrich,
        execution_timeout=timedelta(minutes=5),
    )

    validate = ShortCircuitOperator(
        task_id="task_validate",
        python_callable=task_validate,
        execution_timeout=timedelta(minutes=5),
    )

    quality = ShortCircuitOperator(
        task_id="task_quality",
        python_callable=task_quality,
        execution_timeout=timedelta(minutes=10),
    )

    stats = PythonOperator(
        task_id="task_stats",
        python_callable=task_stats_fn,
        execution_timeout=timedelta(minutes=5),
    )

    anomaly = PythonOperator(
        task_id="task_anomaly",
        python_callable=task_anomaly,
        execution_timeout=timedelta(minutes=5),
    )

    bias = PythonOperator(
        task_id="task_bias",
        python_callable=task_bias_fn,
        execution_timeout=timedelta(minutes=5),
    )

    save_reports = PythonOperator(
        task_id="task_save_reports",
        python_callable=task_save_reports,
        execution_timeout=timedelta(minutes=5),
    )

    dvc_push = PythonOperator(
        task_id="task_dvc_push",
        python_callable=task_dvc_push_fn,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Dependency chain ──────────────────────────────────────────────────────
    fetch_raw >> enrich >> validate >> quality >> [stats, anomaly, bias] >> save_reports >> dvc_push