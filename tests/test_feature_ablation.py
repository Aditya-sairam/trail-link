"""
Feature Ablation Study - TrialLink RAG Pipeline
=================================================
Measures which patient summary features matter most for trial matching.

Method:
    For each patient, run the pipeline with the full summary (baseline),
    then re-run with one feature masked at a time. Compare retrieved trials
    to measure feature importance via overlap drop.

Features tested:
    - diagnosis    : primary condition (e.g., "Type 2 Diabetes", "HER2-positive breast cancer")
    - age          : patient age
    - medications  : current medications
    - lab_values   : HbA1c, BMI, blood pressure, etc.
    - smoking      : smoking status

Metrics per ablation:
    - overlap_ratio : % of baseline trials still retrieved after masking
    - rank_shift    : avg position change of overlapping trials
    - importance    : 1 - overlap_ratio (higher = feature matters more)

All runs logged to MLflow experiment: triallink-feature-ablation
"""

import os
import sys
import re
import json
import logging
import mlflow
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vertexai
import models.rag_service as rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
GCP_REGION   = os.getenv("GCP_REGION", "us-central1")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "test_results", "ablation")

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/degaonkar.sw@northeastern.edu/triallink-feature-ablation"
)

# ── Feature masking patterns ───────────────────────────────────────────────────
# Each entry: (feature_name, regex pattern to remove from summary)
FEATURE_MASKS = {
    "diagnosis": [
        r"Active diagnoses:\s*[^.]+\.",
    ],
    "age": [
        r"\d+-year-old",
    ],
    "medications": [
        r"Current medications:\s*[^.]+\.",
    ],
    "lab_values": [
        r"Recent observations:\s*[^.]+\.",
        r"HbA1c:\s*[\d.]+%[;,]?\s*",
        r"BMI:\s*[\d.]+[;,]?\s*",
        r"BP\s*[\d/]+\s*mmHg[;,]?\s*",
        r"eGFR\s*[\d.]+\s*mL/min[;,]?\s*",
        r"FEV1:\s*[\d.]+%\s*predicted[;,]?\s*",
        r"MMSE\s*score\s*[\d/]+[;,]?\s*",
        r"fasting glucose:\s*[\d.]+\s*mg/dL[;,]?\s*",
        r"LVEF\s*\w+[;,.]?\s*",
        r"AHI:\s*[\d.]+\s*events/hour[;,]?\s*",
    ],
    "smoking": [
        r"Smoking status:\s*[^.]+\.",
    ],
}

# ── Test patients (subset: 2 per condition for speed) ──────────────────────────
TEST_PATIENTS = [
    {
        "patient_id": "test_diabetes_001",
        "slice": "diabetes",
        "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."
    },
    {
        "patient_id": "test_diabetes_002",
        "slice": "diabetes",
        "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."
    },
    {
        "patient_id": "test_breast_cancer_001",
        "slice": "breast_cancer",
        "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."
    },
    {
        "patient_id": "test_breast_cancer_002",
        "slice": "breast_cancer",
        "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."
    },
    {
        "patient_id": "test_ood_copd_001",
        "slice": "ood",
        "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."
    },
    {
        "patient_id": "test_ood_obesity_001",
        "slice": "ood",
        "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."
    },
]


def mask_feature(summary: str, feature_name: str) -> str:
    """Remove a feature from the patient summary using regex patterns."""
    masked = summary
    for pattern in FEATURE_MASKS[feature_name]:
        masked = re.sub(pattern, "", masked, flags=re.IGNORECASE)
    # Clean up extra whitespace
    masked = re.sub(r"\s{2,}", " ", masked).strip()
    return masked


def get_trial_ids(result: dict) -> list[str]:
    """Extract ordered list of NCT IDs from pipeline result."""
    return [
        t.get("nct_number", "")
        for t in result.get("retrieved_trials", [])
    ]


def compute_overlap(baseline_ids: list[str], ablated_ids: list[str]) -> dict:
    """
    Compare ablated retrieval against baseline.
    Returns overlap ratio, rank shift, and importance score.
    """
    if not baseline_ids:
        return {"overlap_ratio": 0.0, "avg_rank_shift": 0.0, "importance": 1.0}

    baseline_set = set(baseline_ids)
    ablated_set  = set(ablated_ids)
    overlap      = baseline_set & ablated_set
    overlap_ratio = len(overlap) / len(baseline_set) if baseline_set else 0.0

    # Compute avg rank shift for overlapping trials
    rank_shifts = []
    baseline_rank = {nct: i for i, nct in enumerate(baseline_ids)}
    ablated_rank  = {nct: i for i, nct in enumerate(ablated_ids)}
    for nct in overlap:
        if nct in ablated_rank:
            rank_shifts.append(abs(baseline_rank[nct] - ablated_rank[nct]))

    avg_rank_shift = round(sum(rank_shifts) / max(len(rank_shifts), 1), 2)
    importance     = round(1 - overlap_ratio, 3)

    return {
        "overlap_ratio" : round(overlap_ratio, 3),
        "avg_rank_shift": avg_rank_shift,
        "importance"     : importance,
    }


def reinit_vertexai():
    """Re-initialize Vertex AI to correct project after MedGemma switches it."""
    vertexai.init(project=GCP_PROJECT, location=GCP_REGION)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    all_results      = []
    feature_importance_agg = {f: [] for f in FEATURE_MASKS}

    with mlflow.start_run(run_name=f"ablation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):

        mlflow.log_params({
            "features_tested"  : list(FEATURE_MASKS.keys()),
            "total_patients"   : len(TEST_PATIENTS),
            "total_runs"       : len(TEST_PATIENTS) * (1 + len(FEATURE_MASKS)),
        })

        for patient in TEST_PATIENTS:
            pid     = patient["patient_id"]
            summary = patient["summary"]
            slice_  = patient["slice"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Patient: {pid} [{slice_}]")
            logger.info(f"{'='*60}")

            # ── Baseline run (full summary) ───────────────────────────────
            logger.info("  Running baseline (full summary)...")
            try:
                baseline_result = rag_service.rag_pipeline(summary)
                baseline_ids    = get_trial_ids(baseline_result)
                reinit_vertexai()
                logger.info(f"  Baseline: {len(baseline_ids)} trials retrieved")
            except Exception as e:
                logger.error(f"  Baseline FAILED for {pid}: {e}")
                reinit_vertexai()
                continue

            patient_results = {
                "patient_id"    : pid,
                "slice"         : slice_,
                "baseline_trials": baseline_ids,
                "ablations"     : {},
            }

            # ── Ablation runs (one feature removed at a time) ─────────────
            for feature_name in FEATURE_MASKS:
                masked_summary = mask_feature(summary, feature_name)

                # Skip if masking didn't change anything
                if masked_summary.strip() == summary.strip():
                    logger.info(f"  {feature_name}: no change after masking, skipping")
                    patient_results["ablations"][feature_name] = {
                        "skipped": True,
                        "reason": "feature not present in summary",
                    }
                    continue

                logger.info(f"  Ablating: {feature_name}")
                logger.info(f"    Masked summary: {masked_summary[:100]}...")

                try:
                    ablated_result = rag_service.rag_pipeline(masked_summary)
                    ablated_ids    = get_trial_ids(ablated_result)
                    reinit_vertexai()

                    metrics = compute_overlap(baseline_ids, ablated_ids)

                    patient_results["ablations"][feature_name] = {
                        "masked_summary"  : masked_summary,
                        "retrieved_trials": ablated_ids,
                        **metrics,
                    }

                    feature_importance_agg[feature_name].append(metrics["importance"])

                    mlflow.log_metrics({
                        f"{pid}_{feature_name}_overlap"    : metrics["overlap_ratio"],
                        f"{pid}_{feature_name}_rank_shift" : metrics["avg_rank_shift"],
                        f"{pid}_{feature_name}_importance" : metrics["importance"],
                    })

                    logger.info(f"    overlap={metrics['overlap_ratio']} | importance={metrics['importance']}")

                except Exception as e:
                    logger.error(f"    FAILED for {pid}/{feature_name}: {e}")
                    reinit_vertexai()
                    patient_results["ablations"][feature_name] = {
                        "error": str(e), "status": "failed"
                    }

            all_results.append(patient_results)

        # ── Compute aggregate feature importance ──────────────────────────
        avg_importance = {}
        for feature, scores in feature_importance_agg.items():
            if scores:
                avg = round(sum(scores) / len(scores), 3)
            else:
                avg = 0.0
            avg_importance[feature] = avg
            mlflow.log_metric(f"avg_importance_{feature}", avg)

        # ── Sort by importance (most important first) ─────────────────────
        ranked_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

        # ── Save artifacts ────────────────────────────────────────────────
        save_json = lambda data, fp: (
            os.makedirs(os.path.dirname(fp), exist_ok=True),
            open(fp, "w").write(json.dumps(data, indent=2, default=str))
        )

        ablation_summary = {
            "feature_importance_ranking": [
                {"rank": i+1, "feature": f, "avg_importance": imp}
                for i, (f, imp) in enumerate(ranked_features)
            ],
            "per_patient_results": all_results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        summary_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
        save_json(ablation_summary, summary_path)
        mlflow.log_artifacts(RESULTS_DIR, artifact_path="ablation")

        # ── Print results ─────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE RANKING (via ablation)")
        print("=" * 60)
        print(f"{'Rank':<6} {'Feature':<15} {'Avg Importance':>15}")
        print("-" * 40)
        for i, (feature, imp) in enumerate(ranked_features):
            bar = "█" * int(imp * 20)
            print(f"  {i+1:<4} {feature:<15} {imp:>13.3f}  {bar}")

        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")