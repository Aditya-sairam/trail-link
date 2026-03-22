# test_rag_pipeline.py
"""
End-to-end test with MLflow tracking:
  1. Run RAG pipeline for 13 diverse test patients
  2. Save structured results to test_results/
  3. Log all metrics and artifacts to MLflow (Databricks)

MLflow logs per run:
  - params  : model config, top-k values, patient counts
  - metrics : candidates retrieved, trials matched, success rate per slice
  - artifacts: rag_output.json, rag_summary.json, per-patient JSONs

Output files:
  test_results/rag_output.json
  test_results/rag_summary.json
  test_results/patients/<patient_id>.json
"""

import sys
import os
import json
import logging
import mlflow
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google.cloud import firestore
from models.rag_service import rag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
PATIENT_DB     = "patient-db-dev"
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "test_results")

# MLflow config
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/degaonkar.sw@northeastern.edu/triallink-rag-pipeline"
)


# ── Patient slices — used for bias metrics ─────────────────────────────────────
TEST_PATIENTS = [
    # ── DIABETES ──────────────────────────────────────────────────────────────
    {
        "patient_id"  : "test_diabetes_001",
        "slice_condition": "diabetes",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."
    },
    {
        "patient_id"  : "test_diabetes_002",
        "slice_condition": "diabetes",
        "slice_age_group": "elderly",
        "slice_sex"      : "male",
        "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."
    },
    {
        "patient_id"  : "test_diabetes_003",
        "slice_condition": "diabetes",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 55-year-old female. Active diagnoses: Type 2 Diabetes, Obesity. Current medications: Metformin 1000mg, Empagliflozin. Recent observations: HbA1c: 7.8%; BMI: 35; fasting glucose: 145 mg/dL. No insulin therapy. Smoking status: Never smoker."
    },
    {
        "patient_id"  : "test_diabetes_004",
        "slice_condition": "diabetes",
        "slice_age_group": "adult",
        "slice_sex"      : "male",
        "summary": "Patient is a 38-year-old male. Active diagnoses: Type 1 Diabetes. Current medications: Insulin pump therapy. Recent observations: HbA1c: 7.5%; CGM in use. No oral antidiabetics. Smoking status: Never smoker."
    },
    {
        "patient_id"  : "test_prediabetes_001",
        "slice_condition": "diabetes",
        "slice_age_group": "adult",
        "slice_sex"      : "male",
        "summary": "Patient is a 50-year-old male. Active diagnoses: Prediabetes, Metabolic Syndrome. Current medications: None. Recent observations: HbA1c: 6.1%; BMI: 29; fasting glucose: 112 mg/dL; triglycerides elevated. Smoking status: Current smoker."
    },

    # ── BREAST CANCER ─────────────────────────────────────────────────────────
    {
        "patient_id"  : "test_breast_cancer_001",
        "slice_condition": "breast_cancer",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."
    },
    {
        "patient_id"  : "test_breast_cancer_002",
        "slice_condition": "breast_cancer",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."
    },
    {
        "patient_id"  : "test_breast_cancer_003",
        "slice_condition": "breast_cancer",
        "slice_age_group": "elderly",
        "slice_sex"      : "female",
        "summary": "Patient is a 61-year-old female. Active diagnoses: HR-positive HER2-negative breast cancer stage II. Post-menopausal. Current medications: Letrozole (aromatase inhibitor). No prior CDK4/6 inhibitor therapy. ECOG performance status 0."
    },
    {
        "patient_id"  : "test_breast_cancer_004",
        "slice_condition": "breast_cancer",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 48-year-old female. Active diagnoses: Metastatic breast cancer, HER2-positive. Prior treatments: Trastuzumab, Pertuzumab. Recent observations: LVEF normal. Currently on Capecitabine. ECOG performance status 1."
    },

    # ── OUT OF DISTRIBUTION ───────────────────────────────────────────────────
    {
        "patient_id"  : "test_ood_copd_001",
        "slice_condition": "ood",
        "slice_age_group": "elderly",
        "slice_sex"      : "male",
        "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."
    },
    {
        "patient_id"  : "test_ood_alzheimers_001",
        "slice_condition": "ood",
        "slice_age_group": "elderly",
        "slice_sex"      : "female",
        "summary": "Patient is a 74-year-old female. Active diagnoses: Early-stage Alzheimer's disease. Current medications: Donepezil 10mg. Recent observations: MMSE score 22/30; MRI shows hippocampal atrophy. No cardiovascular disease. Smoking status: Never smoker."
    },
    {
        "patient_id"  : "test_ood_hypertension_001",
        "slice_condition": "ood",
        "slice_age_group": "adult",
        "slice_sex"      : "male",
        "summary": "Patient is a 58-year-old male. Active diagnoses: Resistant Hypertension, Chronic Kidney Disease stage 3. Current medications: Amlodipine, Losartan, Hydrochlorothiazide. Recent observations: BP 158/96 mmHg; eGFR 42 mL/min. Smoking status: Never smoker."
    },
    {
        "patient_id"  : "test_ood_obesity_001",
        "slice_condition": "ood",
        "slice_age_group": "adult",
        "slice_sex"      : "female",
        "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_trial_result(trial: dict) -> dict:
    return {
        "nct_number"   : trial.get("nct_number", "N/A"),
        "title"        : trial.get("study_title") or trial.get("title", "N/A"),
        "condition"    : trial.get("conditions", "N/A"),
        "disease"      : trial.get("disease", "N/A"),
        "phase"        : trial.get("phase", "N/A"),
        "status"       : trial.get("recruitment_status", "N/A"),
        "eligibility"  : trial.get("eligibility_criteria", "N/A"),
        "min_age"      : trial.get("min_age", "N/A"),
        "max_age"      : trial.get("max_age", "N/A"),
        "sex"          : trial.get("sex", "N/A"),
        "interventions": trial.get("interventions", "N/A"),
        "url"          : trial.get("study_url", "N/A"),
    }


def save_json(data, filepath: str):
    """Write JSON to disk for local output and MLflow artifact logging."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  Saved: {filepath}")


def check_validation_threshold(success_rate: float, avg_matched: float):
    """
    Validation threshold check.
    Fails CI if core metrics drop below minimum acceptable values.
    """
    THRESHOLD_SUCCESS_RATE = 0.7
    THRESHOLD_AVG_MATCHED  = 3.0

    errors = []
    if success_rate < THRESHOLD_SUCCESS_RATE:
        errors.append(f"success_rate={success_rate} below threshold={THRESHOLD_SUCCESS_RATE}")
    if avg_matched < THRESHOLD_AVG_MATCHED:
        errors.append(f"avg_matched={avg_matched} below threshold={THRESHOLD_AVG_MATCHED}")

    if errors:
        raise ValueError(f"Validation failed: {'; '.join(errors)}")

    logger.info(f"Validation passed: success_rate={success_rate}, avg_matched={avg_matched}")


def check_bias_alert(slice_metrics: dict, avg_matched: float):
    """
    Bias alert check.
    Flags if any demographic slice performs significantly worse than overall avg.
    Fails CI if difference > 2.0 trials matched.
    """
    BIAS_THRESHOLD = 2.0
    warnings       = []
    errors         = []

    for metric_name, slice_avg in slice_metrics.items():
        diff = avg_matched - slice_avg
        if diff > BIAS_THRESHOLD:
            errors.append(f"{metric_name}={slice_avg} is {diff:.1f} below overall avg={avg_matched}")
        elif diff > 1.0:
            warnings.append(f"{metric_name}={slice_avg} is {diff:.1f} below overall avg={avg_matched}")

    for w in warnings:
        logger.warning(f"Bias warning: {w}")

    if errors:
        raise ValueError(f"Bias check failed — significant disparity detected: {'; '.join(errors)}")

    logger.info("Bias check passed — no significant disparities detected across slices")


def check_rollback(current_success_rate: float, current_avg_matched: float):
    """
    Rollback check.
    Compares current run metrics against previous MLflow run.
    Falls back to hardcoded baseline if no previous run exists.
    Uses 10% degradation tolerance.
    """
    BASELINE_SUCCESS_RATE  = 0.7
    BASELINE_AVG_MATCHED   = 3.0
    DEGRADATION_TOLERANCE  = 0.1

    try:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_NAME],
            order_by=["start_time DESC"],
            max_results=2,
            filter_string="status = 'FINISHED'"
        )

        if len(runs) < 2:
            logger.info("No previous run found — using hardcoded baseline for rollback check")
            prev_success_rate = BASELINE_SUCCESS_RATE
            prev_avg_matched  = BASELINE_AVG_MATCHED
        else:
            prev_run          = runs.iloc[1]
            prev_success_rate = prev_run.get("metrics.success_rate", BASELINE_SUCCESS_RATE)
            prev_avg_matched  = prev_run.get("metrics.avg_trials_matched", BASELINE_AVG_MATCHED)
            logger.info(f"Previous run metrics: success_rate={prev_success_rate}, avg_matched={prev_avg_matched}")

    except Exception as e:
        logger.warning(f"Could not query previous run: {e}. Using hardcoded baseline.")
        prev_success_rate = BASELINE_SUCCESS_RATE
        prev_avg_matched  = BASELINE_AVG_MATCHED

    errors = []
    if current_success_rate < prev_success_rate * (1 - DEGRADATION_TOLERANCE):
        errors.append(
            f"success_rate dropped: {current_success_rate} vs previous {prev_success_rate} "
            f"(tolerance={DEGRADATION_TOLERANCE*100:.0f}%)"
        )
    if current_avg_matched < prev_avg_matched * (1 - DEGRADATION_TOLERANCE):
        errors.append(
            f"avg_matched dropped: {current_avg_matched} vs previous {prev_avg_matched} "
            f"(tolerance={DEGRADATION_TOLERANCE*100:.0f}%)"
        )

    if errors:
        raise ValueError(f"Rollback triggered — performance degradation detected: {'; '.join(errors)}")

    logger.info("Rollback check passed — no performance degradation detected")


def compute_slice_metrics(results: list[dict]) -> dict:
    """
    Compute success rate and avg trials matched per slice.
    Slices: condition, age_group, sex.
    """
    slices = {
        "condition" : {},
        "age_group" : {},
        "sex"       : {},
    }

    for r in results:
        if r.get("status") != "success":
            continue

        matched = r.get("reranked_count", 0)

        for slice_key, slice_val in [
            ("condition", r.get("slice_condition")),
            ("age_group", r.get("slice_age_group")),
            ("sex",       r.get("slice_sex")),
        ]:
            if slice_val not in slices[slice_key]:
                slices[slice_key][slice_val] = {"total": 0, "matched": 0, "count": 0}
            slices[slice_key][slice_val]["total"]   += 1
            slices[slice_key][slice_val]["matched"] += matched
            slices[slice_key][slice_val]["count"]   += 1

    metrics = {}
    for slice_key, groups in slices.items():
        for group_name, vals in groups.items():
            avg = round(vals["matched"] / vals["count"], 2) if vals["count"] > 0 else 0
            metrics[f"avg_trials_matched_{slice_key}_{group_name}"] = avg

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Prepare output directories ────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "patients"), exist_ok=True)

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"rag_pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):

        # ── Log pipeline config as params ─────────────────────────────────
        mlflow.log_params({
            "model"               : "medgemma-4b-it",
            "embedding_model"     : "text-embedding-005",
            "retrieval_top_k"     : 20,
            "rerank_top_k"        : 5,
            "total_test_patients" : len(TEST_PATIENTS),
            "gcp_project"         : GCP_PROJECT_ID,
            "medgemma_endpoint"   : os.getenv("MEDGEMMA_ENDPOINT_ID", "4966380223210717184"),
        })

        all_results = []
        summary     = []

        for i, patient in enumerate(TEST_PATIENTS):
            patient_id       = patient["patient_id"]
            text_summary     = patient["summary"]
            slice_condition  = patient["slice_condition"]
            slice_age_group  = patient["slice_age_group"]
            slice_sex        = patient["slice_sex"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Patient {i+1}/{len(TEST_PATIENTS)}: {patient_id}")
            logger.info(f"{'='*60}")

            try:
                result = rag_pipeline(text_summary)

                candidates_count = len(result["candidates_before_rerank"])
                reranked_count   = len(result["retrieved_trials"])

                patient_result = {
                    "patient_id"      : patient_id,
                    "patient_summary" : text_summary,
                    "slice_condition" : slice_condition,
                    "slice_age_group" : slice_age_group,
                    "slice_sex"       : slice_sex,
                    "vector_search_candidates": {
                        "count" : candidates_count,
                        "trials": [build_trial_result(t) for t in result["candidates_before_rerank"]]
                    },
                    "reranked_trials": {
                        "count" : reranked_count,
                        "trials": [build_trial_result(t) for t in result["retrieved_trials"]]
                    },
                    "recommendation"  : result["recommendation"],
                    "timestamp"       : datetime.utcnow().isoformat(),
                    "status"          : "success",
                }

                # ── Save per-patient JSON ─────────────────────────────────
                save_json(
                    patient_result,
                    os.path.join(RESULTS_DIR, "patients", f"{patient_id}.json")
                )

                # Log per-patient metrics to MLflow
                mlflow.log_metrics({
                    f"{patient_id}_candidates_retrieved": candidates_count,
                    f"{patient_id}_trials_matched"      : reranked_count,
                })

                all_results.append(patient_result)
                summary.append({
                    "patient_id"            : patient_id,
                    "patient_summary"       : text_summary,
                    "slice_condition"       : slice_condition,
                    "slice_age_group"       : slice_age_group,
                    "slice_sex"             : slice_sex,
                    "candidates_count"      : candidates_count,
                    "reranked_count"        : reranked_count,
                    "top_trials"            : [
                        {
                            "nct_number": t.get("nct_number"),
                            "title"     : t.get("study_title") or t.get("title"),
                            "condition" : t.get("conditions"),
                            "phase"     : t.get("phase"),
                        }
                        for t in result["retrieved_trials"]
                    ],
                    "recommendation_preview": result["recommendation"][:300] + "..."
                        if len(result["recommendation"]) > 300
                        else result["recommendation"],
                    "status"                : "success",
                    "timestamp"             : datetime.utcnow().isoformat(),
                })

                logger.info(f"  DONE — {reranked_count} trials matched")

            except Exception as e:
                logger.error(f"Failed for {patient_id}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"patient_id": patient_id, "error": str(e), "status": "failed"})
                summary.append({
                    "patient_id"     : patient_id,
                    "slice_condition": slice_condition,
                    "slice_age_group": slice_age_group,
                    "slice_sex"      : slice_sex,
                    "error"          : str(e),
                    "status"         : "failed"
                })

        # ── Save full results + summary JSON ──────────────────────────────
        save_json(all_results, os.path.join(RESULTS_DIR, "rag_output.json"))
        save_json(summary, os.path.join(RESULTS_DIR, "rag_summary.json"))

        # ── Log all artifacts to MLflow (Databricks) ──────────────────────
        mlflow.log_artifacts(RESULTS_DIR, artifact_path="test_results")
        logger.info(f"Logged artifacts to MLflow from {RESULTS_DIR}")

        # ── Compute and log overall metrics ───────────────────────────────
        success_results = [r for r in summary if r.get("status") == "success"]
        failed_results  = [r for r in summary if r.get("status") == "failed"]

        success_rate   = round(len(success_results) / len(TEST_PATIENTS), 2)
        avg_candidates = round(sum(r["candidates_count"] for r in success_results) / max(len(success_results), 1), 2)
        avg_matched    = round(sum(r["reranked_count"]   for r in success_results) / max(len(success_results), 1), 2)

        mlflow.log_metrics({
            "success_rate"            : success_rate,
            "avg_candidates_retrieved": avg_candidates,
            "avg_trials_matched"      : avg_matched,
            "total_failed"            : len(failed_results),
            "total_success"           : len(success_results),
        })

        # ── Log slice metrics (bias detection) ────────────────────────────
        slice_metrics = compute_slice_metrics(summary)
        mlflow.log_metrics(slice_metrics)

        # ── Validation threshold check ─────────────────────────────────────
        check_validation_threshold(success_rate, avg_matched)

        # ── Bias alert check ───────────────────────────────────────────────
        check_bias_alert(slice_metrics, avg_matched)

        # ── Rollback check ─────────────────────────────────────────────────
        check_rollback(success_rate, avg_matched)

        # ── Print summary ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for r in summary:
            if r.get("status") == "failed":
                print(f"\n  [FAIL] {r['patient_id']}: {r.get('error')}")
                continue
            print(f"\n  [PASS] {r['patient_id']} [{r['slice_condition']} | {r['slice_age_group']} | {r['slice_sex']}]")
            print(f"   Candidates: {r['candidates_count']} → Matched: {r['reranked_count']}")
            print(f"   Top trials:")
            for t in r["top_trials"]:
                print(f"     [{t['nct_number']}] {t['title']}")
            print(f"\n   Recommendation:\n   {r['recommendation_preview']}")
            print("-" * 60)

        print(f"\nOverall: success={success_rate*100:.0f}% | avg_candidates={avg_candidates} | avg_matched={avg_matched}")
        print(f"\nSlice metrics:")
        for k, v in slice_metrics.items():
            print(f"   {k}: {v}")

        print(f"\nMLflow run logged to experiment: {MLFLOW_EXPERIMENT_NAME}")











# # test_rag_pipeline.py
# """
# End-to-end test with MLflow tracking:
#   1. Run RAG pipeline for 13 diverse test patients
#   2. Save structured results to test_results/
#   3. Log all metrics and artifacts to MLflow

# MLflow logs per run:
#   - params  : model config, top-k values, patient counts
#   - metrics : candidates retrieved, trials matched, success rate per slice
#   - artifacts: rag_output.json, rag_summary.json, per-patient JSONs

# Output files:
#   test_results/rag_output.json
#   test_results/rag_summary.json
#   test_results/patients/<patient_id>.json
# """

# import sys
# import os
# import logging
# import mlflow
# from datetime import datetime



# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sdk/patient_package")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))  # ← point to models folder

# from rag_service import rag_pipeline  # ← import directly


# from google.cloud import firestore


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)s  %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ── Config ─────────────────────────────────────────────────────────────────────
# GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
# PATIENT_DB     = "patient-db-dev"
# # MLflow config
# MLFLOW_EXPERIMENT_NAME = os.getenv(
#     "MLFLOW_EXPERIMENT_NAME",
#     "/Users/degaonkar.sw@northeastern.edu/triallink-rag-pipeline"
# )


# # ── Patient slices — used for bias metrics ─────────────────────────────────────
# # Each patient tagged with slice metadata for per-group analysis
# TEST_PATIENTS = [
#     # ── DIABETES ──────────────────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_diabetes_001",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",       # 18-59
#         "slice_sex"      : "female",
#         "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_002",
#         "slice_condition": "diabetes",
#         "slice_age_group": "elderly",     # 60+
#         "slice_sex"      : "male",
#         "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_003",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 55-year-old female. Active diagnoses: Type 2 Diabetes, Obesity. Current medications: Metformin 1000mg, Empagliflozin. Recent observations: HbA1c: 7.8%; BMI: 35; fasting glucose: 145 mg/dL. No insulin therapy. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_004",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 38-year-old male. Active diagnoses: Type 1 Diabetes. Current medications: Insulin pump therapy. Recent observations: HbA1c: 7.5%; CGM in use. No oral antidiabetics. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_prediabetes_001",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 50-year-old male. Active diagnoses: Prediabetes, Metabolic Syndrome. Current medications: None. Recent observations: HbA1c: 6.1%; BMI: 29; fasting glucose: 112 mg/dL; triglycerides elevated. Smoking status: Current smoker."
#     },

#     # ── BREAST CANCER ─────────────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_breast_cancer_001",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_002",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_003",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 61-year-old female. Active diagnoses: HR-positive HER2-negative breast cancer stage II. Post-menopausal. Current medications: Letrozole (aromatase inhibitor). No prior CDK4/6 inhibitor therapy. ECOG performance status 0."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_004",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 48-year-old female. Active diagnoses: Metastatic breast cancer, HER2-positive. Prior treatments: Trastuzumab, Pertuzumab. Recent observations: LVEF normal. Currently on Capecitabine. ECOG performance status 1."
#     },

#     # ── OUT OF DISTRIBUTION ───────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_ood_copd_001",
#         "slice_condition": "ood",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."
#     },
#     {
#         "patient_id"  : "test_ood_alzheimers_001",
#         "slice_condition": "ood",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 74-year-old female. Active diagnoses: Early-stage Alzheimer's disease. Current medications: Donepezil 10mg. Recent observations: MMSE score 22/30; MRI shows hippocampal atrophy. No cardiovascular disease. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_ood_hypertension_001",
#         "slice_condition": "ood",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 58-year-old male. Active diagnoses: Resistant Hypertension, Chronic Kidney Disease stage 3. Current medications: Amlodipine, Losartan, Hydrochlorothiazide. Recent observations: BP 158/96 mmHg; eGFR 42 mL/min. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_ood_obesity_001",
#         "slice_condition": "ood",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."
#     },
# ]


# # ── Helpers ────────────────────────────────────────────────────────────────────

# def build_trial_result(trial: dict) -> dict:
#     return {
#         "nct_number"   : trial.get("nct_number", "N/A"),
#         "title"        : trial.get("study_title") or trial.get("title", "N/A"),
#         "condition"    : trial.get("conditions", "N/A"),
#         "disease"      : trial.get("disease", "N/A"),
#         "phase"        : trial.get("phase", "N/A"),
#         "status"       : trial.get("recruitment_status", "N/A"),
#         "eligibility"  : trial.get("eligibility_criteria", "N/A"),
#         "min_age"      : trial.get("min_age", "N/A"),
#         "max_age"      : trial.get("max_age", "N/A"),
#         "sex"          : trial.get("sex", "N/A"),
#         "interventions": trial.get("interventions", "N/A"),
#         "url"          : trial.get("study_url", "N/A"),
#     }


# def check_validation_threshold(success_rate: float, avg_matched: float):
#     """
#     Validation threshold check.
#     Fails CI if core metrics drop below minimum acceptable values.
#     """
#     THRESHOLD_SUCCESS_RATE = 0.7
#     THRESHOLD_AVG_MATCHED  = 3.0

#     errors = []
#     if success_rate < THRESHOLD_SUCCESS_RATE:
#         errors.append(f"success_rate={success_rate} below threshold={THRESHOLD_SUCCESS_RATE}")
#     if avg_matched < THRESHOLD_AVG_MATCHED:
#         errors.append(f"avg_matched={avg_matched} below threshold={THRESHOLD_AVG_MATCHED}")

#     if errors:
#         raise ValueError(f"Validation failed: {'; '.join(errors)}")

#     logger.info(f"Validation passed: success_rate={success_rate}, avg_matched={avg_matched}")


# def check_bias_alert(slice_metrics: dict, avg_matched: float):
#     """
#     Bias alert check.
#     Flags if any demographic slice performs significantly worse than overall avg.
#     Fails CI if difference > 2.0 trials matched.
#     """
#     BIAS_THRESHOLD = 2.0
#     warnings       = []
#     errors         = []

#     for metric_name, slice_avg in slice_metrics.items():
#         diff = avg_matched - slice_avg
#         if diff > BIAS_THRESHOLD:
#             errors.append(f"{metric_name}={slice_avg} is {diff:.1f} below overall avg={avg_matched}")
#         elif diff > 1.0:
#             warnings.append(f"{metric_name}={slice_avg} is {diff:.1f} below overall avg={avg_matched}")

#     for w in warnings:
#         logger.warning(f"Bias warning: {w}")

#     if errors:
#         raise ValueError(f"Bias check failed — significant disparity detected: {'; '.join(errors)}")

#     logger.info("Bias check passed — no significant disparities detected across slices")


# def check_rollback(current_success_rate: float, current_avg_matched: float):
#     """
#     Rollback check.
#     Compares current run metrics against previous MLflow run.
#     Falls back to hardcoded baseline if no previous run exists.
#     Uses 10% degradation tolerance.
#     """
#     BASELINE_SUCCESS_RATE  = 0.7   # fallback if no previous run
#     BASELINE_AVG_MATCHED   = 3.0   # fallback if no previous run
#     DEGRADATION_TOLERANCE  = 0.1   # allow 10% drop

#     try:
#         runs = mlflow.search_runs(
#             experiment_names=[MLFLOW_EXPERIMENT_NAME],
#             order_by=["start_time DESC"],
#             max_results=2,
#             filter_string="status = 'FINISHED'"
#         )

#         if len(runs) < 2:
#             logger.info("No previous run found — using hardcoded baseline for rollback check")
#             prev_success_rate = BASELINE_SUCCESS_RATE
#             prev_avg_matched  = BASELINE_AVG_MATCHED
#         else:
#             prev_run          = runs.iloc[1]
#             prev_success_rate = prev_run.get("metrics.success_rate", BASELINE_SUCCESS_RATE)
#             prev_avg_matched  = prev_run.get("metrics.avg_trials_matched", BASELINE_AVG_MATCHED)
#             logger.info(f"Previous run metrics: success_rate={prev_success_rate}, avg_matched={prev_avg_matched}")

#     except Exception as e:
#         logger.warning(f"Could not query previous run: {e}. Using hardcoded baseline.")
#         prev_success_rate = BASELINE_SUCCESS_RATE
#         prev_avg_matched  = BASELINE_AVG_MATCHED

#     errors = []
#     if current_success_rate < prev_success_rate * (1 - DEGRADATION_TOLERANCE):
#         errors.append(
#             f"success_rate dropped: {current_success_rate} vs previous {prev_success_rate} "
#             f"(tolerance={DEGRADATION_TOLERANCE*100:.0f}%)"
#         )
#     if current_avg_matched < prev_avg_matched * (1 - DEGRADATION_TOLERANCE):
#         errors.append(
#             f"avg_matched dropped: {current_avg_matched} vs previous {prev_avg_matched} "
#             f"(tolerance={DEGRADATION_TOLERANCE*100:.0f}%)"
#         )

#     if errors:
#         raise ValueError(f"Rollback triggered — performance degradation detected: {'; '.join(errors)}")

#     logger.info("Rollback check passed — no performance degradation detected")


# def compute_slice_metrics(results: list[dict]) -> dict:
#     """
#     Compute success rate and avg trials matched per slice.
#     Slices: condition, age_group, sex.
#     Used for bias detection — flags if any slice performs significantly worse.
#     """
#     slices = {
#         "condition" : {},
#         "age_group" : {},
#         "sex"       : {},
#     }

#     for r in results:
#         if r.get("status") != "success":
#             continue

#         matched = r.get("reranked_count", 0)

#         for slice_key, slice_val in [
#             ("condition", r.get("slice_condition")),
#             ("age_group", r.get("slice_age_group")),
#             ("sex",       r.get("slice_sex")),
#         ]:
#             if slice_val not in slices[slice_key]:
#                 slices[slice_key][slice_val] = {"total": 0, "matched": 0, "count": 0}
#             slices[slice_key][slice_val]["total"]   += 1
#             slices[slice_key][slice_val]["matched"] += matched
#             slices[slice_key][slice_val]["count"]   += 1

#     # Compute averages
#     metrics = {}
#     for slice_key, groups in slices.items():
#         for group_name, vals in groups.items():
#             avg = round(vals["matched"] / vals["count"], 2) if vals["count"] > 0 else 0
#             metrics[f"avg_trials_matched_{slice_key}_{group_name}"] = avg

#     return metrics


# # ── Main ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":

#     # ── MLflow setup ──────────────────────────────────────────────────────
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

#     with mlflow.start_run(run_name=f"rag_pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):

#         # ── Log pipeline config as params ─────────────────────────────────
#         mlflow.log_params({
#             "model"               : "medgemma-4b-it",
#             "embedding_model"     : "text-embedding-005",
#             "retrieval_top_k"     : 20,
#             "rerank_top_k"        : 5,
#             "total_test_patients" : len(TEST_PATIENTS),
#             "gcp_project"         : GCP_PROJECT_ID,
#             "medgemma_endpoint"   : os.getenv("MEDGEMMA_ENDPOINT_ID", "4966380223210717184"),
#         })

#         all_results = []
#         summary     = []

#         for i, patient in enumerate(TEST_PATIENTS):
#             patient_id       = patient["patient_id"]
#             text_summary     = patient["summary"]
#             slice_condition  = patient["slice_condition"]
#             slice_age_group  = patient["slice_age_group"]
#             slice_sex        = patient["slice_sex"]

#             logger.info(f"\n{'='*60}")
#             logger.info(f"Patient {i+1}/{len(TEST_PATIENTS)}: {patient_id}")
#             logger.info(f"{'='*60}")

#             try:
                
#                 result = rag_pipeline(text_summary)

#                 candidates_count = len(result["candidates_before_rerank"])
#                 reranked_count   = len(result["retrieved_trials"])

#                 patient_result = {
#                     "patient_id"      : patient_id,
#                     "patient_summary" : text_summary,
#                     "slice_condition" : slice_condition,
#                     "slice_age_group" : slice_age_group,
#                     "slice_sex"       : slice_sex,
#                     "vector_search_candidates": {
#                         "count" : candidates_count,
#                         "trials": [build_trial_result(t) for t in result["candidates_before_rerank"]]
#                     },
#                     "reranked_trials": {
#                         "count" : reranked_count,
#                         "trials": [build_trial_result(t) for t in result["retrieved_trials"]]
#                     },
#                     "recommendation"  : result["recommendation"],
#                     "timestamp"       : datetime.utcnow().isoformat(),
#                     "status"          : "success",
#                 }

#                 # Log per-patient metrics to MLflow
#                 mlflow.log_metrics({
#                     f"{patient_id}_candidates_retrieved": candidates_count,
#                     f"{patient_id}_trials_matched"      : reranked_count,
#                 })

#                 all_results.append(patient_result)
#                 summary.append({
#                     "patient_id"            : patient_id,
#                     "patient_summary"       : text_summary,
#                     "slice_condition"       : slice_condition,
#                     "slice_age_group"       : slice_age_group,
#                     "slice_sex"             : slice_sex,
#                     "candidates_count"      : candidates_count,
#                     "reranked_count"        : reranked_count,
#                     "top_trials"            : [
#                         {
#                             "nct_number": t.get("nct_number"),
#                             "title"     : t.get("study_title") or t.get("title"),
#                             "condition" : t.get("conditions"),
#                             "phase"     : t.get("phase"),
#                         }
#                         for t in result["retrieved_trials"]
#                     ],
#                     "recommendation_preview": result["recommendation"][:300] + "..."
#                         if len(result["recommendation"]) > 300
#                         else result["recommendation"],
#                     "status"                : "success",
#                     "timestamp"             : datetime.utcnow().isoformat(),
#                 })

#                 logger.info(f"  DONE — {reranked_count} trials matched")

#             except Exception as e:
#                 logger.error(f"Failed for {patient_id}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 all_results.append({"patient_id": patient_id, "error": str(e), "status": "failed"})
#                 summary.append({
#                     "patient_id"     : patient_id,
#                     "slice_condition": slice_condition,
#                     "slice_age_group": slice_age_group,
#                     "slice_sex"      : slice_sex,
#                     "error"          : str(e),
#                     "status"         : "failed"
#                 })

#         # ── Compute and log overall metrics ───────────────────────────────
#         success_results = [r for r in summary if r.get("status") == "success"]
#         failed_results  = [r for r in summary if r.get("status") == "failed"]

#         success_rate   = round(len(success_results) / len(TEST_PATIENTS), 2)
#         avg_candidates = round(sum(r["candidates_count"] for r in success_results) / max(len(success_results), 1), 2)
#         avg_matched    = round(sum(r["reranked_count"]   for r in success_results) / max(len(success_results), 1), 2)

#         mlflow.log_metrics({
#             "success_rate"            : success_rate,
#             "avg_candidates_retrieved": avg_candidates,
#             "avg_trials_matched"      : avg_matched,
#             "total_failed"            : len(failed_results),
#             "total_success"           : len(success_results),
#         })

#         # ── Log slice metrics (bias detection) ────────────────────────────
#         slice_metrics = compute_slice_metrics(summary)
#         mlflow.log_metrics(slice_metrics)

#         # ── Validation threshold check ─────────────────────────────────────
#         check_validation_threshold(success_rate, avg_matched)

#         # ── Bias alert check ───────────────────────────────────────────────
#         check_bias_alert(slice_metrics, avg_matched)

#         # ── Rollback check ─────────────────────────────────────────────────
#         check_rollback(success_rate, avg_matched)

#         # ── Print summary ──────────────────────────────────────────────────
#         print("\n" + "=" * 60)
#         print("RESULTS SUMMARY")
#         print("=" * 60)
#         for r in summary:
#             if r.get("status") == "failed":
#                 print(f"\n  [FAIL] {r['patient_id']}: {r.get('error')}")
#                 continue
#             print(f"\n  [PASS] {r['patient_id']} [{r['slice_condition']} | {r['slice_age_group']} | {r['slice_sex']}]")
#             print(f"   Candidates: {r['candidates_count']} → Matched: {r['reranked_count']}")
#             print(f"   Top trials:")
#             for t in r["top_trials"]:
#                 print(f"     [{t['nct_number']}] {t['title']}")
#             print(f"\n   Recommendation:\n   {r['recommendation_preview']}")
#             print("-" * 60)

#         print(f"\nOverall: success={success_rate*100:.0f}% | avg_candidates={avg_candidates} | avg_matched={avg_matched}")
#         print(f"\nSlice metrics:")
#         for k, v in slice_metrics.items():
#             print(f"   {k}: {v}")

#             print(f"\nMLflow run logged to experiment: {MLFLOW_EXPERIMENT_NAME}")


# # test_rag_pipeline.py
# """
# End-to-end test with MLflow tracking:
#   1. Run RAG pipeline for 13 diverse test patients
#   2. Save structured results to test_results/
#   3. Log all metrics and artifacts to MLflow

# MLflow logs per run:
#   - params  : model config, top-k values, patient counts
#   - metrics : candidates retrieved, trials matched, success rate per slice
#   - artifacts: rag_output.json, rag_summary.json, per-patient JSONs

# Output files:
#   test_results/rag_output.json
#   test_results/rag_summary.json
#   test_results/patients/<patient_id>.json
# """

# import sys
# import os
# import logging
# import mlflow
# from datetime import datetime

# sys.path.insert(0, os.path.abspath("sdk/patient_package"))
# sys.path.insert(0, os.path.abspath("pipelines/dags/src"))

# from google.cloud import firestore
# from rag_service import rag_pipeline

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)s  %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ── Config ─────────────────────────────────────────────────────────────────────
# GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
# PATIENT_DB     = "patient-db-dev"
# # MLflow config
# MLFLOW_EXPERIMENT_NAME = os.getenv(
#     "MLFLOW_EXPERIMENT_NAME",
#     "/Users/degaonkar.sw@northeastern.edu/triallink-rag-pipeline"
# )


# # ── Patient slices — used for bias metrics ─────────────────────────────────────
# # Each patient tagged with slice metadata for per-group analysis
# TEST_PATIENTS = [
#     # ── DIABETES ──────────────────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_diabetes_001",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",       # 18-59
#         "slice_sex"      : "female",
#         "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_002",
#         "slice_condition": "diabetes",
#         "slice_age_group": "elderly",     # 60+
#         "slice_sex"      : "male",
#         "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_003",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 55-year-old female. Active diagnoses: Type 2 Diabetes, Obesity. Current medications: Metformin 1000mg, Empagliflozin. Recent observations: HbA1c: 7.8%; BMI: 35; fasting glucose: 145 mg/dL. No insulin therapy. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_diabetes_004",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 38-year-old male. Active diagnoses: Type 1 Diabetes. Current medications: Insulin pump therapy. Recent observations: HbA1c: 7.5%; CGM in use. No oral antidiabetics. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_prediabetes_001",
#         "slice_condition": "diabetes",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 50-year-old male. Active diagnoses: Prediabetes, Metabolic Syndrome. Current medications: None. Recent observations: HbA1c: 6.1%; BMI: 29; fasting glucose: 112 mg/dL; triglycerides elevated. Smoking status: Current smoker."
#     },

#     # ── BREAST CANCER ─────────────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_breast_cancer_001",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_002",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_003",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 61-year-old female. Active diagnoses: HR-positive HER2-negative breast cancer stage II. Post-menopausal. Current medications: Letrozole (aromatase inhibitor). No prior CDK4/6 inhibitor therapy. ECOG performance status 0."
#     },
#     {
#         "patient_id"  : "test_breast_cancer_004",
#         "slice_condition": "breast_cancer",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 48-year-old female. Active diagnoses: Metastatic breast cancer, HER2-positive. Prior treatments: Trastuzumab, Pertuzumab. Recent observations: LVEF normal. Currently on Capecitabine. ECOG performance status 1."
#     },

#     # ── OUT OF DISTRIBUTION ───────────────────────────────────────────────────
#     {
#         "patient_id"  : "test_ood_copd_001",
#         "slice_condition": "ood",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."
#     },
#     {
#         "patient_id"  : "test_ood_alzheimers_001",
#         "slice_condition": "ood",
#         "slice_age_group": "elderly",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 74-year-old female. Active diagnoses: Early-stage Alzheimer's disease. Current medications: Donepezil 10mg. Recent observations: MMSE score 22/30; MRI shows hippocampal atrophy. No cardiovascular disease. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_ood_hypertension_001",
#         "slice_condition": "ood",
#         "slice_age_group": "adult",
#         "slice_sex"      : "male",
#         "summary": "Patient is a 58-year-old male. Active diagnoses: Resistant Hypertension, Chronic Kidney Disease stage 3. Current medications: Amlodipine, Losartan, Hydrochlorothiazide. Recent observations: BP 158/96 mmHg; eGFR 42 mL/min. Smoking status: Never smoker."
#     },
#     {
#         "patient_id"  : "test_ood_obesity_001",
#         "slice_condition": "ood",
#         "slice_age_group": "adult",
#         "slice_sex"      : "female",
#         "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."
#     },
# ]


# # ── Helpers ────────────────────────────────────────────────────────────────────

# def build_trial_result(trial: dict) -> dict:
#     return {
#         "nct_number"   : trial.get("nct_number", "N/A"),
#         "title"        : trial.get("study_title") or trial.get("title", "N/A"),
#         "condition"    : trial.get("conditions", "N/A"),
#         "disease"      : trial.get("disease", "N/A"),
#         "phase"        : trial.get("phase", "N/A"),
#         "status"       : trial.get("recruitment_status", "N/A"),
#         "eligibility"  : trial.get("eligibility_criteria", "N/A"),
#         "min_age"      : trial.get("min_age", "N/A"),
#         "max_age"      : trial.get("max_age", "N/A"),
#         "sex"          : trial.get("sex", "N/A"),
#         "interventions": trial.get("interventions", "N/A"),
#         "url"          : trial.get("study_url", "N/A"),
#     }


# def compute_slice_metrics(results: list[dict]) -> dict:
#     """
#     Compute success rate and avg trials matched per slice.
#     Slices: condition, age_group, sex.
#     Used for bias detection — flags if any slice performs significantly worse.
#     """
#     slices = {
#         "condition" : {},
#         "age_group" : {},
#         "sex"       : {},
#     }

#     for r in results:
#         if r.get("status") != "success":
#             continue

#         matched = r.get("reranked_count", 0)

#         for slice_key, slice_val in [
#             ("condition", r.get("slice_condition")),
#             ("age_group", r.get("slice_age_group")),
#             ("sex",       r.get("slice_sex")),
#         ]:
#             if slice_val not in slices[slice_key]:
#                 slices[slice_key][slice_val] = {"total": 0, "matched": 0, "count": 0}
#             slices[slice_key][slice_val]["total"]   += 1
#             slices[slice_key][slice_val]["matched"] += matched
#             slices[slice_key][slice_val]["count"]   += 1

#     # Compute averages
#     metrics = {}
#     for slice_key, groups in slices.items():
#         for group_name, vals in groups.items():
#             avg = round(vals["matched"] / vals["count"], 2) if vals["count"] > 0 else 0
#             metrics[f"avg_trials_matched_{slice_key}_{group_name}"] = avg

#     return metrics


# # ── Main ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":

#     # ── MLflow setup ──────────────────────────────────────────────────────
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

#     with mlflow.start_run(run_name=f"rag_pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):

#         # ── Log pipeline config as params ─────────────────────────────────
#         mlflow.log_params({
#             "model"               : "medgemma-4b-it",
#             "embedding_model"     : "text-embedding-005",
#             "retrieval_top_k"     : 20,
#             "rerank_top_k"        : 5,
#             "total_test_patients" : len(TEST_PATIENTS),
#             "gcp_project"         : GCP_PROJECT_ID,
#             "medgemma_endpoint"   : os.getenv("MEDGEMMA_ENDPOINT_ID", "4966380223210717184"),
#         })

#         all_results = []
#         summary     = []

#         for i, patient in enumerate(TEST_PATIENTS):
#             patient_id       = patient["patient_id"]
#             text_summary     = patient["summary"]
#             slice_condition  = patient["slice_condition"]
#             slice_age_group  = patient["slice_age_group"]
#             slice_sex        = patient["slice_sex"]

#             logger.info(f"\n{'='*60}")
#             logger.info(f"Patient {i+1}/{len(TEST_PATIENTS)}: {patient_id}")
#             logger.info(f"{'='*60}")

#             try:
#                 result = rag_pipeline(text_summary)

#                 candidates_count = len(result["candidates_before_rerank"])
#                 reranked_count   = len(result["retrieved_trials"])

#                 patient_result = {
#                     "patient_id"      : patient_id,
#                     "patient_summary" : text_summary,
#                     "slice_condition" : slice_condition,
#                     "slice_age_group" : slice_age_group,
#                     "slice_sex"       : slice_sex,
#                     "vector_search_candidates": {
#                         "count" : candidates_count,
#                         "trials": [build_trial_result(t) for t in result["candidates_before_rerank"]]
#                     },
#                     "reranked_trials": {
#                         "count" : reranked_count,
#                         "trials": [build_trial_result(t) for t in result["retrieved_trials"]]
#                     },
#                     "recommendation"  : result["recommendation"],
#                     "timestamp"       : datetime.utcnow().isoformat(),
#                     "status"          : "success",
#                 }

#                 # Log per-patient metrics to MLflow
#                 mlflow.log_metrics({
#                     f"{patient_id}_candidates_retrieved": candidates_count,
#                     f"{patient_id}_trials_matched"      : reranked_count,
#                 })

#                 all_results.append(patient_result)
#                 summary.append({
#                     "patient_id"            : patient_id,
#                     "patient_summary"       : text_summary,
#                     "slice_condition"       : slice_condition,
#                     "slice_age_group"       : slice_age_group,
#                     "slice_sex"             : slice_sex,
#                     "candidates_count"      : candidates_count,
#                     "reranked_count"        : reranked_count,
#                     "top_trials"            : [
#                         {
#                             "nct_number": t.get("nct_number"),
#                             "title"     : t.get("study_title") or t.get("title"),
#                             "condition" : t.get("conditions"),
#                             "phase"     : t.get("phase"),
#                         }
#                         for t in result["retrieved_trials"]
#                     ],
#                     "recommendation_preview": result["recommendation"][:300] + "..."
#                         if len(result["recommendation"]) > 300
#                         else result["recommendation"],
#                     "status"                : "success",
#                     "timestamp"             : datetime.utcnow().isoformat(),
#                 })

#                 logger.info(f"  DONE — {reranked_count} trials matched")

#             except Exception as e:
#                 logger.error(f"Failed for {patient_id}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 all_results.append({"patient_id": patient_id, "error": str(e), "status": "failed"})
#                 summary.append({
#                     "patient_id"     : patient_id,
#                     "slice_condition": slice_condition,
#                     "slice_age_group": slice_age_group,
#                     "slice_sex"      : slice_sex,
#                     "error"          : str(e),
#                     "status"         : "failed"
#                 })

#         # ── Compute and log overall metrics ───────────────────────────────
#         success_results = [r for r in summary if r.get("status") == "success"]
#         failed_results  = [r for r in summary if r.get("status") == "failed"]

#         success_rate   = round(len(success_results) / len(TEST_PATIENTS), 2)
#         avg_candidates = round(sum(r["candidates_count"] for r in success_results) / max(len(success_results), 1), 2)
#         avg_matched    = round(sum(r["reranked_count"]   for r in success_results) / max(len(success_results), 1), 2)

#         mlflow.log_metrics({
#             "success_rate"            : success_rate,
#             "avg_candidates_retrieved": avg_candidates,
#             "avg_trials_matched"      : avg_matched,
#             "total_failed"            : len(failed_results),
#             "total_success"           : len(success_results),
#         })

#         # ── Log slice metrics (bias detection) ────────────────────────────
#         slice_metrics = compute_slice_metrics(summary)
#         mlflow.log_metrics(slice_metrics)

#         # ── Print summary ──────────────────────────────────────────────────
#         print("\n" + "=" * 60)
#         print("RESULTS SUMMARY")
#         print("=" * 60)
#         for r in summary:
#             if r.get("status") == "failed":
#                 print(f"\n  [FAIL] {r['patient_id']}: {r.get('error')}")
#                 continue
#             print(f"\n  [PASS] {r['patient_id']} [{r['slice_condition']} | {r['slice_age_group']} | {r['slice_sex']}]")
#             print(f"   Candidates: {r['candidates_count']} → Matched: {r['reranked_count']}")
#             print(f"   Top trials:")
#             for t in r["top_trials"]:
#                 print(f"     [{t['nct_number']}] {t['title']}")
#             print(f"\n   Recommendation:\n   {r['recommendation_preview']}")
#             print("-" * 60)

#         print(f"\nOverall: success={success_rate*100:.0f}% | avg_candidates={avg_candidates} | avg_matched={avg_matched}")
#         print(f"\nSlice metrics:")
#         for k, v in slice_metrics.items():
#             print(f"   {k}: {v}")

#         print(f"\nMLflow run logged to experiment: {MLFLOW_EXPERIMENT_NAME}")
#         logger.info(f"test_results/rag_output.json")
#         logger.info(f"test_results/rag_summary.json")
#         logger.info(f"test_results/patients/<patient_id>.json")

