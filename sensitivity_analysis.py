"""
Sensitivity Analysis - TrialLink RAG Pipeline
==============================================
Varies one hyperparameter at a time while keeping others fixed.
Each combination is logged as a separate MLflow run.

Parameters tested:
    RETRIEVAL_TOP_K    : [10, 20, 50]   — candidates from vector search
    RERANK_TOP_K       : [3, 5, 10]     — trials kept after reranking
    SIMILARITY_THRESHOLD: [0.5, 0.7, 0.9] — min similarity score

Metrics logged per run:
    - candidates_retrieved  : avg across all patients
    - trials_matched        : avg across all patients
    - ood_trials_found      : avg for OOD patients (tests threshold sensitivity)
    - success_rate          : % of patients that got results
    - per-patient metrics

All runs logged to MLflow experiment: triallink-sensitivity-analysis
"""

import os
import sys
import json
import logging
import mlflow
from datetime import datetime

sys.path.insert(0, os.path.abspath("sdk/patient_package"))
sys.path.insert(0, os.path.abspath("pipelines/dags/src"))

# Patch rag_service config before importing pipeline
import rag_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "/Users/degaonkar.sw@northeastern.edu/triallink-sensitivity-analysis"
)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ── Default (baseline) values ──────────────────────────────────────────────────
DEFAULT_RETRIEVAL_TOP_K      = 20
DEFAULT_RERANK_TOP_K         = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7

# ── Parameter search space ─────────────────────────────────────────────────────
SENSITIVITY_PARAMS = {
    "RETRIEVAL_TOP_K"    : [10, 20, 50],
    "RERANK_TOP_K"       : [3, 5, 10],
    "SIMILARITY_THRESHOLD": [0.5, 0.7, 0.9],
}

# ── All 13 test patients ───────────────────────────────────────────────────────
TEST_PATIENTS = [
    {"patient_id": "test_diabetes_001",      "slice": "diabetes",      "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."},
    {"patient_id": "test_diabetes_002",      "slice": "diabetes",      "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."},
    {"patient_id": "test_diabetes_003",      "slice": "diabetes",      "summary": "Patient is a 55-year-old female. Active diagnoses: Type 2 Diabetes, Obesity. Current medications: Metformin 1000mg, Empagliflozin. Recent observations: HbA1c: 7.8%; BMI: 35; fasting glucose: 145 mg/dL. No insulin therapy. Smoking status: Never smoker."},
    {"patient_id": "test_diabetes_004",      "slice": "diabetes",      "summary": "Patient is a 38-year-old male. Active diagnoses: Type 1 Diabetes. Current medications: Insulin pump therapy. Recent observations: HbA1c: 7.5%; CGM in use. No oral antidiabetics. Smoking status: Never smoker."},
    {"patient_id": "test_prediabetes_001",   "slice": "diabetes",      "summary": "Patient is a 50-year-old male. Active diagnoses: Prediabetes, Metabolic Syndrome. Current medications: None. Recent observations: HbA1c: 6.1%; BMI: 29; fasting glucose: 112 mg/dL; triglycerides elevated. Smoking status: Current smoker."},
    {"patient_id": "test_breast_cancer_001", "slice": "breast_cancer", "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."},
    {"patient_id": "test_breast_cancer_002", "slice": "breast_cancer", "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."},
    {"patient_id": "test_breast_cancer_003", "slice": "breast_cancer", "summary": "Patient is a 61-year-old female. Active diagnoses: HR-positive HER2-negative breast cancer stage II. Post-menopausal. Current medications: Letrozole (aromatase inhibitor). No prior CDK4/6 inhibitor therapy. ECOG performance status 0."},
    {"patient_id": "test_breast_cancer_004", "slice": "breast_cancer", "summary": "Patient is a 48-year-old female. Active diagnoses: Metastatic breast cancer, HER2-positive. Prior treatments: Trastuzumab, Pertuzumab. Recent observations: LVEF normal. Currently on Capecitabine. ECOG performance status 1."},
    {"patient_id": "test_ood_copd_001",      "slice": "ood",           "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."},
    {"patient_id": "test_ood_alzheimers_001","slice": "ood",           "summary": "Patient is a 74-year-old female. Active diagnoses: Early-stage Alzheimer's disease. Current medications: Donepezil 10mg. Recent observations: MMSE score 22/30; MRI shows hippocampal atrophy. No cardiovascular disease. Smoking status: Never smoker."},
    {"patient_id": "test_ood_hypertension_001","slice": "ood",         "summary": "Patient is a 58-year-old male. Active diagnoses: Resistant Hypertension, Chronic Kidney Disease stage 3. Current medications: Amlodipine, Losartan, Hydrochlorothiazide. Recent observations: BP 158/96 mmHg; eGFR 42 mL/min. Smoking status: Never smoker."},
    {"patient_id": "test_ood_obesity_001",   "slice": "ood",           "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."},
]


def run_with_config(
    retrieval_top_k      : int,
    rerank_top_k         : int,
    similarity_threshold : float,
    param_name           : str,   # which param is being varied
) -> dict:
    """
    Run the full pipeline for all 13 patients with given config.
    Patches rag_service constants before each run.
    Logs results to a single MLflow run.

    Returns:
        summary dict with avg metrics
    """
    run_name = (
        f"{param_name}_"
        f"ret{retrieval_top_k}_"
        f"rer{rerank_top_k}_"
        f"thr{str(similarity_threshold).replace('.', '')}"
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Run: {run_name}")
    logger.info(f"  RETRIEVAL_TOP_K     = {retrieval_top_k}")
    logger.info(f"  RERANK_TOP_K        = {rerank_top_k}")
    logger.info(f"  SIMILARITY_THRESHOLD= {similarity_threshold}")
    logger.info(f"{'='*60}")

    # Patch rag_service constants
    rag_service.RETRIEVAL_TOP_K      = retrieval_top_k
    rag_service.RERANK_TOP_K         = rerank_top_k

    # Patch similarity threshold inside query_vector_search
    # We do this by monkeypatching the function
    original_query = rag_service.query_vector_search

    def patched_query(patient_embedding, top_k=retrieval_top_k):
        from google.cloud import aiplatform
        aiplatform.init(
            project=rag_service.GCP_PROJECT_ID,
            location=os.getenv("GCP_REGION", "us-central1")
        )
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
        )
        fetch_k = top_k * 3
        results = index_endpoint.find_neighbors(
            deployed_index_id=os.getenv("DEPLOYED_INDEX_ID"),
            queries=[patient_embedding],
            num_neighbors=fetch_k
        )
        matches     = results[0]
        seen_nct_ids = {}
        for match in matches:
            nct_id = match.id.rsplit("_", 1)[0]
            score  = match.distance
            if score < similarity_threshold:
                if nct_id not in seen_nct_ids or score < seen_nct_ids[nct_id]:
                    seen_nct_ids[nct_id] = score
        if not seen_nct_ids:
            logger.warning("No trials above similarity threshold")
            return []
        sorted_trials = sorted(seen_nct_ids.items(), key=lambda x: x[1])
        top_nct_ids   = [nct_id for nct_id, _ in sorted_trials[:top_k]]
        logger.info(f"Vector search top {top_k}: {top_nct_ids}")
        return top_nct_ids

    rag_service.query_vector_search = patched_query

    results       = []
    ood_matched   = []
    total_candidates = []
    total_matched    = []
    success_count    = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "retrieval_top_k"      : retrieval_top_k,
            "rerank_top_k"         : rerank_top_k,
            "similarity_threshold" : similarity_threshold,
            "varied_param"         : param_name,
            "total_patients"       : len(TEST_PATIENTS),
        })

        for patient in TEST_PATIENTS:
            pid     = patient["patient_id"]
            summary = patient["summary"]
            slice_  = patient["slice"]

            try:
                result           = rag_service.rag_pipeline(summary)
                candidates_count = len(result["candidates_before_rerank"])
                matched_count    = len(result["retrieved_trials"])

                total_candidates.append(candidates_count)
                total_matched.append(matched_count)
                success_count += 1

                if slice_ == "ood":
                    ood_matched.append(matched_count)

                mlflow.log_metrics({
                    f"{pid}_candidates": candidates_count,
                    f"{pid}_matched"   : matched_count,
                })

                results.append({
                    "patient_id"  : pid,
                    "slice"       : slice_,
                    "candidates"  : candidates_count,
                    "matched"     : matched_count,
                    "status"      : "success",
                })
                logger.info(f"  {pid}: {candidates_count} candidates → {matched_count} matched")

            except Exception as e:
                logger.error(f"  {pid}: FAILED — {e}")
                results.append({"patient_id": pid, "slice": slice_, "status": "failed", "error": str(e)})

        # Overall metrics
        n               = max(len(total_candidates), 1)
        avg_candidates  = round(sum(total_candidates) / n, 2)
        avg_matched     = round(sum(total_matched) / n, 2)
        avg_ood_matched = round(sum(ood_matched) / max(len(ood_matched), 1), 2)
        success_rate    = round(success_count / len(TEST_PATIENTS), 2)

        mlflow.log_metrics({
            "avg_candidates_retrieved": avg_candidates,
            "avg_trials_matched"      : avg_matched,
            "avg_ood_trials_matched"  : avg_ood_matched,
            "success_rate"            : success_rate,
        })

        logger.info(f"\n  avg_candidates={avg_candidates} | avg_matched={avg_matched} | avg_ood={avg_ood_matched} | success={success_rate}")

    # Restore original function
    rag_service.query_vector_search = original_query

    return {
        "run_name"       : run_name,
        "avg_candidates" : avg_candidates,
        "avg_matched"    : avg_matched,
        "avg_ood_matched": avg_ood_matched,
        "success_rate"   : success_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — vary one parameter at a time
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_run_results = []

    # ── 1. Vary RETRIEVAL_TOP_K ────────────────────────────────────────────────
    logger.info("\n\nVARYING RETRIEVAL_TOP_K")
    for val in SENSITIVITY_PARAMS["RETRIEVAL_TOP_K"]:
        r = run_with_config(
            retrieval_top_k      = val,
            rerank_top_k         = DEFAULT_RERANK_TOP_K,
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD,
            param_name           = "RETRIEVAL_TOP_K",
        )
        all_run_results.append(r)

    # ── 2. Vary RERANK_TOP_K ───────────────────────────────────────────────────
    logger.info("\n\nVARYING RERANK_TOP_K")
    for val in SENSITIVITY_PARAMS["RERANK_TOP_K"]:
        r = run_with_config(
            retrieval_top_k      = DEFAULT_RETRIEVAL_TOP_K,
            rerank_top_k         = val,
            similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD,
            param_name           = "RERANK_TOP_K",
        )
        all_run_results.append(r)

    # ── 3. Vary SIMILARITY_THRESHOLD ──────────────────────────────────────────
    logger.info("\n\nVARYING SIMILARITY_THRESHOLD")
    for val in SENSITIVITY_PARAMS["SIMILARITY_THRESHOLD"]:
        r = run_with_config(
            retrieval_top_k      = DEFAULT_RETRIEVAL_TOP_K,
            rerank_top_k         = DEFAULT_RERANK_TOP_K,
            similarity_threshold = val,
            param_name           = "SIMILARITY_THRESHOLD",
        )
        all_run_results.append(r)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"{'Run':<45} {'Avg Candidates':>15} {'Avg Matched':>12} {'Avg OOD':>10} {'Success':>9}")
    print("-" * 80)
    for r in all_run_results:
        print(
            f"{r['run_name']:<45} "
            f"{r['avg_candidates']:>15} "
            f"{r['avg_matched']:>12} "
            f"{r['avg_ood_matched']:>10} "
            f"{r['success_rate']:>9}"
        )

    # Save summary
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/sensitivity_results.json", "w") as f:
        json.dump(all_run_results, f, indent=2)

    print(f"\nAll runs logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Results saved to: test_results/sensitivity_results.json")
    print(f"View in MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")