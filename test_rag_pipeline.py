# test_rag_pipeline.py
"""
End-to-end test:
  1. Parse Synthea FHIR files using FHIRParser
  2. Convert to Patient object using data_models.py
  3. Generate text summary using Patient.to_text_summary()
  4. Store patient in Firestore (patient-db-dev)
  5. Run RAG pipeline
  6. Save structured results to test_results/ for evaluation

Output files:
  test_results/rag_output.json          ← full results, all patients
  test_results/rag_summary.json         ← lightweight summary for quick review
  test_results/patients/<patient_id>.json ← one file per patient with full detail
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Add sdk/patient_package and pipelines to path ─────────────────────────────
sys.path.insert(0, os.path.abspath("sdk/patient_package"))
sys.path.insert(0, os.path.abspath("pipelines/dags/src"))

from data_parser import FHIRParser
from data_models import Patient
from google.cloud import firestore
from rag_service import rag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
PATIENT_DB     = "patient-db-dev"
FHIR_DIR       = os.getenv("FHIR_DIR", "data/synthea_output/fhir")
OUTPUT_DIR     = "test_results"
MAX_PATIENTS   = int(os.getenv("MAX_PATIENTS", "5"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_fhir_files(directory: str, limit: int) -> list[str]:
    fhir_files = [
        str(f) for f in Path(directory).rglob("*.json")
        if "hospitalInformation" not in str(f)  # skip non-patient files
    ]
    logger.info(f"Found {len(fhir_files)} FHIR patient files in {directory}")
    return fhir_files[:limit]


def parse_fhir_to_patient(fhir_file: str) -> Patient:
    parser = FHIRParser(fhir_file)
    return parser.parse_to_patient()


def store_patient_in_firestore(patient: Patient) -> str:
    db = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    patient_id = patient.demographics.patient_id
    doc_ref = db.collection("patients").document(patient_id)
    doc_ref.set(patient.model_dump(mode="json"))
    logger.info(f"Stored patient {patient_id} in Firestore")
    return patient_id


def build_patient_profile(patient: Patient) -> dict:
    """Build a clean patient profile dict for output."""
    return {
        "patient_id"   : patient.demographics.patient_id,
        "age"          : patient.demographics.age,
        "gender"       : patient.demographics.gender.value,
        "race"         : patient.demographics.race,
        "location"     : f"{patient.demographics.city}, {patient.demographics.state}",
        "conditions"   : [
            {
                "code"        : c.code,
                "name"        : c.display_name,
                "status"      : c.status.value,
                "onset_date"  : str(c.onset_date) if c.onset_date else None,
            }
            for c in patient.conditions
        ],
        "active_conditions": [c.display_name for c in patient.get_active_conditions()],
        "medications"  : [
            {
                "name"   : m.display_name,
                "status" : m.status,
                "dosage" : m.dosage,
                "reason" : m.reason,
            }
            for m in patient.medications
        ],
        "current_medications": [m.display_name for m in patient.get_current_medications()],
        "observations" : [
            {
                "name"  : o.display_name,
                "value" : o.value,
                "unit"  : o.unit,
                "date"  : str(o.date),
            }
            for o in patient.get_recent_observations(days=365)[:10]
        ],
        "procedures"   : [p.display_name for p in patient.procedures[-3:]] if patient.procedures else [],
        "allergies"    : [a.substance for a in patient.allergies],
        "lifestyle"    : {
            "smoking" : patient.lifestyle.smoking_status if patient.lifestyle else None,
            "alcohol" : patient.lifestyle.alcohol_use if patient.lifestyle else None,
        },
    }


def build_trial_result(trial: dict) -> dict:
    """Build a clean trial result dict for output."""
    return {
        "nct_number"  : trial.get("nct_number", "N/A"),
        "title"       : trial.get("study_title") or trial.get("title", "N/A"),
        "condition"   : trial.get("conditions", "N/A"),
        "disease"     : trial.get("disease", "N/A"),
        "phase"       : trial.get("phase", "N/A"),
        "status"      : trial.get("recruitment_status", "N/A"),
        "eligibility" : trial.get("eligibility_criteria", "N/A"),
        "min_age"     : trial.get("min_age", "N/A"),
        "max_age"     : trial.get("max_age", "N/A"),
        "sex"         : trial.get("sex", "N/A"),
        "interventions": trial.get("interventions", "N/A"),
        "url"         : trial.get("study_url", "N/A"),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from datetime import datetime

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/patients", exist_ok=True)

    # ── Synthetic test patients ───────────────────────────────────────────
    test_patients = [
        # ── DIABETES PATIENTS ─────────────────────────────────────────────────
        {
            "patient_id": "test_diabetes_001",
            "summary": "Patient is a 45-year-old female. Active diagnoses: Type 2 Diabetes. Current medications: Metformin 500mg. Recent observations: HbA1c: 8.2%; BMI: 28. Smoking status: Never smoker."
        },
        {
            "patient_id": "test_diabetes_002",
            "summary": "Patient is a 62-year-old male. Active diagnoses: Type 2 Diabetes, Hypertension. Current medications: Metformin, Lisinopril. Recent observations: HbA1c: 9.1%; BMI: 32. Smoking status: Former smoker."
        },
        {
            "patient_id": "test_diabetes_003",
            "summary": "Patient is a 55-year-old female. Active diagnoses: Type 2 Diabetes, Obesity. Current medications: Metformin 1000mg, Empagliflozin. Recent observations: HbA1c: 7.8%; BMI: 35; fasting glucose: 145 mg/dL. No insulin therapy. Smoking status: Never smoker."
        },
        {
            "patient_id": "test_diabetes_004",
            "summary": "Patient is a 38-year-old male. Active diagnoses: Type 1 Diabetes. Current medications: Insulin pump therapy. Recent observations: HbA1c: 7.5%; CGM in use. No oral antidiabetics. Smoking status: Never smoker."
        },
        {
            "patient_id": "test_prediabetes_001",
            "summary": "Patient is a 50-year-old male. Active diagnoses: Prediabetes, Metabolic Syndrome. Current medications: None. Recent observations: HbA1c: 6.1%; BMI: 29; fasting glucose: 112 mg/dL; triglycerides elevated. Smoking status: Current smoker."
        },

        # ── BREAST CANCER PATIENTS ────────────────────────────────────────────
        {
            "patient_id": "test_breast_cancer_001",
            "summary": "Patient is a 52-year-old female. Active diagnoses: HER2-positive breast cancer stage II. Recent procedures: Lumpectomy. Smoking status: Never smoker. Post-menopausal. No prior targeted therapy. ECOG performance status 0."
        },
        {
            "patient_id": "test_breast_cancer_002",
            "summary": "Patient is a 44-year-old female. Active diagnoses: Triple Negative Breast Cancer (TNBC) stage III. Pre-menopausal. Recent procedures: Mastectomy, completed 4 cycles of chemotherapy. No prior immunotherapy. ECOG performance status 1."
        },
        {
            "patient_id": "test_breast_cancer_003",
            "summary": "Patient is a 61-year-old female. Active diagnoses: HR-positive HER2-negative breast cancer stage II. Post-menopausal. Current medications: Letrozole (aromatase inhibitor). No prior CDK4/6 inhibitor therapy. ECOG performance status 0."
        },
        {
            "patient_id": "test_breast_cancer_004",
            "summary": "Patient is a 48-year-old female. Active diagnoses: Metastatic breast cancer, HER2-positive. Prior treatments: Trastuzumab, Pertuzumab. Recent observations: LVEF normal. Currently on Capecitabine. ECOG performance status 1."
        },

        # ── OUT OF DISTRIBUTION PATIENTS (not diabetes or breast cancer) ──────
        {
            "patient_id": "test_ood_copd_001",
            "summary": "Patient is a 67-year-old male. Active diagnoses: Chronic Obstructive Pulmonary Disease (COPD), stage 3. Current medications: Tiotropium, Salmeterol, Fluticasone inhaler. Recent observations: FEV1: 42% predicted; frequent exacerbations. Smoking status: Former smoker, 40 pack-years."
        },
        {
            "patient_id": "test_ood_alzheimers_001",
            "summary": "Patient is a 74-year-old female. Active diagnoses: Early-stage Alzheimer's disease. Current medications: Donepezil 10mg. Recent observations: MMSE score 22/30; MRI shows hippocampal atrophy. No cardiovascular disease. Smoking status: Never smoker."
        },
        {
            "patient_id": "test_ood_hypertension_001",
            "summary": "Patient is a 58-year-old male. Active diagnoses: Resistant Hypertension, Chronic Kidney Disease stage 3. Current medications: Amlodipine, Losartan, Hydrochlorothiazide. Recent observations: BP 158/96 mmHg; eGFR 42 mL/min. Smoking status: Never smoker."
        },
        {
            "patient_id": "test_ood_obesity_001",
            "summary": "Patient is a 41-year-old female. Active diagnoses: Morbid Obesity, Sleep Apnea. Current medications: None. Recent observations: BMI: 42; AHI: 28 events/hour; no diabetes. Smoking status: Never smoker. No prior bariatric surgery."
        },
    ]

    all_results = []
    summary     = []

    for i, patient in enumerate(test_patients):
        patient_id   = patient["patient_id"]
        text_summary = patient["summary"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Patient {i+1}/{len(test_patients)}: {patient_id}")
        logger.info(f"{'='*60}")
        logger.info(f"  Summary: {text_summary}")

        try:
            result = rag_pipeline(text_summary)

            patient_result = {
                "patient_id"     : patient_id,
                "patient_summary": text_summary,
                "vector_search_candidates": {
                    "count"  : len(result["candidates_before_rerank"]),
                    "trials" : [build_trial_result(t) for t in result["candidates_before_rerank"]]
                },
                "reranked_trials": {
                    "count"  : len(result["retrieved_trials"]),
                    "trials" : [build_trial_result(t) for t in result["retrieved_trials"]]
                },
                "recommendation" : result["recommendation"],
                "timestamp"      : datetime.utcnow().isoformat(),
                "status"         : "success",
            }

            # Save per patient
            with open(f"{OUTPUT_DIR}/patients/{patient_id}.json", "w") as f:
                json.dump(patient_result, f, indent=2, default=str)

            all_results.append(patient_result)
            summary.append({
                "patient_id"             : patient_id,
                "patient_summary"        : text_summary,
                "candidates_count"       : len(result["candidates_before_rerank"]),
                "reranked_count"         : len(result["retrieved_trials"]),
                "top_trials"             : [
                    {
                        "nct_number" : t.get("nct_number"),
                        "title"      : t.get("study_title") or t.get("title"),
                        "condition"  : t.get("conditions"),
                        "phase"      : t.get("phase"),
                    }
                    for t in result["retrieved_trials"]
                ],
                "recommendation_preview" : result["recommendation"][:300] + "..."
                    if len(result["recommendation"]) > 300
                    else result["recommendation"],
                "status"                 : "success",
                "timestamp"              : datetime.utcnow().isoformat(),
            })

            logger.info(f"  ✅ Done — {len(result['retrieved_trials'])} trials matched")

        except Exception as e:
            logger.error(f"Failed for {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"patient_id": patient_id, "error": str(e), "status": "failed"})
            summary.append({"patient_id": patient_id, "error": str(e), "status": "failed"})

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/rag_output.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(f"{OUTPUT_DIR}/rag_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in summary:
        if r.get("status") == "failed":
            print(f"\n❌ {r['patient_id']}: {r.get('error')}")
            continue
        print(f"\n👤 {r['patient_id']}")
        print(f"   Summary   : {r['patient_summary']}")
        print(f"   Candidates: {r['candidates_count']} → Reranked: {r['reranked_count']}")
        print(f"   Top trials:")
        for t in r["top_trials"]:
            print(f"     [{t['nct_number']}] {t['title']}")
        print(f"\n   🤖 Recommendation:\n   {r['recommendation_preview']}")
        print("-" * 60)

    logger.info(f"\n✅ test_results/rag_output.json")
    logger.info(f"✅ test_results/rag_summary.json")
    logger.info(f"✅ test_results/patients/<patient_id>.json")