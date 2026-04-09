"""
Run RAG pipeline locally for a patient uploaded via the TrialLink frontend.

Usage:
    cd /Users/vaishnavisarmalkar/Documents/trail-link
    source venv/bin/activate
    python tests/prompt_testing/run_patient.py
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sdk/patient_package")))

from models.rag_service import rag_pipeline_for_patient

# Patient uploaded via frontend — swap this ID to test different patients
PATIENT_ID ="d1693b81-62c4-4149-af0e-ee6d9a99d3a1"

print(f"\nRunning RAG pipeline for patient: {PATIENT_ID}")

result = rag_pipeline_for_patient(PATIENT_ID)

print("\n" + "=" * 60)
print("PATIENT SUMMARY")
print("=" * 60)
print(result["patient_summary"])

print("\n" + "=" * 60)
print(f"GUARDRAIL: {result['guardrail']['status'].upper()}")
if result["guardrail"].get("reason"):
    print(f"Reason:   {result['guardrail']['reason']}")
print("=" * 60)

trials = result.get("retrieved_trials", [])
print(f"\nTOP {len(trials)} TRIALS RECOMMENDED:")
for i, t in enumerate(trials, 1):
    print(f"  {i}. {t.get('nct_number','?')} — {t.get('study_title','')[:70]}")

print("\n" + "=" * 60)
print("MEDGEMMA RECOMMENDATION")
print("=" * 60)
print(result.get("recommendation") or "(no recommendation — patient was blocked/flagged)")
