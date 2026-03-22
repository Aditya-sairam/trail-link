# pipelines/dags/src/evaluate_rag.py

"""
RAG Evaluation — LLM-as-Judge (Gemini 2.5 Flash)
===================================================
Evaluates the 5 trials recommended by MedGemma for each patient.
Uses exactly 2 Gemini calls per patient:

  Call 1 — Per-Trial Verdict
      For each of the 5 reranked trials: checks patient profile vs
      eligibility criteria and produces a verdict + reasoning per trial.

  Call 2 — Overall Assessment
      Aggregate quality score + best trial recommendation + summary.

Env vars (same as rag_service.py, plus):
  EVAL_PROJECT_ID     GCP project with Vertex AI / Gemini enabled
  EVAL_REGION         defaults to us-central1
  EVAL_MAX_PATIENTS   how many patients to evaluate (default: 5)

Output:
  eval_results/
    patients/   {patient_id}_eval.json   — per-patient verdicts + scores
    summary.json                         — aggregate across all patients
"""

from __future__ import annotations

import os
import sys
import json
import logging
import re
from datetime import datetime, timezone

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

sys.path.insert(0, os.path.dirname(__file__))
from rag_service import (
    rag_pipeline_for_patient,
    GCP_PROJECT_ID,
    PATIENT_DB,
)
from google.cloud import firestore

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_PROJECT_ID   = os.getenv("EVAL_PROJECT_ID", "triallink-eval-001")
EVAL_REGION       = os.getenv("EVAL_REGION", "us-central1")
EVAL_MAX_PATIENTS = int(os.getenv("EVAL_MAX_PATIENTS", "5"))
GEMINI_MODEL      = "gemini-2.5-flash"

OUTPUT_DIR         = "eval_results"
PATIENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "patients")


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS  (2 total per patient)
# ══════════════════════════════════════════════════════════════════════════════

# Call 1 — verdict + reasoning for each of the 5 trials
PER_TRIAL_PROMPT = """\
You are a clinical trial eligibility reviewer.

PATIENT PROFILE:
{patient_summary}

You must evaluate the following 5 clinical trials against this patient profile.
For each trial check:
  - Does the patient meet ALL inclusion criteria? (age, sex, diagnosis, lab values, etc.)
  - Does the patient trigger ANY exclusion criterion? (prior procedures, medications, comorbidities, etc.)
  - Do current medications or allergies conflict with the trial?

Verdict options:
  ELIGIBLE       — patient meets all inclusion criteria and triggers no exclusion criteria
  NOT ELIGIBLE   — patient triggers at least one hard exclusion criterion (name it)
  NEEDS REVIEW   — eligibility is ambiguous or requires clinical judgment on borderline criteria

TRIALS TO EVALUATE:
{trials_block}

Respond ONLY with valid JSON. No markdown, no extra text, no newlines inside string values.

{{
  "trials": [
    {{
      "nct_id": "NCT_ID_HERE",
      "title": "trial title here",
      "verdict": "ELIGIBLE or NOT ELIGIBLE or NEEDS REVIEW",
      "fitness_score": 1,
      "reasoning": "One paragraph explaining why. Cite specific criteria and specific patient data.",
      "key_matches": ["match 1", "match 2"],
      "key_concerns": ["concern 1"],
      "disqualifying_criterion": "exact exclusion criterion triggered, or empty string if none"
    }}
  ]
}}
"""

# Call 2 — overall quality score + best pick
OVERALL_PROMPT = """\
You are a clinical research director reviewing a set of trial eligibility verdicts.

PATIENT PROFILE:
{patient_summary}

TRIAL VERDICTS:
{verdicts_block}

Based on these verdicts, provide an overall assessment.

Respond ONLY with valid JSON. No markdown, no extra text.

{{
  "overall_score": 3,
  "top_trial": "NCT_ID_HERE",
  "summary": "2-3 sentence summary of best match and overall recommendation quality.",
  "score_reasoning": "Why this overall score was given."
}}

Overall score guide (1-5):
  5 — Multiple strong eligible matches, reasoning is clinically precise
  4 — At least one clear eligible match, minor gaps in reasoning
  3 — Partial matches or borderline eligibility, some reasoning gaps
  2 — Mostly ineligible or significant reasoning errors
  1 — All ineligible or critically flawed assessment
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean(text: str, max_chars: int = 800) -> str:
    """Strip newlines and quotes from text going into JSON string values."""
    if not text:
        return ""
    text = str(text).replace('"', "'").replace("\n", " ").replace("\r", " ")
    return text[:max_chars].strip()


def build_trials_block(trials: list[dict]) -> str:
    """Format top-5 trials as a plain numbered block for the prompt."""
    blocks = []
    for i, t in enumerate(trials, 1):
        nct   = t.get("nct_number") or t.get("_doc_id", "UNKNOWN")
        title = _clean(t.get("study_title") or t.get("title", ""), 120)
        cond  = _clean(t.get("conditions", ""), 200)
        elig  = _clean(t.get("eligibility_criteria", ""), 800)
        age_min = t.get("min_age", "N/A")
        age_max = t.get("max_age", "N/A")
        sex     = t.get("sex", "N/A")
        status  = t.get("recruitment_status", "N/A")

        blocks.append(
            f"TRIAL {i}: {nct}\n"
            f"Title: {title}\n"
            f"Condition: {cond}\n"
            f"Age: {age_min} - {age_max}  |  Sex: {sex}  |  Status: {status}\n"
            f"Eligibility Criteria: {elig}"
        )
    return "\n\n".join(blocks)


def build_verdicts_block(trials: list[dict]) -> str:
    """Compact summary of per-trial verdicts for the overall prompt."""
    lines = []
    for t in trials:
        lines.append(
            f"- {t['nct_id']}  [{t['verdict']}  {t['fitness_score']}/5]  "
            f"{t.get('disqualifying_criterion') or 'no disqualifier'}"
        )
    return "\n".join(lines)


def init_gemini() -> GenerativeModel:
    vertexai.init(project=EVAL_PROJECT_ID, location=EVAL_REGION)
    model = GenerativeModel(GEMINI_MODEL)
    logger.info(f"Gemini judge: {GEMINI_MODEL} on {EVAL_PROJECT_ID}/{EVAL_REGION}")
    return model


def call_gemini(model: GenerativeModel, prompt: str, label: str) -> dict:
    """Send prompt to Gemini, parse JSON, return dict."""
    logger.info(f"  Gemini call → {label}")
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"  {label}: JSON parse error — {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"  {label}: Gemini call failed — {e}")
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_patient(
    patient_id: str,
    rag_result: dict,
    model: GenerativeModel,
) -> dict:
    """
    Run 2 Gemini calls for one patient:
      1. Per-trial verdicts for the 5 reranked trials
      2. Overall quality score + best recommendation
    """
    patient_summary = rag_result["patient_summary"]
    trials          = rag_result.get("retrieved_trials", [])

    # ── Call 1: per-trial verdicts ─────────────────────────────────────────────
    trials_block   = build_trials_block(trials)
    per_trial_raw  = call_gemini(
        model,
        PER_TRIAL_PROMPT.format(
            patient_summary=patient_summary,
            trials_block=trials_block,
        ),
        label="Per-Trial Verdicts",
    )
    trial_verdicts = per_trial_raw.get("trials", [])

    # ── Call 2: overall assessment ─────────────────────────────────────────────
    if trial_verdicts:
        verdicts_block = build_verdicts_block(trial_verdicts)
    else:
        verdicts_block = "No trial verdicts available."

    overall_raw = call_gemini(
        model,
        OVERALL_PROMPT.format(
            patient_summary=patient_summary,
            verdicts_block=verdicts_block,
        ),
        label="Overall Assessment",
    )

    return {
        "patient_id"     : patient_id,
        "patient_summary": patient_summary,
        "num_trials"     : len(trials),
        "trial_verdicts" : trial_verdicts,
        "overall": {
            "score"          : overall_raw.get("overall_score"),
            "top_trial"      : overall_raw.get("top_trial", ""),
            "summary"        : overall_raw.get("summary", ""),
            "score_reasoning": overall_raw.get("score_reasoning", ""),
            "error"          : overall_raw.get("error"),
        },
        "gemini_calls" : 2,
        "evaluated_at" : datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(patient_evals: list[dict]) -> dict:
    scores = [
        e["overall"]["score"]
        for e in patient_evals
        if isinstance(e["overall"].get("score"), (int, float))
    ]
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    verdict_counts: dict[str, int] = {}
    for e in patient_evals:
        for t in e.get("trial_verdicts", []):
            v = t.get("verdict", "UNKNOWN")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

    return {
        "evaluation_run": {
            "timestamp"         : datetime.now(timezone.utc).isoformat(),
            "gemini_model"      : GEMINI_MODEL,
            "eval_project"      : EVAL_PROJECT_ID,
            "patients_evaluated": len(patient_evals),
            "gemini_calls_total": len(patient_evals) * 2,
        },
        "average_overall_score": avg_score,
        "verdict_distribution" : verdict_counts,
        "top_recommendations"  : [
            {
                "patient_id": e["patient_id"],
                "top_trial" : e["overall"].get("top_trial", "N/A"),
                "score"     : e["overall"].get("score"),
            }
            for e in patient_evals
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

VERDICT_ICON = {
    "ELIGIBLE"    : "✅",
    "NOT ELIGIBLE": "❌",
    "NEEDS REVIEW": "⚠️ ",
}


def main() -> None:
    logger.info("=" * 65)
    logger.info("TrialLink RAG Evaluation — LLM-as-Judge (Gemini)")
    logger.info("=" * 65)
    logger.info(f"Eval project  : {EVAL_PROJECT_ID}")
    logger.info(f"Gemini model  : {GEMINI_MODEL}")
    logger.info(f"RAG project   : {GCP_PROJECT_ID}")
    logger.info(f"Max patients  : {EVAL_MAX_PATIENTS}")
    logger.info(f"Gemini calls  : {EVAL_MAX_PATIENTS * 2} total")
    logger.info("=" * 65)

    os.makedirs(PATIENT_OUTPUT_DIR, exist_ok=True)

    model = init_gemini()

    logger.info(f"Fetching up to {EVAL_MAX_PATIENTS} patients from Firestore ({PATIENT_DB})...")
    db   = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    docs = list(db.collection("patients").limit(EVAL_MAX_PATIENTS).stream())

    if not docs:
        logger.error("No patients found in Firestore. Exiting.")
        return

    logger.info(f"Found {len(docs)} patients to evaluate.")

    patient_evals: list[dict] = []
    failed: list[str] = []

    for i, doc in enumerate(docs, 1):
        patient_id = doc.id
        logger.info("")
        logger.info(f"[{i}/{len(docs)}] Patient: {patient_id}")
        logger.info("-" * 55)

        # Run RAG
        try:
            logger.info("  Running RAG pipeline...")
            rag_result = rag_pipeline_for_patient(patient_id)
            logger.info(
                f"  RAG complete — "
                f"{len(rag_result.get('candidates_before_rerank', []))} candidates, "
                f"{len(rag_result.get('retrieved_trials', []))} reranked"
            )
        except Exception as e:
            logger.error(f"  RAG failed: {e}")
            failed.append(patient_id)
            continue

        # Evaluate
        try:
            eval_result = evaluate_patient(patient_id, rag_result, model)
        except Exception as e:
            logger.error(f"  Evaluation failed: {e}")
            failed.append(patient_id)
            continue

        # ── Print per-trial table ──────────────────────────────────────────────
        logger.info("  ── Trial Verdicts ────────────────────────────────────")
        for t in eval_result.get("trial_verdicts", []):
            icon  = VERDICT_ICON.get(t.get("verdict", ""), "  ")
            score = t.get("fitness_score", "?")
            nct   = t.get("nct_id", "N/A")
            title = (t.get("title") or "")[:55]
            disq  = t.get("disqualifying_criterion", "")
            disq_str = f"  — {disq}" if disq else ""
            logger.info(f"  {icon} [{score}/5]  {nct}  {title}{disq_str}")

        overall = eval_result["overall"]
        logger.info("  ─────────────────────────────────────────────────────")
        logger.info(f"  Overall score   : {overall.get('score')}/5")
        logger.info(f"  Top trial       : {overall.get('top_trial', 'N/A')}")
        logger.info(f"  Summary         : {overall.get('summary', '')}")

        # Save
        out_path = os.path.join(PATIENT_OUTPUT_DIR, f"{patient_id}_eval.json")
        with open(out_path, "w") as f:
            json.dump(eval_result, f, indent=2, default=str)
        logger.info(f"  Saved → {out_path}")

        patient_evals.append(eval_result)

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = build_summary(patient_evals)
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("")
    logger.info("=" * 65)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Patients evaluated  : {summary['evaluation_run']['patients_evaluated']}")
    logger.info(f"Patients failed     : {len(failed)}")
    logger.info(f"Avg overall score   : {summary['average_overall_score']}/5")
    logger.info(f"Verdict breakdown   : {summary['verdict_distribution']}")
    logger.info("─" * 65)
    logger.info("Top Recommendations:")
    for rec in summary["top_recommendations"]:
        logger.info(f"  {rec['patient_id'][:8]}…  →  {rec['top_trial']}  (score {rec['score']}/5)")
    logger.info(f"Summary saved       → {summary_path}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
