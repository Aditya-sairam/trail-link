"""
RAG Evaluation — LLM-as-Judge (Gemini 2.5 Flash)
===================================================
Same as before but now:
  1. Writes summary.json to GCS after evaluation
  2. Logs structured metrics so Cloud Logging can parse them
  3. GCS write triggers Pub/Sub → Alert Cloud Function
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
from google.cloud import storage

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
EVAL_PROJECT_ID    = os.getenv("EVAL_PROJECT_ID", GCP_PROJECT_ID)
EVAL_REGION        = os.getenv("EVAL_REGION", "us-central1")
EVAL_MAX_PATIENTS  = int(os.getenv("EVAL_MAX_PATIENTS", "5"))
GEMINI_MODEL       = "gemini-2.5-flash"
EVAL_BUCKET        = os.getenv("EVAL_BUCKET", "")  # GCS bucket for eval results

OUTPUT_DIR         = "eval_results"
PATIENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "patients")

# ── Alert thresholds ───────────────────────────────────────────────────────────
THRESHOLD_AVG_SCORE        = 3.0   # alert if avg overall score drops below this
THRESHOLD_ELIGIBLE_PCT     = 0.30  # alert if <30% of verdicts are ELIGIBLE
THRESHOLD_NOT_ELIGIBLE_PCT = 0.70  # alert if >70% of verdicts are NOT ELIGIBLE
THRESHOLD_MIN_PATIENTS     = 3     # alert if fewer than 3 patients evaluated


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
      "reasoning": "One paragraph explaining why.",
      "key_matches": ["match 1", "match 2"],
      "key_concerns": ["concern 1"],
      "disqualifying_criterion": "exact exclusion criterion triggered, or empty string if none"
    }}
  ]
}}
"""

OVERALL_PROMPT = """\
You are a clinical research director reviewing a set of trial eligibility verdicts.

PATIENT PROFILE:
{patient_summary}

TRIAL VERDICTS:
{verdicts_block}

Respond ONLY with valid JSON. No markdown, no extra text.

{{
  "overall_score": 3,
  "top_trial": "NCT_ID_HERE",
  "summary": "2-3 sentence summary.",
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
# HELPERS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _clean(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""
    text = str(text).replace('"', "'").replace("\n", " ").replace("\r", " ")
    return text[:max_chars].strip()


def build_trials_block(trials: list[dict]) -> str:
    blocks = []
    for i, t in enumerate(trials, 1):
        nct     = t.get("nct_number") or t.get("_doc_id", "UNKNOWN")
        title   = _clean(t.get("study_title") or t.get("title", ""), 120)
        cond    = _clean(t.get("conditions", ""), 200)
        elig    = _clean(t.get("eligibility_criteria", ""), 800)
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
    lines = []
    for t in trials:
        lines.append(
            f"- {t['nct_id']}  [{t['verdict']}  {t['fitness_score']}/5]  "
            f"{t.get('disqualifying_criterion') or 'no disqualifier'}"
        )
    return "\n".join(lines)


def init_gemini() -> GenerativeModel:
    vertexai.init(project=EVAL_PROJECT_ID, location=EVAL_REGION)
    return GenerativeModel(GEMINI_MODEL)


def call_gemini(model: GenerativeModel, prompt: str, label: str) -> dict:
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
# EVALUATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_patient(
    patient_id: str,
    rag_result: dict,
    model: GenerativeModel,
) -> dict:
    patient_summary = rag_result["patient_summary"]
    trials          = rag_result.get("retrieved_trials", [])

    trials_block  = build_trials_block(trials)
    per_trial_raw = call_gemini(
        model,
        PER_TRIAL_PROMPT.format(
            patient_summary=patient_summary,
            trials_block=trials_block,
        ),
        label="Per-Trial Verdicts",
    )
    trial_verdicts = per_trial_raw.get("trials", [])

    verdicts_block = build_verdicts_block(trial_verdicts) if trial_verdicts else "No trial verdicts available."
    overall_raw    = call_gemini(
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
# THRESHOLD CHECK — runs after evaluation, before GCS upload
# ══════════════════════════════════════════════════════════════════════════════

def check_thresholds(summary: dict) -> list[dict]:
    """
    Check evaluation metrics against alert thresholds.
    Returns a list of breached thresholds — empty list means all good.
    Each breach is a dict with: metric, value, threshold, severity, message
    """
    breaches = []
    avg_score = summary.get("average_overall_score")
    verdicts  = summary.get("verdict_distribution", {})
    evaluated = summary["evaluation_run"]["patients_evaluated"]

    total_verdicts = sum(verdicts.values()) or 1  # avoid division by zero
    eligible_pct     = verdicts.get("ELIGIBLE", 0) / total_verdicts
    not_eligible_pct = verdicts.get("NOT ELIGIBLE", 0) / total_verdicts

    # Check 1 — average score
    if avg_score is not None and avg_score < THRESHOLD_AVG_SCORE:
        breaches.append({
            "metric"   : "average_overall_score",
            "value"    : avg_score,
            "threshold": THRESHOLD_AVG_SCORE,
            "severity" : "HIGH" if avg_score < 2.0 else "MEDIUM",
            "message"  : f"Average RAG quality score {avg_score}/5 is below threshold {THRESHOLD_AVG_SCORE}/5"
        })

    # Check 2 — eligible percentage too low
    if eligible_pct < THRESHOLD_ELIGIBLE_PCT:
        breaches.append({
            "metric"   : "eligible_percentage",
            "value"    : round(eligible_pct * 100, 1),
            "threshold": THRESHOLD_ELIGIBLE_PCT * 100,
            "severity" : "MEDIUM",
            "message"  : f"Only {eligible_pct*100:.1f}% of trials marked ELIGIBLE — below {THRESHOLD_ELIGIBLE_PCT*100}% threshold"
        })

    # Check 3 — not eligible percentage too high
    if not_eligible_pct > THRESHOLD_NOT_ELIGIBLE_PCT:
        breaches.append({
            "metric"   : "not_eligible_percentage",
            "value"    : round(not_eligible_pct * 100, 1),
            "threshold": THRESHOLD_NOT_ELIGIBLE_PCT * 100,
            "severity" : "HIGH",
            "message"  : f"{not_eligible_pct*100:.1f}% of trials marked NOT ELIGIBLE — vector search may be returning wrong trials"
        })

    # Check 4 — too few patients evaluated
    if evaluated < THRESHOLD_MIN_PATIENTS:
        breaches.append({
            "metric"   : "patients_evaluated",
            "value"    : evaluated,
            "threshold": THRESHOLD_MIN_PATIENTS,
            "severity" : "MEDIUM",
            "message"  : f"Only {evaluated} patients evaluated — pipeline may have partially failed"
        })

    return breaches


# ══════════════════════════════════════════════════════════════════════════════
# GCS UPLOAD — triggers Pub/Sub notification → Alert Cloud Function
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_gcs(bucket_name: str, data: dict, gcs_path: str) -> None:
    """Upload a dict as JSON to GCS."""
    try:
        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json"
        )
        logger.info(f"Uploaded to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        logger.error(f"GCS upload failed: {e}")


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

    os.makedirs(PATIENT_OUTPUT_DIR, exist_ok=True)

    model = init_gemini()

    db   = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    docs = list(db.collection("patients").limit(EVAL_MAX_PATIENTS).stream())

    if not docs:
        logger.error("No patients found in Firestore. Exiting.")
        return

    patient_evals: list[dict] = []
    failed: list[str]         = []

    for i, doc in enumerate(docs, 1):
        patient_id = doc.id
        logger.info(f"\n[{i}/{len(docs)}] Patient: {patient_id}")

        try:
            logger.info("  Running RAG pipeline...")
            rag_result = rag_pipeline_for_patient(patient_id)
        except Exception as e:
            logger.error(f"  RAG failed: {e}")
            failed.append(patient_id)
            continue

        try:
            eval_result = evaluate_patient(patient_id, rag_result, model)
        except Exception as e:
            logger.error(f"  Evaluation failed: {e}")
            failed.append(patient_id)
            continue

        # Print verdict table
        for t in eval_result.get("trial_verdicts", []):
            icon  = VERDICT_ICON.get(t.get("verdict", ""), "  ")
            score = t.get("fitness_score", "?")
            nct   = t.get("nct_id", "N/A")
            title = (t.get("title") or "")[:55]
            disq  = t.get("disqualifying_criterion", "")
            logger.info(f"  {icon} [{score}/5]  {nct}  {title}{'  — ' + disq if disq else ''}")

        overall = eval_result["overall"]
        logger.info(f"  Overall score: {overall.get('score')}/5  |  Top trial: {overall.get('top_trial', 'N/A')}")

        # Save locally
        out_path = os.path.join(PATIENT_OUTPUT_DIR, f"{patient_id}_eval.json")
        with open(out_path, "w") as f:
            json.dump(eval_result, f, indent=2, default=str)

        # Upload per-patient result to GCS
        if EVAL_BUCKET:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            upload_to_gcs(
                EVAL_BUCKET,
                eval_result,
                f"eval_results/{timestamp}/patients/{patient_id}_eval.json"
            )

        patient_evals.append(eval_result)

    # ── Build summary ──────────────────────────────────────────────────────────
    summary = build_summary(patient_evals)

    # ── Check thresholds ───────────────────────────────────────────────────────
    breaches = check_thresholds(summary)
    summary["alert_breaches"] = breaches
    summary["alert_status"]   = "ALERT" if breaches else "OK"

    # Log structured metrics so Cloud Logging can parse them
    # These log lines are picked up by Cloud Logging metric filters
    logger.info(f"EVAL_METRIC avg_overall_score={summary['average_overall_score']}")
    logger.info(f"EVAL_METRIC patients_evaluated={summary['evaluation_run']['patients_evaluated']}")
    logger.info(f"EVAL_METRIC alert_status={summary['alert_status']}")
    logger.info(f"EVAL_METRIC breaches_count={len(breaches)}")

    if breaches:
        for b in breaches:
            logger.warning(
                f"THRESHOLD_BREACH metric={b['metric']} "
                f"value={b['value']} threshold={b['threshold']} "
                f"severity={b['severity']} message={b['message']}"
            )

    # Save summary locally
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Upload summary to GCS — this triggers the Pub/Sub notification
    # which wakes up the Alert Cloud Function
    if EVAL_BUCKET:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        upload_to_gcs(
            EVAL_BUCKET,
            summary,
            f"eval_results/{timestamp}/summary.json"  # ← GCS notification watches this path
        )
        logger.info("Summary uploaded to GCS — alert function will be triggered if thresholds breached")

    # Print final summary
    logger.info("\n" + "=" * 65)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Patients evaluated : {summary['evaluation_run']['patients_evaluated']}")
    logger.info(f"Avg overall score  : {summary['average_overall_score']}/5")
    logger.info(f"Verdict breakdown  : {summary['verdict_distribution']}")
    logger.info(f"Alert status       : {summary['alert_status']}")
    if breaches:
        logger.warning(f"BREACHES DETECTED  : {len(breaches)}")
        for b in breaches:
            logger.warning(f"  [{b['severity']}] {b['message']}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()