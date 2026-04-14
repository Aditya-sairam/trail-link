# pipelines/dags/src/rag_service.py

"""
RAG Service - TrialLink Cloud Function
======================================
Pub/Sub-triggered Cloud Function that runs the full guarded RAG pipeline for a patient.

Pipeline:
  0.  Input guardrails
      0A. PII redaction
      0B. Structural validation
      0C. LLM semantic input guardrail
  1A. LLM clinical context enrichment       (Gemini — semantic condition detection)
  1B. Rule-based condition detection         (fallback)
  1C. Enriched retrieval query builder
  1.  Embed enriched query                   (Vertex AI text-embedding-005)
  2.  Query Vertex AI Vector Search          (condition-scoped)
  3.  Fetch trials from Firestore            (disease field validated)
  3.5 Rerank                                 (Vertex AI Ranking API — multi-condition aware)
  3.6 Subtype filter                         (disease_type field + rule-based + LLM exclusions)
  4.  Generate recommendation                (Gemini 2.5 Flash, temp=0.6)
  4B. MedGemma as second-opinion judge       (chatCompletions format)
  5A. Policy checks
  5B. Grounding checks
  5C. LLM output guardrail
  5E. Filter to consensus eligible/borderline

Trigger:
    Pub/Sub CloudEvent with payload:
    { "patient_id": "uuid-here" }

Returned pipeline result:
    {
        "patient_summary": str,
        "detected_conditions": list[str],
        "is_dual_condition": bool,
        "candidates_before_rerank": list[dict],
        "retrieved_trials": list[dict],
        "recommendation": str,
        "medgemma_judgment": str,
        "guardrail": {
            "status": "passed" | "blocked" | "flagged",
            ...
        }
    }
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import requests
import google.auth
from datetime import datetime
from typing import Any

try:
    import functions_framework
except ImportError:
    functions_framework = None

import vertexai
from google.auth.transport.requests import Request
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import firestore
from vertexai.generative_models import GenerationConfig, GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID",  "triallink-eval-001")
MODEL_PROJECT_ID = os.getenv("MODEL_PROJECT_ID", "mlops-triallink")
GCP_REGION      = os.getenv("GCP_REGION",      "us-central1")

VECTOR_SEARCH_ENDPOINT_ID = os.getenv(
    "VECTOR_SEARCH_ENDPOINT_ID",
    "projects/408416535077/locations/us-central1/indexEndpoints/4231811348100546560",
)
DEPLOYED_INDEX_ID = os.getenv("DEPLOYED_INDEX_ID", "clinical_trials_dev")

<<<<<<< Updated upstream
FIRESTORE_DB            = os.getenv("FIRESTORE_DATABASE",   "clinical-trials-db")
PATIENT_DB              = os.getenv("PATIENT_DB",           "patient-db-dev")
TRAIL_SUGGESTIONS_STORE = os.getenv("TRAIL_SUGGESTIONS_STORE", "")

MEDGEMMA_ENDPOINT_ID = os.getenv(
    "MEDGEMMA_ENDPOINT_ID",
    "mg-endpoint-1284414a-76e1-4adf-9478-d8ca475dedd9",
=======
FIRESTORE_DB = os.getenv("FIRESTORE_DATABASE", "clinical-trials-db")
PATIENT_DB = os.getenv("PATIENT_DB", "patient-db-dev")
TRAIL_SUGGESTIONS_STORE = os.getenv("TRAIL_SUGGESTIONS_STORE", "clinical-trials-suggestions-db")

MEDGEMMA_ENDPOINT_ID = os.getenv(
    "MEDGEMMA_ENDPOINT_ID",
    "mg-endpoint-5cf27b97-44c1-4130-9a77-40ce2c8cd79b",
>>>>>>> Stashed changes
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "15"))
<<<<<<< Updated upstream
RERANK_TOP_K    = int(os.getenv("RERANK_TOP_K",    "5"))
CONDITIONS      = ["diabetes", "breast_cancer"]
=======
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
CONDITIONS = ["diabetes", "breast_cancer"]
>>>>>>> Stashed changes

GUARDRAIL_MODEL             = os.getenv("GUARDRAIL_MODEL", "gemini-2.5-flash")
ENABLE_INPUT_LLM_GUARDRAIL  = os.getenv("ENABLE_INPUT_LLM_GUARDRAIL",  "true").lower() == "true"
ENABLE_OUTPUT_LLM_GUARDRAIL = os.getenv("ENABLE_OUTPUT_LLM_GUARDRAIL", "true").lower() == "true"
<<<<<<< Updated upstream
MAX_INPUT_CHARS  = int(os.getenv("MAX_INPUT_CHARS",  "12000"))
=======
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
>>>>>>> Stashed changes
MAX_OUTPUT_CHARS = int(os.getenv("MAX_OUTPUT_CHARS", "40000"))

DISCLAIMER_TEXT = (
    "Disclaimer: This right here! AI-generated output is for informational purposes only and "
    "must not be used as medical advice, diagnosis, prescribing guidance, or treatment "
    "recommendation. Please consult your doctor or healthcare provider before making "
    "any clinical decisions or enrolling in a clinical trial."
)

SUPPORTED_CONDITIONS = {"diabetes", "breast cancer", "breast_cancer"}

BANNED_OUTPUT_PATTERNS = [
<<<<<<< Updated upstream
=======
    # Removed the generic mg/ml pattern — it fires on patient's own medication names
    # (e.g. "Metformin 500 MG") which MedGemma legitimately cites.
    # Prescriptive action verbs remain:
>>>>>>> Stashed changes
    r"\btake\s+\d+",
    r"\bprescribe\b",
    r"\bstart medication\b",
    r"\bincrease medication\b",
    r"\breduce medication\b",
    r"\byou should take\b",
    r"\bi recommend starting\b",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"reveal system prompt",
    r"show hidden prompt",
    r"bypass safety",
    r"developer message",
    r"system prompt",
]

# ── Init Vertex AI ────────────────────────────────────────────────────────────
vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)


# ══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def safe_guardrail_response(
    reason: str,
    patient_summary: str = "",
    guardrail_stage: str = "unknown",
    pii_hits: dict[str, int] | None = None,
) -> dict:
    return {
        "patient_summary":          patient_summary,
        "detected_conditions":      [],
        "is_dual_condition":        False,
        "candidates_before_rerank": [],
        "retrieved_trials":         [],
        "recommendation": (
            f"Guardrail triggered at {guardrail_stage}: {reason}. "
            "TrialLink currently supports informational clinical trial matching only "
            "for diabetes and breast cancer use cases.\n\n"
            f"{DISCLAIMER_TEXT}"
        ),
        "guardrail": {
            "status":              "blocked",
            "stage":               guardrail_stage,
            "reason":              reason,
            "pii_hits":            pii_hits or {},
            "flag_reasons":        [],
            "llm_input_judgment":  None,
            "llm_output_judgment": None,
        },
    }


def redact_basic_pii(text: str) -> tuple[str, dict[str, int]]:
    pii_hits = {"email": 0, "phone": 0, "ssn": 0, "dob": 0}

    email_pattern = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
    phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ssn_pattern   = r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
    dob_pattern   = r"\b(?:dob|date of birth)\s*[:\-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"

    pii_hits["email"] = len(re.findall(email_pattern, text, flags=re.IGNORECASE))
    pii_hits["phone"] = len(re.findall(phone_pattern, text, flags=re.IGNORECASE))
    pii_hits["ssn"]   = len(re.findall(ssn_pattern,   text, flags=re.IGNORECASE))
    pii_hits["dob"]   = len(re.findall(dob_pattern,   text, flags=re.IGNORECASE))

    text = re.sub(email_pattern, "[REDACTED_EMAIL]", text, flags=re.IGNORECASE)
    text = re.sub(phone_pattern, "[REDACTED_PHONE]", text, flags=re.IGNORECASE)
    text = re.sub(ssn_pattern,   "[REDACTED_SSN]",   text, flags=re.IGNORECASE)
    text = re.sub(dob_pattern,   "[REDACTED_DOB]",   text, flags=re.IGNORECASE)

    return text, pii_hits


def validate_input_structure(patient_summary: str) -> tuple[bool, str]:
    if not patient_summary or not patient_summary.strip():
        return False, "Input is empty"
    if len(patient_summary) > MAX_INPUT_CHARS:
        return False, f"Input exceeds max length of {MAX_INPUT_CHARS} characters"

    lowered = patient_summary.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, "Prompt injection or instruction override pattern detected"

    has_age_signal       = bool(re.search(r"\b(age|years old|yo|min age|max age)\b", lowered))
    has_sex_signal       = bool(re.search(r"\b(male|female|sex)\b", lowered))
    has_condition_signal = bool(re.search(r"\b(diagnosis|diagnosed|condition|disease|history of)\b", lowered))
    has_clinical_signal  = bool(re.search(
        r"\b(medication|allergy|stage|a1c|hba1c|glucose|insulin|metformin|tumor|bp|blood pressure|bmi|eligibility|trial)\b",
        lowered,
    ))

    score = sum([has_age_signal, has_sex_signal, has_condition_signal, has_clinical_signal])
    if score < 2:
        return False, "Input lacks sufficient clinical structure"
    return True, "Input structure passed"


def get_guardrail_model() -> GenerativeModel:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    return GenerativeModel(GUARDRAIL_MODEL)


def extract_json_block(text: str) -> dict[str, Any]:
    if text is None:
        raise ValueError("Model response text is None")
    text = text.strip()
    if not text:
        raise ValueError("Model response text is empty")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model response: {text[:500]}")
    return json.loads(text[start:end + 1])


def llm_input_guardrail(patient_summary: str) -> dict[str, Any]:
    prompt = f"""
You are a safety validator for a clinical trial matching system.

Classify whether the following input is appropriate for TrialLink.

Input:
{patient_summary}

Return strict JSON with this schema only:
{{
  "is_valid": true,
  "category": "valid_supported_clinical_summary",
  "supported_conditions": ["diabetes", "breast_cancer"],
  "reason": "short explanation",
  "risk_flags": ["none"]
}}

supported_conditions must be a LIST — include all supported conditions present.
Supported values: "diabetes", "breast_cancer". Can include both for dual-condition patients.

Allowed categories:
- valid_supported_clinical_summary
- valid_but_unsupported_condition
- non_medical_or_irrelevant
- unsafe_or_prompt_injection
- missing_required_clinical_information

Return JSON only.
"""
    model    = get_guardrail_model()
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    raw_text = getattr(response, "text", None)
    logger.info(f"Raw LLM input guardrail response: {repr(raw_text)[:800]}")
    result = extract_json_block(raw_text)

    # Handle both old string and new list format
    supported = result.get("supported_conditions", result.get("supported_condition", []))
    if isinstance(supported, str):
        supported = [supported] if supported not in ("unsupported", "unknown") else []

    return {
        "is_valid":             bool(result.get("is_valid", False)),
        "category":             str(result.get("category", "unsafe_or_prompt_injection")),
        "supported_conditions": [c for c in supported if c in CONDITIONS],
        "reason":               str(result.get("reason", "No reason provided")),
        "risk_flags":           result.get("risk_flags", []),
    }


def validate_retrieved_trials(trials: list[dict]) -> tuple[bool, str]:
    if not trials:
        return False, "No trials retrieved"
    for trial in trials:
        if not (trial.get("nct_number") or trial.get("_doc_id")):
            return False, "Retrieved trial missing identifier"
    return True, "Retrieved trials passed"


def validate_trials_scope(retrieved_trials: list[dict]) -> tuple[bool, str]:
    if not retrieved_trials:
        return False, "No reranked trials available"
    for trial in retrieved_trials:
        condition_text = str(trial.get("conditions", "")).lower()
        disease_text   = str(trial.get("disease",    "")).lower()
        combined       = f"{condition_text} {disease_text}"
        if not any(cond in combined for cond in SUPPORTED_CONDITIONS):
            return False, "Retrieved trial outside supported disease scope"
    return True, "Trial scope passed"


def append_disclaimer(recommendation: str) -> str:
    lowered = recommendation.lower()
    if "consult your doctor" in lowered or "healthcare provider" in lowered:
        return recommendation
    return recommendation.strip() + "\n\n" + DISCLAIMER_TEXT


def policy_check_output(recommendation: str) -> tuple[bool, str]:
    if not recommendation or not recommendation.strip():
        return False, "Recommendation is empty"
    if len(recommendation) > MAX_OUTPUT_CHARS:
        return False, f"Recommendation exceeds max length of {MAX_OUTPUT_CHARS} characters"
    lowered = recommendation.lower()
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern, lowered):
            return False, "Dosage or prescribing style advice detected"
    if "consult your doctor" not in lowered and "healthcare provider" not in lowered:
        return False, "Missing required medical disclaimer"
    return True, "Output policy passed"


def grounding_check_output(recommendation: str, retrieved_trials: list[dict]) -> tuple[bool, str]:
    allowed_trial_ids = {
        str(t.get("nct_number", "")).strip()
        for t in retrieved_trials if t.get("nct_number")
    }
    mentioned_ids = set(re.findall(r"\bNCT\d{8}\b", recommendation))
    if mentioned_ids and not mentioned_ids.issubset(allowed_trial_ids):
        return False, "Recommendation references trials outside retrieved context"

    allowed_titles = {
        str(t.get("study_title") or t.get("title", "")).strip().lower()
        for t in retrieved_trials if (t.get("study_title") or t.get("title"))
    }
    hallucination_risk_count = 0
    for line in recommendation.splitlines():
        line_lower = line.strip().lower()
        if line_lower.startswith("**trial") and not any(title in line_lower for title in allowed_titles):
            hallucination_risk_count += 1
    if hallucination_risk_count > 0 and not mentioned_ids:
        return False, "Recommendation may contain ungrounded trial references"
    return True, "Output grounding passed"


def llm_output_guardrail(
    patient_summary: str,
    recommendation: str,
    retrieved_trials: list[dict],
) -> dict[str, Any]:
    trial_ids = [str(t.get("nct_number", "")) for t in retrieved_trials if t.get("nct_number")]
    prompt = f"""
You are a safety and grounding judge for a clinical trial matching system.

Patient Summary:
{patient_summary}

Allowed Retrieved Trial IDs:
{trial_ids}

Generated Recommendation:
{recommendation}

Return strict JSON with this schema only:
{{
  "is_safe": true,
  "is_grounded": true,
  "category": "pass" | "unsafe_medical_advice" | "hallucinated_trials" | "unsupported_claims" | "missing_disclaimer",
  "reason": "short explanation"
}}

Judging criteria:
1. Unsafe if it gives medication dosage, prescribing advice, or treatment decisions.
2. Ungrounded if it mentions trials outside the allowed trial IDs or invents evidence.
3. Fail if disclaimer is missing.
4. Pass only if safe and grounded.

Return JSON only.
"""
    model    = get_guardrail_model()
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
        ),
    )
    result = extract_json_block(response.text)
    return {
        "is_safe":     bool(result.get("is_safe",     False)),
        "is_grounded": bool(result.get("is_grounded", False)),
        "category":    str(result.get("category",     "unsupported_claims")),
        "reason":      str(result.get("reason",       "No reason provided")),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLOUD FUNCTION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _run_rag_pipeline_impl(cloud_event) -> None:
    db   = firestore.Client(project=GCP_PROJECT_ID, database=TRAIL_SUGGESTIONS_STORE)
    data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    message    = json.loads(data)
    patient_id = message.get("patient_id")
    if not patient_id:
        logger.error("No patient_id in Pub/Sub message")
        return

    logger.info(f"Running RAG pipeline for patient: {patient_id}")
    try:
        result         = rag_pipeline_for_patient(patient_id)
        recommendation = result["recommendation"]
        guardrail_info = result.get("guardrail", {})
        status         = "completed"

        if guardrail_info.get("status") == "blocked" or recommendation.startswith("Guardrail triggered at"):
            status = "guardrail_blocked"
        elif guardrail_info.get("status") == "flagged" or "did not pass" in recommendation.lower():
            status = "guardrail_flagged"

        db.collection("trial_suggestions").document(patient_id).set({
            "status":                   status,
            "patient_id":               patient_id,
            "patient_summary":          result.get("patient_summary", ""),
            "detected_conditions":      result.get("detected_conditions", []),
            "is_dual_condition":        result.get("is_dual_condition", False),
            "recommendation":           recommendation,
            "retrieved_trials":         result.get("retrieved_trials", []),
            "candidates_before_rerank": result.get("candidates_before_rerank", []),
            "guardrail":                guardrail_info,
            "generated_at":             datetime.utcnow().isoformat(),
        })
        logger.info(f"Pipeline complete for patient {patient_id} with status {status}")

    except Exception as e:
        logger.exception(f"RAG pipeline failed for patient {patient_id}: {e}")
        db.collection("trial_suggestions").document(patient_id).set({
            "status":       "failed",
            "patient_id":   patient_id,
            "error":        str(e),
            "generated_at": datetime.utcnow().isoformat(),
        })
        raise


if functions_framework is not None:
    @functions_framework.cloud_event
    def run_rag_pipeline(cloud_event):
        return _run_rag_pipeline_impl(cloud_event)
else:
    def run_rag_pipeline(cloud_event):
        return _run_rag_pipeline_impl(cloud_event)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EMBED TEXT
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    try:
        vertexai.init(project=GCP_PROJECT_ID, location=os.getenv("GCP_REGION", "us-central1"))
        model      = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        inputs     = [TextEmbeddingInput(text=text, task_type=task_type)]
        embeddings = model.get_embeddings(inputs)
        return embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def trial_to_text(trial: dict) -> str:
    def _get(*keys) -> str:
        for k in keys:
            v = trial.get(k)
            if v and str(v).strip() not in ("", "nan", "None"):
                return str(v).strip()
        return ""
    return (
        f"Title: {_get('study_title', 'title')}. "
        f"Condition: {_get('conditions')}. "
        f"Disease: {_get('disease')}. "
        f"Disease Type: {_get('disease_type')}. "
        f"Keywords: {_get('keywords')}. "
        f"Phase: {_get('phase')}. "
        f"Status: {_get('recruitment_status')}. "
        f"Interventions: {_get('interventions')}. "
        f"Eligibility: {_get('eligibility_criteria')}. "
        f"Summary: {_get('brief_summary')}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1A — LLM CLINICAL CONTEXT ENRICHER
# ══════════════════════════════════════════════════════════════════════════════

def enrich_patient_context(patient_summary: str) -> dict:
    """
    Use Gemini to extract structured clinical context from patient summary.
    Understands semantic relationships e.g.:
      prediabetes → T2DM prevention trials (not T1DM, not active T2DM treatment)
      metabolic syndrome → insulin resistance trials
      ER+ breast cancer → hormone receptor trials (not TNBC)
    Also handles dual-condition patients (breast cancer + diabetes).
    """
    prompt = f"""
You are a clinical informatics specialist. Analyze this patient summary and
extract structured information for clinical trial matching.

PATIENT SUMMARY:
{patient_summary}

Return strict JSON only:
{{
  "condition_categories": ["diabetes" and/or "breast_cancer" — only include supported conditions present],
  "is_dual_condition": true or false,
  "trial_search_terms": [
    "10-15 specific medical terms to search for relevant trials",
    "include exact diagnosis AND semantically related terms",
    "for prediabetes include: prediabetes prevention, diabetes prevention program, glucose intolerance, lifestyle intervention, metabolic syndrome, insulin resistance, T2DM risk reduction",
    "for ER+ breast cancer include: hormone receptor positive breast cancer, endocrine therapy, aromatase inhibitor, ER+ HER2- early stage"
  ],
  "exclude_trial_types": [
    "trial types to EXCLUDE for this specific patient",
    "for prediabetes: Type 1 diabetes trials, confirmed T2DM treatment trials requiring HbA1c above 6.5%",
    "for ER+ breast cancer: triple negative breast cancer trials, HER2+ only trials"
  ],
  "metabolic_profile": {{
    "hba1c": "value% or null",
    "bmi": "value or null",
    "age": "value or null",
    "stage": "cancer stage or null"
  }},
  "patient_eligibility_context": "2-3 sentence clinical summary of what makes this patient a good candidate for trials and what trial types are most relevant"
}}

Rules:
- condition_categories MUST only contain 'diabetes' or 'breast_cancer'
- Include BOTH if patient has both conditions
- For prediabetes: search T2DM PREVENTION trials, EXCLUDE T1DM and active T2DM treatment trials
- For metabolic syndrome without diabetes diagnosis: include diabetes prevention and metabolic trials
- For breast cancer: specify subtype search terms (ER+, HER2, TNBC) based on what is documented
- If condition is COPD, heart failure, Alzheimer's etc: condition_categories = []

Return JSON only. No explanation outside JSON.
"""
    model = get_guardrail_model()
    try:
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        result = extract_json_block(response.text)
        logger.info(
            f"Clinical context: conditions={result.get('condition_categories')}, "
            f"dual={result.get('is_dual_condition')}, "
            f"terms={result.get('trial_search_terms', [])[:3]}"
        )
        return result
    except Exception as e:
        logger.warning(f"Clinical context enrichment failed: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1B — RULE-BASED CONDITION DETECTOR (fallback)
# ══════════════════════════════════════════════════════════════════════════════

CONDITION_SIGNALS = {
    "breast_cancer": {
        "strong": [
            "malignant neoplasm of breast", "breast cancer", "breast carcinoma",
            "ductal carcinoma", "her2", "estrogen receptor positive",
            "triple negative", "dcis", "lobular carcinoma", "breast neoplasm",
        ],
        "weak": ["breast", "mammogram", "lumpectomy", "mastectomy"],
    },
    "diabetes": {
        "strong": [
            "diabetes mellitus type 2", "type 2 diabetes", "prediabetes",
            "prediabetic", "hba1c", "insulin resistance", "metabolic syndrome",
            "hyperglycemia", "t2dm", "diabetes mellitus",
        ],
        "weak": ["glucose", "insulin", "glycemic", "a1c", "obesity"],
    },
}


def detect_patient_conditions(patient_summary: str) -> list[str]:
    """Rule-based fallback condition detection using strong/weak signals."""
    summary_lower = patient_summary.lower()
    detected = []
    for condition, signals in CONDITION_SIGNALS.items():
        strong_match = any(s in summary_lower for s in signals["strong"])
        weak_count   = sum(1 for s in signals["weak"] if s in summary_lower)
        if strong_match or weak_count >= 2:
            detected.append(condition)
            logger.info(f"Rule-based detected: {condition} (strong={strong_match}, weak={weak_count})")
    if not detected:
        logger.warning("No condition detected — falling back to all conditions")
        detected = CONDITIONS
    return detected


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1C — ENRICHED RETRIEVAL QUERY BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_retrieval_query(
    patient_summary: str,
    conditions: list[str],
    clinical_context: dict | None = None,
) -> str:
    """
    Build a clinically enriched retrieval query.
    Uses LLM-extracted search terms when available,
    falls back to rule-based extraction.
    """
    query_parts = []

    if clinical_context and clinical_context.get("trial_search_terms"):
        search_terms = clinical_context["trial_search_terms"]
        query_parts.extend(search_terms[:12])

        metabolic = clinical_context.get("metabolic_profile", {})
        if metabolic.get("hba1c"):
            query_parts.append(f"HbA1c {metabolic['hba1c']}")
        if metabolic.get("bmi"):
            query_parts.append(f"BMI {metabolic['bmi']}")
        if metabolic.get("stage"):
            query_parts.append(f"cancer stage {metabolic['stage']}")

        context_text = clinical_context.get("patient_eligibility_context", "")
        if context_text:
            query_parts.append(context_text)

        logger.info(f"Using LLM-enriched query with {len(search_terms)} terms")

    else:
        logger.info("Using rule-based query enrichment (LLM context unavailable)")
        summary_lower = patient_summary.lower()

        condition_terms = {
            "breast_cancer": "breast cancer clinical trial oncology",
            "diabetes":      "diabetes clinical trial glycemic control glucose",
        }
        for cond in conditions:
            query_parts.append(condition_terms.get(cond, cond))

        hba1c_match = re.search(r"hba1c[^\d]*(\d+\.?\d*)\s*%", summary_lower)
        if hba1c_match:
            query_parts.append(f"HbA1c {hba1c_match.group(1)}%")

        bmi_match = re.search(r"bmi[^\d]*(\d+\.?\d*)", summary_lower)
        if bmi_match:
            query_parts.append(f"BMI {bmi_match.group(1)}")

        age_match = re.search(r"(\d+)-year-old", summary_lower)
        if age_match:
            query_parts.append(f"adult age {age_match.group(1)}")

        comorbidity_map = {
            "hypertension":       "hypertension blood pressure",
            "obesity":            "obesity overweight BMI",
            "prediabetes":        "prediabetes prevention T2DM risk lifestyle intervention",
            "metabolic syndrome": "metabolic syndrome insulin resistance",
            "breast cancer":      "breast cancer oncology",
            "her2":               "HER2 breast cancer",
            "estrogen receptor":  "hormone receptor positive breast cancer ER+",
            "triple negative":    "triple negative breast cancer TNBC",
        }
        for keyword, term in comorbidity_map.items():
            if keyword in summary_lower:
                query_parts.append(term)

        if "female" in summary_lower:
            query_parts.append("female women")

    enriched_query = " ".join(query_parts)
    logger.info(f"Enriched retrieval query ({len(enriched_query)} chars): {enriched_query[:300]}")
    return enriched_query


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QUERY VERTEX AI VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def query_vector_search(
    patient_embedding: list[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, api_transport="grpc")
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=VECTOR_SEARCH_ENDPOINT_ID
    )

    fetch_k = top_k * 3
    logger.info("Querying Vertex AI Vector Search...")
    results = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[patient_embedding],
        num_neighbors=fetch_k,
    )

    matches       = results[0]
    seen_nct_ids: dict[str, float] = {}
    for match in matches:
        nct_id = match.id.rsplit("_", 1)[0]
        score  = match.distance
        if nct_id and (nct_id not in seen_nct_ids or score < seen_nct_ids[nct_id]):
            seen_nct_ids[nct_id] = score

    sorted_trials = sorted(seen_nct_ids.items(), key=lambda x: x[1])
    top_nct_ids   = [nct_id for nct_id, _ in sorted_trials[:top_k]]
    logger.info(f"Vector search -> top {top_k}: {top_nct_ids}")
    return top_nct_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FETCH MATCHED TRIALS FROM FIRESTORE (condition + disease field aware)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_trials_from_firestore(
    nct_ids: list[str],
    target_conditions: list[str] | None = None,
) -> list[dict]:
    """
    Fetch trial documents from Firestore.
    - Only searches collections relevant to detected patient conditions
    - Validates each document's 'disease' field matches target conditions
    - Uses engineered 'disease' and 'disease_type' fields for hard filtering
    """
    db       = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB)
    trials   = []
    seen_ids = set()

    collections_to_search = target_conditions if target_conditions else CONDITIONS
    logger.info(f"Fetching from collections: {collections_to_search}")

    for condition in collections_to_search:
        collection_name = f"clinical_trials_{condition}"
        for nct_id in nct_ids:
            if nct_id in seen_ids:
                continue
            try:
                doc = db.collection(collection_name).document(nct_id).get()
                if not doc.exists:
                    continue

                trial         = doc.to_dict()
                trial_disease = str(trial.get("disease", "")).lower().strip()

                # Hard filter using engineered 'disease' field
                if trial_disease and trial_disease not in collections_to_search:
                    logger.info(
                        f"disease field filter: skipping {nct_id} "
                        f"(disease='{trial_disease}', target={collections_to_search})"
                    )
                    continue

                trial["_doc_id"] = doc.id
                trials.append(trial)
                seen_ids.add(nct_id)

            except Exception as e:
                logger.warning(f"Could not fetch {nct_id} from {collection_name}: {e}")

    logger.info(f"Fetched {len(trials)} trials from Firestore")
    return trials


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3.5 — RERANK (multi-condition aware)
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary: str,
    trials: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """Single-condition rerank using Vertex AI Ranking API."""
    try:
        client         = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project=GCP_PROJECT_ID,
            location=GCP_REGION,
            ranking_config="default_ranking_config",
        )
        records = [
            discoveryengine.RankingRecord(
                id=str(t.get("nct_number") or t.get("_doc_id", "")),
                title=str(t.get("study_title") or t.get("title", "")),
                content=trial_to_text(t),
            )
            for t in trials
            if t.get("nct_number") or t.get("_doc_id")
        ]
        request  = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model="semantic-ranker-512@latest",
            top_n=top_k,
            query=patient_summary,
            records=records,
        )
        response  = client.rank(request=request)
        trial_map = {str(t.get("nct_number") or t.get("_doc_id", "")): t for t in trials}
        reranked  = [
            trial_map[record.id]
            for record in response.records
            if record.id in trial_map
        ]
        logger.info(f"Reranked {len(trials)} -> top {len(reranked)}")
        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to vector search order.")
        return trials[:top_k]


def rerank_trials_multi_condition(
    patient_summary: str,
    trials: list[dict],
    detected_conditions: list[str],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """
    For dual-condition patients: rerank separately per condition
    then interleave so both conditions are represented in top-K.
    For single-condition patients: falls through to standard rerank.
    """
    if len(detected_conditions) <= 1:
        return rerank_trials(patient_summary, trials, top_k)

    # Split trials by disease field
    condition_buckets: dict[str, list[dict]] = {cond: [] for cond in detected_conditions}
    for trial in trials:
        disease = str(trial.get("disease", "")).lower().strip()
        if disease in condition_buckets:
            condition_buckets[disease].append(trial)

    logger.info(
        f"Dual-condition split: "
        f"{[(c, len(t)) for c, t in condition_buckets.items()]}"
    )

    per_condition_top_k = max(2, top_k // len(detected_conditions))
    reranked_per_condition: dict[str, list[dict]] = {}

    for cond, cond_trials in condition_buckets.items():
        if not cond_trials:
            continue
        reranked = rerank_trials(patient_summary, cond_trials, top_k=per_condition_top_k)
        reranked_per_condition[cond] = reranked
        logger.info(f"Reranked {cond}: {len(reranked)} trials")

    # Interleave — round robin
    merged  = []
    max_len = max((len(t) for t in reranked_per_condition.values()), default=0)
    for i in range(max_len):
        for cond in detected_conditions:
            cond_results = reranked_per_condition.get(cond, [])
            if i < len(cond_results):
                merged.append(cond_results[i])
        if len(merged) >= top_k:
            break

    logger.info(f"Multi-condition merged: {len(merged)} trials")
    return merged[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3.6 — CONDITION-SUBTYPE FILTER
# ══════════════════════════════════════════════════════════════════════════════

_SUBTYPE_FILTER_RULES = [
    # ── Diabetes ──────────────────────────────────────────────────────────────
    {
        "patient_has":     ["diabetes mellitus type 2", "type 2 diabetes", "t2dm", "t2d",
                            "prediabetes", "prediabetic", "metabolic syndrome"],
        "exclude_if_only": ["type 1 diabetes", "type 1 diabetes mellitus", "t1dm",
                            "cystic fibrosis-related diabetes", "cfrd"],
        "must_not_contain_patient_type": ["type 2", "t2", "prediabetes"],
    },
    {
        "patient_has":     ["diabetes mellitus type 2", "type 2 diabetes", "t2dm", "t2d"],
        "exclude_if_only": ["prediabetic state", "pre diabetes", "prediabetes"],
        "must_not_contain_patient_type": ["type 2 diabetes", "t2d"],
    },
    # ── Breast cancer subtypes ─────────────────────────────────────────────────
    {
        "patient_has":     ["her2 low", "her2-low", "her2 negative", "her2-negative",
                            "her2 neg", "fish non-amplified"],
        "exclude_if_only": ["her2-positive", "her2 positive", "her2 overexpression",
                            "her2+ breast", "her2-enriched",
                            "her-2-positive", "her-2 positive", "her-2+ breast"],
        "must_not_contain_patient_type": ["her2 low", "her2-low", "her2 negative",
                                          "her2-negative", "her2 neg"],
    },
    {
        "patient_has":     ["estrogen receptor positive", "er positive", "er+",
                            "progesterone receptor positive", "pr positive", "pr+"],
        "exclude_if_only": ["triple negative", "triple-negative", "tnbc",
                            "triple neg breast"],
        "must_not_contain_patient_type": ["hormone receptor positive", "hr positive",
                                          "er positive", "estrogen receptor"],
    },
]


def filter_mismatched_subtypes(
    patient_summary: str,
    trials: list[dict],
    clinical_context: dict | None = None,
) -> list[dict]:
    """
    Remove trials targeting a different disease subtype.
    Layer 1: disease_type Firestore field (most precise)
    Layer 2: rule-based keyword matching
    Layer 3: LLM-extracted exclusion terms
    """
    summary_lower = patient_summary.lower()

    llm_exclusions = []
    if clinical_context:
        llm_exclusions = [
            e.lower() for e in clinical_context.get("exclude_trial_types", [])
        ]
        if llm_exclusions:
            logger.info(f"LLM exclusion terms: {llm_exclusions[:3]}")

    filtered = []
    for trial in trials:
        disease_type     = str(trial.get("disease_type", "")).lower().strip()
        conditions_lower = str(trial.get("conditions",   "")).lower()
        title_lower      = (trial.get("study_title") or trial.get("title", "")).lower()
        combined         = f"{conditions_lower} {title_lower} {disease_type}"
        nct              = trial.get("nct_number", trial.get("_doc_id", "?"))

        skip = False

        # Layer 1: disease_type field
        if disease_type:
            if any(s in summary_lower for s in ["prediabetes", "prediabetic", "metabolic syndrome"]):
                if disease_type in ["type 1 diabetes", "diabetes insipidus"]:
                    logger.info(f"disease_type filter: dropping {nct} ('{disease_type}' for prediabetes patient)")
                    skip = True

            if not skip:
                if any(s in summary_lower for s in ["estrogen receptor positive", "er positive", "er+"]):
                    if disease_type == "triple negative breast cancer":
                        logger.info(f"disease_type filter: dropping {nct} (TNBC for ER+ patient)")
                        skip = True

            if not skip:
                if "triple negative" in summary_lower:
                    if "her2+" in disease_type or "her2 positive" in disease_type:
                        logger.info(f"disease_type filter: dropping {nct} (HER2+ for TNBC patient)")
                        skip = True

            if not skip:
                if any(s in summary_lower for s in ["type 2 diabetes", "t2dm"]):
                    if disease_type == "type 1 diabetes":
                        logger.info(f"disease_type filter: dropping {nct} (T1DM for T2DM patient)")
                        skip = True

        # Layer 2: rule-based
        if not skip:
            for rule in _SUBTYPE_FILTER_RULES:
                patient_matches = any(p in summary_lower for p in rule["patient_has"])
                if not patient_matches:
                    continue
                trial_has_wrong_subtype   = any(e in combined for e in rule["exclude_if_only"])
                trial_also_covers_patient = any(p in combined for p in rule["must_not_contain_patient_type"])
                if trial_has_wrong_subtype and not trial_also_covers_patient:
                    logger.info(f"Rule filter: dropping {nct}")
                    skip = True
                    break

        # Layer 3: LLM exclusions
        if not skip and llm_exclusions:
            for exclusion in llm_exclusions:
                if "type 1" in exclusion and "type 1" in combined and "type 2" not in combined:
                    logger.info(f"LLM filter: dropping {nct} (matches: {exclusion})")
                    skip = True
                    break

        if not skip:
            filtered.append(trial)

    logger.info(f"Subtype filter: {len(trials)} → {len(filtered)} trials remain")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3.6 — CONDITION-SUBTYPE FILTER
# ══════════════════════════════════════════════════════════════════════════════

# Mapping: if patient summary contains any of the "patient_has" terms,
# remove any trial whose conditions+title ONLY mention the "exclude_if_only" terms
# (i.e. the trial is for a different subtype and doesn't also cover the patient's type).
_SUBTYPE_FILTER_RULES = [
    # ── Diabetes ────────────────────────────────────────────────────────────────
    {
        "patient_has":     ["diabetes mellitus type 2", "type 2 diabetes", "t2dm", "t2d"],
        "exclude_if_only": ["type 1 diabetes", "type 1 diabetes mellitus", "t1dm",
                            "cystic fibrosis-related diabetes", "cfrd"],
        "must_not_contain_patient_type": ["type 2", "t2"],
    },
    {
        "patient_has":     ["diabetes mellitus type 2", "type 2 diabetes", "t2dm", "t2d"],
        "exclude_if_only": ["prediabetic state", "pre diabetes", "prediabetes"],
        "must_not_contain_patient_type": ["type 2 diabetes", "t2d"],
    },
    # ── Breast cancer subtypes ──────────────────────────────────────────────────
    # Drop HER2-positive-only trials for HER2-low / HER2-negative patients
    {
        "patient_has":     ["her2 low", "her2-low", "her2 negative", "her2-negative",
                            "her2 neg", "fish non-amplified"],
        "exclude_if_only": ["her2-positive", "her2 positive", "her2 overexpression",
                            "her2+ breast", "her2-enriched",
                            "her-2-positive", "her-2 positive", "her-2+ breast"],
        "must_not_contain_patient_type": ["her2 low", "her2-low", "her2 negative",
                                          "her2-negative", "her2 neg", "her-2-negative",
                                          "her-2 negative"],
    },
    # Drop triple-negative (TNBC) trials for ER+ or PR+ patients
    {
        "patient_has":     ["estrogen receptor positive", "er positive", "er+",
                            "progesterone receptor positive", "pr positive", "pr+"],
        "exclude_if_only": ["triple negative", "triple-negative", "tnbc",
                            "triple neg breast"],
        "must_not_contain_patient_type": ["hormone receptor positive", "hr positive",
                                          "er positive", "estrogen receptor"],
    },
]

def filter_mismatched_subtypes(patient_summary: str, trials: list[dict]) -> list[dict]:
    """Remove trials clearly targeting a disease subtype different from the patient's."""
    summary_lower = patient_summary.lower()
    filtered = []
    for trial in trials:
        conditions_lower = str(trial.get("conditions", "")).lower()
        title_lower = (trial.get("study_title") or trial.get("title", "")).lower()
        combined = f"{conditions_lower} {title_lower}"

        skip = False
        for rule in _SUBTYPE_FILTER_RULES:
            patient_matches = any(p in summary_lower for p in rule["patient_has"])
            if not patient_matches:
                continue
            trial_has_wrong_subtype = any(e in combined for e in rule["exclude_if_only"])
            trial_also_covers_patient = any(p in combined for p in rule["must_not_contain_patient_type"])
            if trial_has_wrong_subtype and not trial_also_covers_patient:
                logger.info(
                    f"Subtype filter: dropping {trial.get('nct_number')} "
                    f"(conditions: {conditions_lower[:80]})"
                )
                skip = True
                break
        if not skip:
            filtered.append(trial)
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════

_EC_LIMIT = 1500
_IV_LIMIT = 300
_SU_LIMIT = 300
_CO_LIMIT = 300


def _trim(val, limit: int) -> str:
    s = str(val) if val and str(val).strip() not in ("", "nan", "None") else "N/A"
    return s[:limit] + "…" if len(s) > limit else s


def filter_to_eligible_only(gemini_text: str, medgemma_text: str) -> str:
    """
    Keep only trial blocks where BOTH Gemini AND MedGemma say ELIGIBLE or BORDERLINE.
    Falls back to Gemini-only if MedGemma unavailable.
    """
    gemini_blocks = re.split(r"\n---+\n?", gemini_text)

    medgemma_verdicts: dict[int, str] = {}
    if medgemma_text and not medgemma_text.startswith("(MedGemma judge unavailable"):
        for line in medgemma_text.splitlines():
            match = re.search(
                r"\*\*Trial\s+(\d+)[:\*]*\s*(ELIGIBLE|BORDERLINE|INELIGIBLE)",
                line, re.IGNORECASE
            )
            if match:
                medgemma_verdicts[int(match.group(1))] = match.group(2).upper()

    medgemma_available = bool(medgemma_verdicts)
    logger.info(f"MedGemma available: {medgemma_available}, verdicts: {medgemma_verdicts}")

    kept          = []
    trial_counter = 0

    for block in gemini_blocks:
        block = block.strip()
        if not block:
            continue

        trial_header = re.search(r"\*\*Trial\s+(\d+)", block, re.IGNORECASE)
        if not trial_header:
            kept.append(block)
            continue

        trial_counter += 1
        trial_num = int(trial_header.group(1))

        gemini_match   = re.search(r"VERDICT\s*:\s*(ELIGIBLE|BORDERLINE|INELIGIBLE)", block, re.IGNORECASE)
        gemini_verdict = gemini_match.group(1).upper() if gemini_match else "UNKNOWN"

        if not medgemma_available:
            if gemini_verdict == "INELIGIBLE":
                logger.info(f"Dropping Trial {trial_num}: Gemini=INELIGIBLE (MedGemma unavailable)")
                continue
        else:
            medgemma_verdict = medgemma_verdicts.get(trial_num, "UNKNOWN")
            logger.info(f"Trial {trial_num}: Gemini={gemini_verdict}, MedGemma={medgemma_verdict}")
            if gemini_verdict == "INELIGIBLE" or medgemma_verdict == "INELIGIBLE":
                logger.info(f"Dropping Trial {trial_num}: one or both judges said INELIGIBLE")
                continue

<<<<<<< Updated upstream
        kept.append(block)

    if not kept:
        return "No eligible or borderline trials found for this patient.\n\n" + DISCLAIMER_TEXT

    return "\n\n---\n\n".join(kept)


def generate_recommendation(
    patient_summary: str,
    retrieved_trials: list[dict],
    detected_conditions: list[str] | None = None,
) -> str:
=======
_EC_LIMIT = 1500   # chars per trial for eligibility_criteria
_IV_LIMIT = 300    # chars per trial for interventions
_SU_LIMIT = 300    # chars per trial for brief_summary
_CO_LIMIT = 150    # chars per trial for conditions (can be huge lists)

def _trim(val, limit: int) -> str:
    s = str(val) if val and str(val).strip() not in ("", "nan", "None") else "N/A"
    return s[:limit] + "…" if len(s) > limit else s

def generate_recommendation(patient_summary: str, retrieved_trials: list[dict]) -> str:
>>>>>>> Stashed changes
    context = "\n\n".join([
        f"Trial {i + 1}:\n"
        f"  NCT ID        : {t.get('nct_number', 'N/A')}\n"
        f"  Title         : {t.get('study_title') or t.get('title', 'N/A')}\n"
        f"  Condition     : {_trim(t.get('conditions'), _CO_LIMIT)}\n"
<<<<<<< Updated upstream
        f"  Disease       : {t.get('disease', 'N/A')}\n"
        f"  Disease Type  : {t.get('disease_type', 'N/A')}\n"
=======
>>>>>>> Stashed changes
        f"  Phase         : {t.get('phase', 'N/A')}\n"
        f"  Status        : {t.get('recruitment_status', 'N/A')}\n"
        f"  Age Range     : {t.get('min_age', 'N/A')} – {t.get('max_age', 'N/A')}\n"
        f"  Sex           : {t.get('sex', 'N/A')}\n"
        f"  Eligibility Criteria:\n{_trim(t.get('eligibility_criteria'), _EC_LIMIT)}\n"
        f"  [END OF ELIGIBILITY CRITERIA]\n"
        f"  Intervention  : {_trim(t.get('interventions'), _IV_LIMIT)}\n"
        f"  Summary       : {_trim(t.get('brief_summary'), _SU_LIMIT)}\n"
        f"  URL           : {t.get('study_url', 'N/A')}"
        for i, t in enumerate(retrieved_trials)
    ])

    multi_condition_note = ""
    if detected_conditions and len(detected_conditions) > 1:
        multi_condition_note = (
            f"\nMULTI-CONDITION PATIENT: This patient has conditions spanning "
            f"multiple disease areas: {', '.join(detected_conditions)}. "
            "Evaluate each trial based on its PRIMARY disease focus and whether "
            "the patient's full clinical profile is compatible. "
            "A trial for one condition is still relevant even if the patient also "
            "has another condition, as long as the second condition does not create "
            "an explicit exclusion criterion in the trial.\n"
        )

    system_prompt = (
        "You are a clinical trial matching assistant for TrialLink. "
        "Assess whether each patient is likely eligible for each retrieved trial.\n\n"
        "CORE RULES — follow ALL of these exactly:\n\n"
        "1. USE ONLY THE PROVIDED CONTEXT. Do not apply your training knowledge about "
        "trials, additional eligibility criteria, or medical standards beyond what is "
<<<<<<< Updated upstream
        "written in the eligibility criteria text provided.\n\n"
        "2. Diagnosis synonyms are identical: 'Malignant neoplasm of breast' = 'Breast cancer'. "
        "'HER2 low carcinoma (IHC 2+, FISH non-amplified)' = 'HER2-low' = 'HER2 negative'. "
        "'Estrogen receptor positive tumor' = 'ER+' = 'hormone receptor positive'.\n\n"
        "3. Numeric thresholds: ≥X means value must be X or higher. ≤X means X or lower.\n\n"
        "4. MISSING DATA RULE: if the patient profile does NOT mention a specific lab value "
        "or secondary criterion, DO NOT LIST IT AT ALL. Assume it is met.\n\n"
        "5. Concerns: ONLY list if patient profile EXPLICITLY shows a value failing a criterion. "
        "If nothing fails, write 'None'.\n\n"
        "6. ELIGIBLE: diagnosis matches + age/sex within range + no explicit failures. "
        "This is the DEFAULT verdict.\n\n"
        "7. BORDERLINE: only when a documented value is close to but may not meet a threshold.\n\n"
        "8. INELIGIBLE: ONLY when patient has a wrong disease type explicitly excluded, "
        "or age/sex is definitively outside the stated range.\n\n"
        "MEDICATION RULE: Never mention dosages or drug names. "
        "Refer to trial drugs only as 'the investigational treatment' or by NCT ID.\n\n"
        "ADDITIONAL RULES:\n"
        "- Current use of a trial drug is NOT disqualifying unless criteria say 'no prior exposure'.\n"
        "- 'Advanced' or 'metastatic' trials listing early-stage are open to early-stage patients.\n"
        "- HER2-low (IHC 2+, FISH non-amplified) satisfies 'HER2-negative' criteria."
        + multi_condition_note
=======
        "written in the eligibility criteria text provided. If a criterion is not "
        "listed in the provided eligibility criteria text, ignore it completely.\n\n"
        "2. Diagnosis synonyms are identical: 'Malignant neoplasm of breast' = 'Breast cancer'. "
        "'HER2 low carcinoma (IHC 2+, FISH non-amplified)' = 'HER2-low' = 'HER2 negative'. "
        "'Estrogen receptor positive tumor' = 'ER+' = 'hormone receptor positive'.\n\n"
        "3. Numeric thresholds: ≥X means value must be X or higher. ≤X means X or lower. "
        "Never reverse direction.\n\n"
        "4. MISSING DATA RULE: if the patient profile does NOT mention a specific lab value "
        "or secondary criterion, DO NOT LIST IT AT ALL. Do not write about it. Assume it is met.\n\n"
        "5. Concerns section: ONLY list a concern if the patient profile EXPLICITLY shows "
        "a value that fails a criterion listed in the provided eligibility criteria. "
        "If nothing fails, write 'None'.\n\n"
        "6. ELIGIBLE: patient's diagnosis matches + age/sex within stated range + "
        "no explicit failures in documented values. This is the DEFAULT verdict when "
        "the patient fits the trial's primary disease focus.\n\n"
        "7. BORDERLINE: only when a documented patient value is close to but may not meet "
        "a stated threshold (e.g. documented age 52 vs criterion age ≥55).\n\n"
        "8. INELIGIBLE: ONLY when the patient has a wrong disease type that the trial "
        "explicitly excludes (e.g. patient is ER+ but trial requires triple-negative only), "
        "or age/sex is definitively outside the stated range.\n\n"
        "IMPORTANT: Do not mark INELIGIBLE just because data is missing or unspecified. "
        "Absence of information is NOT a disqualifier.\n\n"
        "ADDITIONAL RULES:\n"
        "- If a patient is currently taking the same drug(s) that the trial is studying "
        "(e.g. patient is on letrozole and trial tests letrozole), this is NOT disqualifying "
        "unless the criteria explicitly say 'no prior exposure to [drug]'. "
        "Trials often enroll patients already on the drug.\n"
        "- 'Advanced' or 'metastatic' trials that also list early-stage breast cancer in their "
        "conditions are open to early-stage patients — do not mark INELIGIBLE for stage alone.\n"
        "- HER2-low (IHC 2+, FISH non-amplified) satisfies 'HER2-negative' criteria."
>>>>>>> Stashed changes
    )

    user_prompt = f"""PATIENT PROFILE:
{patient_summary}

CLINICAL TRIALS TO EVALUATE:
{context}

For EACH trial write your assessment in this exact format:

**Trial [N]: [NCT ID] — [Title]**
VERDICT: ELIGIBLE / BORDERLINE / INELIGIBLE
<<<<<<< Updated upstream

Clinical Rationale:
[Write 3-4 sentences reasoning like a clinician, not a checklist.
Focus on WHY this trial is medically relevant for this specific patient —
connect the patient's primary diagnosis, disease burden, comorbidities,
and clinical profile to the trial's scientific purpose.
Example: "This patient's prediabetes with HbA1c 6.14% and BMI 30.48
places her in the exact at-risk population this prevention trial targets,
making her a strong candidate for lifestyle intervention before T2DM progression."
Do NOT just list age/sex/condition matches. Explain the clinical relevance.]

Key Concerns:
[ONLY mention concerns if the patient's documented values explicitly
conflict with a stated criterion. If none, write None.]

What the Patient Would Do:
[One sentence: study design and what participation involves.]
---"""

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model    = GenerativeModel(GUARDRAIL_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config=GenerationConfig(
                temperature=0.6,
                max_output_tokens=8192,
            ),
        )
        text = response.text.strip()
        logger.info(f"Gemini recommendation length: {len(text)} chars")
        text = filter_to_eligible_only(text, "")
        logger.info(f"After filtering ineligible: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"Gemini recommendation generation failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4B — MEDGEMMA AS JUDGE (chatCompletions format)
# ══════════════════════════════════════════════════════════════════════════════

def medgemma_judge(
    patient_summary: str,
    retrieved_trials: list[dict],
    gemini_analysis: str,
) -> str:
    """
    Use MedGemma as a second-opinion judge using chatCompletions request format.
    """
    trial_lines = "\n".join([
        f"Trial {i+1}: {t.get('nct_number','?')} — {t.get('study_title') or t.get('title','?')}"
        for i, t in enumerate(retrieved_trials)
    ])

    # Clean prompt — no Gemma turn tags, chatCompletions handles conversation structure
    prompt_text = (
        f"You are a board-certified clinical trials physician. "
        f"Read the patient profile and each trial's details, then give YOUR OWN independent "
        f"eligibility verdict. Do NOT look at any other AI's opinion — form your own judgment.\n\n"
        f"PATIENT SUMMARY:\n{patient_summary}\n\n"
        f"TRIALS TO EVALUATE:\n{trial_lines}\n\n"
        f"For each trial write in this exact format:\n"
        f"**Trial N:** ELIGIBLE / INELIGIBLE / BORDERLINE — [2-3 sentences citing specific "
        f"patient data (lab values, diagnosis, age, stage) and the trial's key inclusion or "
        f"exclusion criterion that drove your verdict. Be clinical and precise.]\n\n"
        f"Rules:\n"
        f"- Base verdict ONLY on the patient profile and trial names/conditions listed.\n"
        f"- Cite specific values: 'HbA1c 8.1%', 'BMI 31.8', 'Stage II ER+', 'ECOG 0'.\n"
        f"- ELIGIBLE: documented data clearly fits the trial's primary focus.\n"
        f"- INELIGIBLE: documented value or diagnosis explicitly conflicts.\n"
        f"- BORDERLINE: likely fits but one criterion needs clinician confirmation.\n"
        f"- No extra text outside the Trial lines."
    )
=======

Matched Criteria:
- [2-4 criteria the patient clearly meets based on their documented data]

Concerns:
- [ONLY criteria the patient's profile explicitly fails — if none, write None]

Intervention Summary: [one sentence: what would the patient actually do/receive]
Clinical Rationale: [2 sentences: why this trial fits or does not fit this specific patient]
---"""

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        model = GenerativeModel(GUARDRAIL_MODEL)
        response = model.generate_content(
            full_prompt,
            generation_config=GenerationConfig(
                temperature=0.4,
                max_output_tokens=8192,
            ),
        )
        text = response.text.strip()
        logger.info(f"Gemini recommendation length: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"Gemini recommendation generation failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4B — MEDGEMMA AS JUDGE
# ══════════════════════════════════════════════════════════════════════════════

def medgemma_judge(patient_summary: str, retrieved_trials: list[dict], gemini_analysis: str) -> str:
    """
    Use MedGemma as a second-opinion judge on Gemini 2.5 Flash's per-trial verdicts.
    Returns one AGREE/DISAGREE line per trial.
    """
    trial_lines = "\n".join([
        f"Trial {i+1}: {t.get('nct_number','?')} — {t.get('study_title') or t.get('title','?')}"
        for i, t in enumerate(retrieved_trials)
    ])

    # Keep Gemini analysis short — just the verdict lines — to stay within MedGemma context
    verdict_lines = []
    for line in gemini_analysis.splitlines():
        if re.search(r"VERDICT\s*:", line, re.IGNORECASE) or re.match(r"\*\*Trial\s+\d+", line.strip()):
            verdict_lines.append(line.strip())
    verdicts_summary = "\n".join(verdict_lines) if verdict_lines else gemini_analysis[:800]

    prompt = (
        f"<start_of_turn>user\n"
        f"You are a board-certified clinical trials physician. "
        f"Read the patient profile and each trial's details, then give YOUR OWN independent "
        f"eligibility verdict. Do NOT look at any other AI's opinion — form your own judgment.\n\n"
        f"PATIENT SUMMARY:\n{patient_summary}\n\n"
        f"TRIALS TO EVALUATE:\n{trial_lines}\n\n"
        f"For each trial write in this exact format:\n"
        f"**Trial N:** ELIGIBLE / INELIGIBLE / BORDERLINE — [2-3 sentences citing the specific "
        f"patient data (lab values, diagnosis, age, stage) and the trial's key inclusion or "
        f"exclusion criterion that drove your verdict. Be clinical and precise.]\n\n"
        f"Rules:\n"
        f"- Base your verdict ONLY on the patient profile above and the trial names/conditions listed.\n"
        f"- Cite specific values: 'HbA1c 8.1%', 'BMI 31.8', 'Stage II ER+', 'ECOG 0'.\n"
        f"- ELIGIBLE: patient's documented data clearly fits the trial's primary focus.\n"
        f"- INELIGIBLE: patient has a documented value or diagnosis that explicitly conflicts.\n"
        f"- BORDERLINE: patient likely fits but one criterion needs clinician confirmation.\n"
        f"- No extra text outside the Trial lines.\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    try:
        import google.auth
        from google.auth.transport.requests import Request as AuthRequest
        import requests as _req
>>>>>>> Stashed changes

    try:
        region         = os.getenv("GCP_REGION",         "us-central1")
        project_number = os.getenv("MODEL_PROJECT_NUMBER", "153563619775")
        endpoint_id    = MEDGEMMA_ENDPOINT_ID

        dedicated_domain = f"{endpoint_id}.{region}-{project_number}.prediction.vertexai.goog"
        url = (
            f"https://{dedicated_domain}/v1/projects/{project_number}"
            f"/locations/{region}/endpoints/{endpoint_id}:predict"
        )

<<<<<<< Updated upstream
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())

        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type":  "application/json",
        }

        # chatCompletions format — matches what GCP console shows
        payload = {
            "instances": [{
                "@requestFormat": "chatCompletions",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt_text}]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1
            }]
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
=======
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(AuthRequest())

        headers = {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}
        payload = {"instances": [{"prompt": prompt, "max_tokens": 512, "temperature": 0.1}]}
>>>>>>> Stashed changes

        response = _req.post(url, headers=headers, json=payload, timeout=60)
        result = response.json()["predictions"][0]

<<<<<<< Updated upstream
        # chatCompletions response parsing
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "") or str(result)
            else:
                # Fallback to older response formats
                text = (
                    result.get("generated_text") or
                    result.get("output") or
                    str(result)
                )
        else:
            text = str(result)

        text = text.strip()
=======
        text = (result.get("generated_text") or result.get("output") or str(result)) if isinstance(result, dict) else str(result)

        # Strip prompt echo
        for marker in ("<start_of_turn>model", "Output:"):
            if marker in text:
                text = text.split(marker, 1)[1]
                break
        text = re.sub(r"<end_of_turn>.*", "", text, flags=re.DOTALL).strip()

>>>>>>> Stashed changes
        logger.info(f"MedGemma judge output ({len(text)} chars): {text[:300]}")
        return text if text else "(MedGemma returned empty response)"

    except Exception as e:
        logger.warning(f"MedGemma judge failed: {e}")
        return f"(MedGemma judge unavailable: {e})"


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def rag_pipeline(patient_summary: str) -> dict:
    logger.info("=" * 60)
    logger.info("Starting TrialLink RAG Pipeline")
    logger.info("=" * 60)

    guardrail_meta: dict[str, Any] = {
        "status":              "passed",
        "stage":               "completed",
        "reason":              "",
        "pii_hits":            {},
        "flag_reasons":        [],
        "llm_input_judgment":  None,
        "llm_output_judgment": None,
    }

    # ── Step 0A: PII redaction ────────────────────────────────────────────────
    logger.info("Step 0A: Redacting PII...")
    patient_summary, pii_hits = redact_basic_pii(patient_summary)
    guardrail_meta["pii_hits"] = pii_hits

    # ── Step 0B: Structural validation ───────────────────────────────────────
    logger.info("Step 0B: Structural input validation...")
    valid_structure, structure_reason = validate_input_structure(patient_summary)
    if not valid_structure:
        return safe_guardrail_response(
            reason=structure_reason,
            patient_summary=patient_summary,
            guardrail_stage="input_structure",
            pii_hits=pii_hits,
        )

    # ── Step 0C: LLM input guardrail ─────────────────────────────────────────
    if ENABLE_INPUT_LLM_GUARDRAIL:
        logger.info("Step 0C: Running LLM input guardrail...")
        try:
            input_judgment = llm_input_guardrail(patient_summary)
            guardrail_meta["llm_input_judgment"] = input_judgment
            logger.info(f"Input guardrail: {input_judgment}")

            if not input_judgment["is_valid"]:
                return safe_guardrail_response(
                    reason=input_judgment["reason"],
                    patient_summary=patient_summary,
                    guardrail_stage="input_llm_judge",
                    pii_hits=pii_hits,
                )
            if input_judgment["category"] != "valid_supported_clinical_summary":
                return safe_guardrail_response(
                    reason=input_judgment["reason"],
                    patient_summary=patient_summary,
                    guardrail_stage="input_scope",
                    pii_hits=pii_hits,
                )
        except Exception as e:
            logger.warning(f"LLM input guardrail failed: {e}")
            guardrail_meta["status"] = "flagged"
            guardrail_meta["stage"]  = "input_llm_judge_error"
            guardrail_meta["reason"] = str(e)
            guardrail_meta["flag_reasons"].append(f"input_llm_guardrail_error: {e}")

    # ── Step 1A: LLM clinical context enrichment ──────────────────────────────
    logger.info("Step 1A: LLM clinical context enrichment...")
    clinical_context    = {}
    detected_conditions = []

    try:
        clinical_context = enrich_patient_context(patient_summary)
        llm_conditions   = [
            c for c in clinical_context.get("condition_categories", [])
            if c in CONDITIONS
        ]
        if llm_conditions:
            detected_conditions = llm_conditions
            logger.info(f"LLM-detected conditions: {detected_conditions}")
        else:
            logger.info("LLM returned no supported conditions — using rule-based fallback")
            detected_conditions = detect_patient_conditions(patient_summary)

    except Exception as e:
        logger.warning(f"LLM enrichment failed, using rule-based: {e}")
        clinical_context    = {}
        detected_conditions = detect_patient_conditions(patient_summary)

    is_dual_condition = len(detected_conditions) > 1
    logger.info(f"Final detected conditions: {detected_conditions} (dual={is_dual_condition})")

    # ── Step 1C: Build enriched retrieval query ───────────────────────────────
    logger.info("Step 1C: Building enriched retrieval query...")
    retrieval_query = build_retrieval_query(
        patient_summary, detected_conditions, clinical_context
    )

    # ── Step 1: Embed enriched query ──────────────────────────────────────────
    logger.info("Step 1: Embedding enriched retrieval query...")
    patient_embedding = embed_text(retrieval_query, task_type="RETRIEVAL_QUERY")
    logger.info(f"Embedding dimensions: {len(patient_embedding)}")

    # ── Step 2: Vector search ─────────────────────────────────────────────────
    logger.info(f"Step 2: Querying Vector Search (top {RETRIEVAL_TOP_K})...")
    candidate_nct_ids = query_vector_search(patient_embedding, top_k=RETRIEVAL_TOP_K)
    if not candidate_nct_ids:
        return safe_guardrail_response(
            reason="No clinical trials found for this supported condition",
            patient_summary=patient_summary,
            guardrail_stage="retrieval_empty",
            pii_hits=pii_hits,
        )

    # ── Step 3: Fetch from condition-specific Firestore collections ───────────
    logger.info("Step 3: Fetching from condition-specific Firestore collections...")
    candidates = fetch_trials_from_firestore(
        candidate_nct_ids,
        target_conditions=detected_conditions,
    )
    logger.info(f"Fetched {len(candidates)} trial documents")

    valid_retrieval, retrieval_reason = validate_retrieved_trials(candidates)
    if not valid_retrieval:
        return safe_guardrail_response(
            reason=retrieval_reason,
            patient_summary=patient_summary,
            guardrail_stage="retrieval_validation",
            pii_hits=pii_hits,
        )

<<<<<<< Updated upstream
    # ── Step 3.6: Subtype filter BEFORE reranking ─────────────────────────────
    logger.info("Step 3.6: Filtering mismatched subtypes...")
    candidates = filter_mismatched_subtypes(
        patient_summary, candidates, clinical_context
    )
    logger.info(f"After subtype filter: {len(candidates)} candidates")
=======
    # Step 3.6: Remove trials for the wrong disease subtype BEFORE reranking
    # so the reranker selects top-K from a clean pool, not wasting slots on mismatched subtypes.
    logger.info("Step 3.6: Filtering mismatched subtypes from all candidates...")
    candidates = filter_mismatched_subtypes(patient_summary, candidates)
    logger.info(f"After subtype filter: {len(candidates)} candidates remain")
>>>>>>> Stashed changes

    if not candidates:
        return safe_guardrail_response(
            reason="No matching trials remain after subtype filtering",
            patient_summary=patient_summary,
            guardrail_stage="subtype_filter",
            pii_hits=pii_hits,
        )

<<<<<<< Updated upstream
    # ── Step 3.5: Rerank (multi-condition aware) ──────────────────────────────
=======
    # Step 3.5: Rerank the filtered candidates → top K
>>>>>>> Stashed changes
    logger.info(f"Step 3.5: Reranking {len(candidates)} -> top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials_multi_condition(
        patient_summary,
        candidates,
        detected_conditions,
        top_k=RERANK_TOP_K,
    )

    valid_scope, scope_reason = validate_trials_scope(reranked_trials)
    if not valid_scope:
        return safe_guardrail_response(
            reason=scope_reason,
            patient_summary=patient_summary,
            guardrail_stage="retrieval_scope",
            pii_hits=pii_hits,
        )

    # ── Step 4: Generate recommendation ──────────────────────────────────────
    logger.info("Step 4: Generating recommendation...")
<<<<<<< Updated upstream
    recommendation    = generate_recommendation(
        patient_summary, reranked_trials, detected_conditions
    )
    raw_gemini_output = recommendation
=======
    recommendation = generate_recommendation(patient_summary, reranked_trials)
    raw_medgemma_output = recommendation  # preserve before any guardrail replaces it
>>>>>>> Stashed changes

    # ── Step 5A: Policy checks ────────────────────────────────────────────────
    logger.info("Step 5A: Policy-based output guardrails...")
    recommendation = append_disclaimer(recommendation)
    output_policy_ok, output_policy_reason = policy_check_output(recommendation)
    if not output_policy_ok:
        logger.warning(f"Policy guardrail triggered: {output_policy_reason}")
        guardrail_meta["status"] = "flagged"
        guardrail_meta["stage"]  = "output_policy"
        guardrail_meta["reason"] = output_policy_reason
        guardrail_meta["flag_reasons"].append(output_policy_reason)
        recommendation = (
            "The generated recommendation did not pass output safety policy validation. "
            "Please review the retrieved trials manually.\n\n"
            f"{DISCLAIMER_TEXT}"
        )

    # ── Step 5B: Grounding checks ─────────────────────────────────────────────
    logger.info("Step 5B: Grounding checks...")
    grounding_ok, grounding_reason = grounding_check_output(recommendation, reranked_trials)
    if not grounding_ok:
        logger.warning(f"Grounding guardrail triggered: {grounding_reason}")
        guardrail_meta["status"] = "flagged"
        guardrail_meta["stage"]  = "output_grounding"
        guardrail_meta["reason"] = grounding_reason
        guardrail_meta["flag_reasons"].append(grounding_reason)
        recommendation = (
            "The generated recommendation did not pass grounding validation. "
            "Please review the retrieved trials manually.\n\n"
            f"{DISCLAIMER_TEXT}"
        )

    # ── Step 5C: LLM output guardrail ────────────────────────────────────────
    if ENABLE_OUTPUT_LLM_GUARDRAIL:
        logger.info("Step 5C: LLM output guardrail...")
        try:
            llm_output_judgment = llm_output_guardrail(
                patient_summary=patient_summary,
                recommendation=recommendation,
                retrieved_trials=reranked_trials,
            )
            guardrail_meta["llm_output_judgment"] = llm_output_judgment
            logger.info(f"Output guardrail: {llm_output_judgment}")

            if not (llm_output_judgment["is_safe"] and llm_output_judgment["is_grounded"]):
                guardrail_meta["status"] = "flagged"
                guardrail_meta["stage"]  = "output_llm_judge"
                guardrail_meta["reason"] = llm_output_judgment["reason"]
                guardrail_meta["flag_reasons"].append(llm_output_judgment["reason"])
                recommendation = (
                    "The generated recommendation did not pass final safety and grounding validation. "
                    "Please review the retrieved trials manually.\n\n"
                    f"{DISCLAIMER_TEXT}"
                )
        except Exception as e:
            logger.warning(f"LLM output guardrail failed: {e}")
            guardrail_meta["status"] = "flagged"
            guardrail_meta["stage"]  = "output_llm_judge_error"
            guardrail_meta["reason"] = str(e)
            guardrail_meta["flag_reasons"].append(f"output_llm_guardrail_error: {e}")

<<<<<<< Updated upstream
    # ── Step 5D: MedGemma judge ───────────────────────────────────────────────
    logger.info("Step 5D: MedGemma second-opinion judge...")
    medgemma_judgment = medgemma_judge(patient_summary, reranked_trials, raw_gemini_output)
    logger.info(f"MedGemma output: {medgemma_judgment[:300]}")

    # ── Step 5E: Filter to consensus eligible/borderline ─────────────────────
    logger.info("Step 5E: Filtering to consensus eligible/borderline trials...")
    if recommendation and not recommendation.startswith("The generated recommendation did not pass"):
        recommendation = filter_to_eligible_only(recommendation, medgemma_judgment)
        logger.info(f"Final recommendation: {len(recommendation)} chars")
=======
    # Step 5D: MedGemma as judge — second opinion on Gemini's verdicts
    logger.info("Step 5D: Running MedGemma as judge...")
    medgemma_judgment = medgemma_judge(patient_summary, reranked_trials, raw_medgemma_output)
    logger.info(f"MedGemma judge output: {medgemma_judgment[:300]}")
>>>>>>> Stashed changes

    logger.info("RAG Pipeline complete")
    logger.info("=" * 60)

    return {
        "patient_summary":          patient_summary,
        "detected_conditions":      detected_conditions,
        "is_dual_condition":        is_dual_condition,
        "candidates_before_rerank": candidates,
<<<<<<< Updated upstream
        "retrieved_trials":         reranked_trials,
        "recommendation":           recommendation,
        "raw_gemini_output":        raw_gemini_output,
        "medgemma_judgment":        medgemma_judgment,
        "guardrail":                guardrail_meta,
=======
        "retrieved_trials": reranked_trials,
        "recommendation": recommendation,
        "raw_medgemma_output": raw_medgemma_output,
        "medgemma_judgment": medgemma_judgment,
        "guardrail": guardrail_meta,
>>>>>>> Stashed changes
    }


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT FETCH HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_patient_summary(patient_id: str) -> str:
    try:
        from data_models import Patient
    except ImportError:
        from sdk.patient_package.data_models import Patient

    db  = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    doc = db.collection("patients").document(patient_id).get()

    if not doc.exists:
        raise ValueError(f"Patient {patient_id} not found in Firestore")

    patient = Patient(**doc.to_dict())
    return patient.to_text_summary()


def rag_pipeline_for_patient(patient_id: str) -> dict:
    logger.info(f"Fetching patient {patient_id} from Firestore...")
    summary = get_patient_summary(patient_id)
    logger.info(f"Patient summary: {summary}")
    return rag_pipeline(summary)

rag_pipeline_for_patient("ceec0dea-34f7-4ee4-8c7d-d302e8756672")