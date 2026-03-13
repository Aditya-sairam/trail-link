"""
RAG Service - Local CSV + Cached Embeddings - TrialLink
========================================================
Improvements over v1:
  1. Patient field extraction  — structured fields parsed from free text
  2. Pre-filter                — RECRUITING only, reduces noise
  3. Weighted vector search    — separate embeddings per field, weighted scoring
  4. Larger retrieval pool     — top 50 candidates before reranking
  5. Generic system prompt     — no hardcoded disease rules
  6. MedGemma via endpoint     — replaces Gemini

Cache files (built once, reused forever):
  embeddings_title.npy        — trial title embeddings
  embeddings_condition.npy    — trial condition embeddings
  embeddings_eligibility.npy  — trial eligibility embeddings
  trials_cache.pkl            — trial dicts
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

# ── Init Vertex AI ─────────────────────────────────────────────────────────────
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID   = os.getenv("GCP_PROJECT_ID")
GCP_REGION       = os.getenv("GCP_REGION", "us-central1")
EMBEDDING_MODEL  = "text-embedding-005"
RETRIEVAL_TOP_K  = 50   # increased from 20 — more candidates for reranker
RERANK_TOP_K     = 5

MEDGEMMA_ENDPOINT_ID = os.getenv(
    "MEDGEMMA_ENDPOINT_ID",
    "mg-endpoint-645b70e1-a108-4645-adfa-ccc7f14c9de0"
)
MEDGEMMA_MAX_TOKENS = 2048

# File paths
CSV_PATH              = os.getenv("TRIALS_CSV_PATH", "clinical_trials.csv")
CACHE_TITLE           = os.getenv("CACHE_TITLE",       "embeddings_title.npy")
CACHE_CONDITION       = os.getenv("CACHE_CONDITION",   "embeddings_condition.npy")
CACHE_ELIGIBILITY     = os.getenv("CACHE_ELIGIBILITY", "embeddings_eligibility.npy")
TRIALS_CACHE          = os.getenv("TRIALS_CACHE_PATH", "trials_cache.pkl")

# Weighted scoring — eligibility matters most for matching
WEIGHT_TITLE       = 0.2
WEIGHT_CONDITION   = 0.4
WEIGHT_ELIGIBILITY = 0.4


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Embed any text using Vertex AI text-embedding-005."""
    try:
        model  = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
        return model.get_embeddings(inputs)[0].values
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Batch cosine similarity between one query vector and a matrix."""
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms       = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    normed      = matrix / norms
    return normed @ query_norm  # shape: (n_trials,)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — LOAD CSV + PRE-FILTER
# ══════════════════════════════════════════════════════════════════════════════

def load_trials() -> list[dict]:
    """
    Load trials from local CSV.
    Pre-filters to RECRUITING status only — reduces noise in vector search.
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found at '{CSV_PATH}'. "
            f"Set TRIALS_CSV_PATH env var to point to your existing CSV."
        )
    df = pd.read_csv(CSV_PATH).fillna("")

    # Pre-filter: only actively recruiting trials
    recruiting = df[
        df["recruitment_status"].str.upper().str.strip() == "RECRUITING"
    ]
    logger.info(
        f"Loaded {len(df)} trials from CSV. "
        f"After RECRUITING filter: {len(recruiting)} trials."
    )
    return recruiting.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD OR LOAD FIELD-LEVEL EMBEDDING CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_or_build_embeddings(trials: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load or build three separate embedding matrices:
      - Title embeddings
      - Condition embeddings
      - Eligibility embeddings

    Weighted combination of these three gives better retrieval than
    embedding the full trial text as one blob.

    Cache is invalidated if trial count changes (e.g. new CSV).

    Returns:
        title_embs, condition_embs, eligibility_embs — each shape (n, 768)
    """
    n = len(trials)
    caches = [CACHE_TITLE, CACHE_CONDITION, CACHE_ELIGIBILITY]

    # ── Load if all caches exist and match trial count ────────────────────────
    if all(os.path.exists(c) for c in caches):
        title_embs       = np.load(CACHE_TITLE)
        condition_embs   = np.load(CACHE_CONDITION)
        eligibility_embs = np.load(CACHE_ELIGIBILITY)

        if title_embs.shape[0] == n:
            logger.info(
                f"Loaded field embeddings from cache "
                f"({n} trials, {title_embs.shape[1]}D each field)"
            )
            return title_embs, condition_embs, eligibility_embs
        else:
            logger.warning(f"Cache mismatch ({title_embs.shape[0]} vs {n}). Rebuilding...")

    # ── Build ─────────────────────────────────────────────────────────────────
    logger.info(f"Building field-level embeddings for {n} trials (runs once)...")

    title_list, condition_list, eligibility_list = [], [], []

    for i, trial in enumerate(trials):
        title_text       = str(trial.get("study_title", "") or "")
        condition_text   = (
            f"{trial.get('conditions', '')} "
            f"{trial.get('disease', '')} "
            f"{trial.get('keywords', '')}"
        )
        eligibility_text = (
            f"{trial.get('eligibility_criteria', '')} "
            f"Age: {trial.get('min_age', '')} to {trial.get('max_age', '')}. "
            f"Sex: {trial.get('sex', '')}. "
            f"Phase: {trial.get('phase', '')}."
        )

        title_list.append(embed_text(title_text,       "RETRIEVAL_DOCUMENT"))
        condition_list.append(embed_text(condition_text,   "RETRIEVAL_DOCUMENT"))
        eligibility_list.append(embed_text(eligibility_text, "RETRIEVAL_DOCUMENT"))

        if (i + 1) % 50 == 0:
            logger.info(f"  Embedded {i + 1}/{n} trials...")

    title_embs       = np.array(title_list)
    condition_embs   = np.array(condition_list)
    eligibility_embs = np.array(eligibility_list)

    np.save(CACHE_TITLE,       title_embs)
    np.save(CACHE_CONDITION,   condition_embs)
    np.save(CACHE_ELIGIBILITY, eligibility_embs)
    pd.DataFrame(trials).to_pickle(TRIALS_CACHE)

    logger.info(f"Field embeddings saved. Shape per field: {title_embs.shape}")
    return title_embs, condition_embs, eligibility_embs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PATIENT FIELD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_patient_fields(patient_summary: str) -> dict:
    """
    Parse key structured fields from free-text patient summary.
    Used to build targeted query strings per embedding field.

    Simple keyword extraction — no model needed.
    Returns dict with: condition_query, eligibility_query
    """
    text = patient_summary.lower()

    # Condition query — pull condition + key biomarkers mentioned
    condition_query = patient_summary  # use full text for condition matching

    # Eligibility query — emphasize structured facts
    # Extract age if present
    import re
    age_match = re.search(r'(\d{2})[- ]?year', text)
    age       = age_match.group(1) if age_match else ""

    sex = ""
    if "female" in text or "woman" in text or "her " in text:
        sex = "female"
    elif "male" in text or "man" in text:
        sex = "male"

    eligibility_query = (
        f"{patient_summary} "
        f"Age: {age}. Sex: {sex}."
    )

    return {
        "condition_query"  : condition_query,
        "eligibility_query": eligibility_query,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — WEIGHTED VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def query_vector_search(
    patient_summary   : str,
    trials            : list[dict],
    title_embs        : np.ndarray,
    condition_embs    : np.ndarray,
    eligibility_embs  : np.ndarray,
    top_k             : int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Weighted cosine similarity across three field-level embeddings.

    Score = 0.2 * title_sim + 0.4 * condition_sim + 0.4 * eligibility_sim

    Eligibility and condition weighted higher than title because they
    contain the actual matching criteria (biomarkers, age, sex, stage).

    Args:
        patient_summary  : raw patient text
        trials           : list of trial dicts
        title_embs       : (n, 768) title embeddings
        condition_embs   : (n, 768) condition embeddings
        eligibility_embs : (n, 768) eligibility embeddings
        top_k            : number of candidates to return

    Returns:
        Top-k trial dicts ordered by weighted score (descending)
    """
    fields = extract_patient_fields(patient_summary)

    # Embed patient fields
    q_title       = np.array(embed_text(patient_summary,              "RETRIEVAL_QUERY"))
    q_condition   = np.array(embed_text(fields["condition_query"],    "RETRIEVAL_QUERY"))
    q_eligibility = np.array(embed_text(fields["eligibility_query"],  "RETRIEVAL_QUERY"))

    # Score each field separately
    title_scores       = _cosine_scores(q_title,       title_embs)
    condition_scores   = _cosine_scores(q_condition,   condition_embs)
    eligibility_scores = _cosine_scores(q_eligibility, eligibility_embs)

    # Weighted combination
    final_scores = (
        WEIGHT_TITLE       * title_scores +
        WEIGHT_CONDITION   * condition_scores +
        WEIGHT_ELIGIBILITY * eligibility_scores
    )

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    top_trials  = [trials[i] for i in top_indices]

    logger.info(f"Weighted vector search top {top_k}: {[t.get('nct_number') for t in top_trials]}")
    return top_trials


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RERANK
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary : str,
    trials          : list[dict],
    top_k           : int = RERANK_TOP_K,
) -> list[dict]:
    """
    Rerank candidates using Vertex AI Ranking API.
    Falls back to vector search order if API is unavailable.
    """
    try:
        client         = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project        = GCP_PROJECT_ID,
            location       = GCP_REGION,
            ranking_config = "default_ranking_config"
        )

        records = [
            discoveryengine.RankingRecord(
                id      = str(t.get("nct_number", "")),
                title   = str(t.get("study_title", "")),
                content = (
                    f"Condition: {t.get('conditions', '')}. "
                    f"Disease: {t.get('disease', '')}. "
                    f"Eligibility: {t.get('eligibility_criteria', '')}. "
                    f"Phase: {t.get('phase', '')}. "
                    f"Sex: {t.get('sex', '')}. "
                    f"Age: {t.get('min_age', '')} to {t.get('max_age', '')}. "
                    f"Interventions: {t.get('interventions', '')}."
                )
            )
            for t in trials if t.get("nct_number")
        ]

        request = discoveryengine.RankRequest(
            ranking_config = ranking_config,
            model          = "semantic-ranker-512@latest",
            top_n          = top_k,
            query          = patient_summary,
            records        = records
        )

        response  = client.rank(request=request)
        trial_map = {str(t.get("nct_number", "")): t for t in trials}
        reranked  = [
            trial_map[r.id] for r in response.records if r.id in trial_map
        ]

        logger.info(f"Reranked {len(trials)} → top {len(reranked)}: {[t.get('nct_number') for t in reranked]}")
        return reranked

    except Exception as e:
        logger.warning(f"Reranking unavailable: {e}. Using vector search order.")
        return trials[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE RECOMMENDATION WITH MEDGEMMA
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(
    patient_summary  : str,
    retrieved_trials : list[dict],
) -> str:
    """Generate clinical trial recommendations using MedGemma endpoint."""
    context = "\n\n".join([
        f"Trial {i + 1}:\n"
        f"  NCT ID       : {t.get('nct_number', 'N/A')}\n"
        f"  Title        : {t.get('study_title', 'N/A')}\n"
        f"  Condition    : {t.get('conditions', 'N/A')}\n"
        f"  Disease      : {t.get('disease', 'N/A')}\n"
        f"  Phase        : {t.get('phase', 'N/A')}\n"
        f"  Status       : {t.get('recruitment_status', 'N/A')}\n"
        f"  Eligibility  : {t.get('eligibility_criteria', 'N/A')}\n"
        f"  Min Age      : {t.get('min_age', 'N/A')}\n"
        f"  Max Age      : {t.get('max_age', 'N/A')}\n"
        f"  Sex          : {t.get('sex', 'N/A')}\n"
        f"  Interventions: {t.get('interventions', 'N/A')}\n"
        f"  Summary      : {t.get('brief_summary', 'N/A')}\n"
        f"  URL          : {t.get('study_url', 'N/A')}"
        for i, t in enumerate(retrieved_trials)
    ])

    system_prompt = (
        "You are a clinical trial matching assistant for TrialLink. "
        "Your job is to evaluate whether a patient is eligible for each trial. "
        "For every trial, carefully check ALL eligibility criteria against the patient profile. "
        "If ANY hard exclusion criterion is not met, explicitly exclude the trial and state why. "
        "Do not recommend trials the patient is clearly ineligible for. "
        "Be concise and clinically precise."
    )

    user_prompt = f"""Patient Profile:
{patient_summary}

Top Matching Clinical Trials:
{context}

For each trial:
1. State whether the patient is eligible or not
2. Explain specifically why — check condition, biomarkers, age, sex, stage, prior treatments
3. Describe the intervention offered
4. Flag any disqualifying criteria clearly"""

    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    endpoint = aiplatform.Endpoint(MEDGEMMA_ENDPOINT_ID)

    response = endpoint.predict(instances=[{
        "@requestFormat": "chatCompletions",
        "messages": [
            {
                "role"   : "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role"   : "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ],
        "max_tokens": MEDGEMMA_MAX_TOKENS
    }])

    predictions = response.predictions
    return predictions["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Load once at module level
logger.info("Initializing TrialLink RAG...")
_TRIALS                                    = load_trials()
_TITLE_EMBS, _CONDITION_EMBS, _ELIG_EMBS  = load_or_build_embeddings(_TRIALS)
logger.info(f"Ready — {len(_TRIALS)} trials indexed.")


def rag_pipeline(patient_summary: str) -> dict:
    """
    End-to-end RAG pipeline.

    Args:
        patient_summary: Plain text patient profile

    Returns:
        {
            "patient_summary"          : str,
            "candidates_before_rerank" : list[dict],
            "retrieved_trials"         : list[dict],
            "recommendation"           : str
        }
    """
    logger.info("=" * 60)
    logger.info("TrialLink RAG Pipeline")
    logger.info("=" * 60)

    # Step 1 — Weighted vector search (embeds patient internally)
    logger.info(f"Step 1: Weighted vector search (top {RETRIEVAL_TOP_K})...")
    candidates = query_vector_search(
        patient_summary,
        _TRIALS, _TITLE_EMBS, _CONDITION_EMBS, _ELIG_EMBS,
        top_k=RETRIEVAL_TOP_K
    )

    # Step 2 — Rerank → top 5
    logger.info(f"Step 2: Reranking {len(candidates)} → top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials(patient_summary, candidates, top_k=RERANK_TOP_K)

    # Step 3 — Generate recommendation
    logger.info("Step 3: Generating recommendation with MedGemma...")
    recommendation = generate_recommendation(patient_summary, reranked_trials)

    logger.info("Pipeline complete.")

    return {
        "patient_summary"          : patient_summary,
        "candidates_before_rerank" : candidates,
        "retrieved_trials"         : reranked_trials,
        "recommendation"           : recommendation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)s  %(message)s"
    )

    patients = {
        "Diabetic Female": """
            45-year-old female with Type 2 diabetes diagnosed 3 years ago.
            HbA1c: 8.2%, BMI: 28, no prior insulin therapy.
            Currently on Metformin. No cardiovascular disease. Non-smoker.
        """,
        "Breast Cancer Patient": """
            52-year-old female diagnosed with HER2-positive breast cancer, stage II.
            Post-menopausal. No prior targeted therapy. ECOG performance status 0.
        """,
    }

    for label, summary in patients.items():
        print("\n" + "=" * 60)
        print(f"TEST PATIENT: {label}")
        print("=" * 60)

        result = rag_pipeline(summary)

        print(f"\nCANDIDATES ({len(result['candidates_before_rerank'])}):")
        for t in result["candidates_before_rerank"]:
            print(f"  [{t.get('nct_number', 'N/A')}] {t.get('study_title', 'N/A')}")

        print(f"\nAFTER RERANKING (top {RERANK_TOP_K}):")
        for t in result["retrieved_trials"]:
            print(f"  [{t.get('nct_number', 'N/A')}] {t.get('study_title', 'N/A')}")

        print("\nRECOMMENDATION:")
        print(result["recommendation"])