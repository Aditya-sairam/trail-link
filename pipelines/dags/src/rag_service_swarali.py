"""
Local RAG Service with CSV Backend - TrialLink
===============================================
Same pipeline as mock_rag_service.py but reads/writes from local CSV
instead of Firestore. No GCP Firestore access required.

Pipeline:
  1. Load trials from local CSV (create if not found, update if found)
  2. Embed patient summary       (real Vertex AI text-embedding-005)
  3. Mock vector search          (cosine on in-memory index)
  4. Rerank                      (real Vertex AI Ranking API)
  5. Generate recommendation     (real Gemini)

CSV Behaviour:
  - If CSV does not exist  → fetches from ClinicalTrials.gov API and creates it
  - If CSV already exists  → loads it, checks for new trials, appends new rows only
"""

from __future__ import annotations

import os
import logging
import time
import numpy as np
import pandas as pd
import requests
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from google.cloud import discoveryengine_v1alpha as discoveryengine

logger = logging.getLogger(__name__)

# ── Init Vertex AI ─────────────────────────────────────────────────────────────
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
EMBEDDING_MODEL = "text-embedding-005"
LLM_MODEL       = "gemini-2.0-flash"
RETRIEVAL_TOP_K = 20
RERANK_TOP_K    = 5

# CSV settings
CSV_PATH        = os.getenv("TRIALS_CSV_PATH", "clinical_trials.csv")
CONDITIONS      = ["diabetes", "breast cancer"]   # used for API fetch
MAX_TRIALS_PER_CONDITION = 500                     # limit for initial fetch

# ClinicalTrials.gov API v2
CTGOV_API_URL   = "https://clinicaltrials.gov/api/v2/studies"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — CSV MANAGEMENT: CREATE OR UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_trials_from_ctgov(condition: str, max_trials: int = MAX_TRIALS_PER_CONDITION) -> list[dict]:
    """
    Fetch trials from ClinicalTrials.gov API v2 for a given condition.
    Handles pagination automatically.

    Args:
        condition  : Search condition e.g. "diabetes", "breast cancer"
        max_trials : Max number of trials to fetch per condition

    Returns:
        List of trial dicts with normalized fields
    """
    trials   = []
    next_token = None
    page_size  = 100  # max allowed per request

    logger.info(f"Fetching trials for condition: '{condition}'...")

    while len(trials) < max_trials:
        params = {
            "query.cond"   : condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize"     : min(page_size, max_trials - len(trials)),
            "format"       : "json",
        }
        if next_token:
            params["pageToken"] = next_token

        try:
            response = requests.get(CTGOV_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"API request failed for '{condition}': {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            trial = _parse_ctgov_study(study, condition)
            trials.append(trial)

        next_token = data.get("nextPageToken")
        if not next_token:
            break

        time.sleep(0.3)  # rate limiting
        logger.info(f"  Fetched {len(trials)} trials so far for '{condition}'...")

    logger.info(f"Total fetched for '{condition}': {len(trials)}")
    return trials


def _parse_ctgov_study(study: dict, condition: str) -> dict:
    """
    Parse a raw ClinicalTrials.gov v2 study JSON into a flat dict.
    Maps to the same field names used in mock_rag_service.py.
    """
    proto   = study.get("protocolSection", {})
    id_mod  = proto.get("identificationModule", {})
    status  = proto.get("statusModule", {})
    desc    = proto.get("descriptionModule", {})
    elig    = proto.get("eligibilityModule", {})
    design  = proto.get("designModule", {})
    arms    = proto.get("armsInterventionsModule", {})
    cond_mod= proto.get("conditionsModule", {})

    # Interventions — join all intervention names
    interventions = ", ".join([
        i.get("name", "") for i in arms.get("interventions", [])
    ])

    # Phase
    phases = design.get("phases", [])
    phase  = ", ".join(phases) if phases else "N/A"

    return {
        "nct_number"         : id_mod.get("nctId", ""),
        "study_title"        : id_mod.get("briefTitle", ""),
        "recruitment_status" : status.get("overallStatus", ""),
        "brief_summary"      : desc.get("briefSummary", ""),
        "eligibility_criteria": elig.get("eligibilityCriteria", ""),
        "phase"              : phase,
        "conditions"         : ", ".join(cond_mod.get("conditions", [])),
        "keywords"           : ", ".join(cond_mod.get("keywords", [])),
        "interventions"      : interventions,
        "min_age"            : elig.get("minimumAge", ""),
        "max_age"            : elig.get("maximumAge", ""),
        "sex"                : elig.get("sex", ""),
        "study_url"          : f"https://clinicaltrials.gov/study/{id_mod.get('nctId', '')}",
        "disease"            : condition,   # tag with the condition we searched for
    }


def load_or_create_csv() -> pd.DataFrame:
    """
    Core CSV management function.

    Behaviour:
      - CSV does not exist → fetch from ClinicalTrials.gov, create CSV, return DataFrame
      - CSV exists         → load it, fetch fresh trials, append only NEW rows (by nct_number), save, return DataFrame

    Returns:
        DataFrame of all trials (existing + any new ones appended)
    """
    if not os.path.exists(CSV_PATH):
        # ── CREATE ────────────────────────────────────────────────────────────
        logger.info(f"CSV not found at '{CSV_PATH}'. Fetching from ClinicalTrials.gov...")
        all_trials = []
        for condition in CONDITIONS:
            trials = fetch_trials_from_ctgov(condition)
            all_trials.extend(trials)

        df = pd.DataFrame(all_trials)
        df.drop_duplicates(subset=["nct_number"], inplace=True)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"CSV created at '{CSV_PATH}' with {len(df)} trials.")
        return df

    else:
        # ── UPDATE ────────────────────────────────────────────────────────────
        logger.info(f"CSV found at '{CSV_PATH}'. Loading and checking for updates...")
        existing_df = pd.read_csv(CSV_PATH)
        existing_nct_ids = set(existing_df["nct_number"].dropna().astype(str))
        logger.info(f"  Existing trials in CSV: {len(existing_df)}")

        new_trials = []
        for condition in CONDITIONS:
            trials = fetch_trials_from_ctgov(condition)
            for trial in trials:
                if str(trial["nct_number"]) not in existing_nct_ids:
                    new_trials.append(trial)

        if new_trials:
            new_df     = pd.DataFrame(new_trials)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.drop_duplicates(subset=["nct_number"], inplace=True)
            updated_df.to_csv(CSV_PATH, index=False)
            logger.info(f"  Appended {len(new_trials)} new trials. Total now: {len(updated_df)}")
            return updated_df
        else:
            logger.info("  No new trials found. CSV is already up to date.")
            return existing_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EMBED TEXT
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


def trial_to_text(trial: dict) -> str:
    """Convert a trial dict to plain text for embedding and reranking."""
    return (
        f"Title: {trial.get('study_title', '')}. "
        f"Condition: {trial.get('conditions', '')}. "
        f"Disease: {trial.get('disease', '')}. "
        f"Keywords: {trial.get('keywords', '')}. "
        f"Eligibility: {trial.get('eligibility_criteria', '')}. "
        f"Phase: {trial.get('phase', '')}. "
        f"Status: {trial.get('recruitment_status', '')}. "
        f"Interventions: {trial.get('interventions', '')}. "
        f"Summary: {trial.get('brief_summary', '')}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD IN-MEMORY INDEX + VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def build_index(df: pd.DataFrame) -> tuple[list[dict], list[list[float]]]:
    """
    Embed all trials from the CSV DataFrame into an in-memory index.

    Args:
        df: DataFrame loaded from CSV

    Returns:
        trials    : list of trial dicts
        embeddings: corresponding 768-dim embedding vectors
    """
    logger.info(f"Building in-memory index for {len(df)} trials...")
    trials     = df.fillna("").to_dict(orient="records")
    embeddings = []

    for i, trial in enumerate(trials):
        text      = trial_to_text(trial)
        embedding = embed_text(text, task_type="RETRIEVAL_DOCUMENT")
        embeddings.append(embedding)
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(trials)} trials...")

    logger.info(f"Index ready — {len(trials)} trials embedded.")
    return trials, embeddings


def query_vector_search(
    patient_embedding : list[float],
    trials            : list[dict],
    embeddings        : list[list[float]],
    top_k             : int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Cosine similarity search over the in-memory index.

    Args:
        patient_embedding : 768-dim query vector
        trials            : list of trial dicts (from build_index)
        embeddings        : corresponding embedding vectors
        top_k             : number of candidates to return

    Returns:
        Top-k trial dicts ordered by cosine similarity
    """
    query  = np.array(patient_embedding)
    scored = []

    for idx, trial_embedding in enumerate(embeddings):
        doc        = np.array(trial_embedding)
        cosine_sim = np.dot(query, doc) / (
            np.linalg.norm(query) * np.linalg.norm(doc) + 1e-9
        )
        scored.append((cosine_sim, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_trials = [trials[idx] for _, idx in scored[:top_k]]

    logger.info(f"Vector search top {top_k}: {[t.get('nct_number') for t in top_trials]}")
    return top_trials


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — RERANK
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary : str,
    trials          : list[dict],
    top_k           : int = RERANK_TOP_K,
) -> list[dict]:
    """
    Rerank candidates using Vertex AI Ranking API.
    Falls back to vector search order if Ranking API fails.
    """
    try:
        client         = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project        = GCP_PROJECT_ID,
            location       = os.getenv("GCP_REGION", "us-central1"),
            ranking_config = "default_ranking_config"
        )

        records = [
            discoveryengine.RankingRecord(
                id      = str(t.get("nct_number", "")),
                title   = str(t.get("study_title", "")),
                content = trial_to_text(t)
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

        logger.info(f"Reranked {len(trials)} → top {len(reranked)}")
        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to vector search order.")
        return trials[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(
    patient_summary  : str,
    retrieved_trials : list[dict],
) -> str:
    """Generate clinical trial recommendations using Gemini."""
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

    prompt = f"""
You are a clinical trial matching assistant for TrialLink, an MLOps platform
that connects patients with relevant clinical trials.

Patient Profile:
{patient_summary}

Top Matching Clinical Trials (reranked by relevance):
{context}

Task:
Recommend the most suitable trials for this patient.
For each recommended trial explain specifically:
  - Why it matches the patient's condition and diagnosis
  - Whether the patient meets the eligibility criteria (age, sex, disease stage)
  - What intervention or treatment the trial offers
Be concise and clinically precise.
"""
    model    = GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def rag_pipeline(patient_summary: str) -> dict:
    """
    End-to-end RAG pipeline using local CSV as data source.

    Flow:
        load/update CSV → embed index → vector search → rerank → Gemini

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
    logger.info("Starting TrialLink RAG Pipeline (CSV backend)")
    logger.info("=" * 60)

    # Step 0 — Load or create CSV
    logger.info("Step 0: Loading/updating CSV...")
    df = load_or_create_csv()

    # Step 1 — Build index + embed patient
    logger.info("Step 1: Building index and embedding patient summary...")
    trials, embeddings  = build_index(df)
    patient_embedding   = embed_text(patient_summary, task_type="RETRIEVAL_QUERY")

    # Step 2 — Vector search → top 20
    logger.info(f"Step 2: Vector search (top {RETRIEVAL_TOP_K})...")
    candidates = query_vector_search(
        patient_embedding, trials, embeddings, top_k=RETRIEVAL_TOP_K
    )

    # Step 3 — Rerank → top 5
    logger.info(f"Step 3: Reranking {len(candidates)} candidates → top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials(patient_summary, candidates, top_k=RERANK_TOP_K)

    # Step 4 — Generate recommendation
    logger.info("Step 4: Generating recommendation with Gemini...")
    recommendation = generate_recommendation(patient_summary, reranked_trials)

    logger.info("RAG Pipeline complete.")
    logger.info("=" * 60)

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

    test_patient_diabetes = """
    45-year-old female with Type 2 diabetes diagnosed 3 years ago.
    HbA1c: 8.2%, BMI: 28, no prior insulin therapy.
    Currently on Metformin. No cardiovascular disease. Non-smoker.
    """

    test_patient_breast_cancer = """
    52-year-old female diagnosed with HER2-positive breast cancer, stage II.
    Post-menopausal. No prior targeted therapy. ECOG performance status 0.
    """

    for label, patient in [
        ("Diabetic Female", test_patient_diabetes),
        ("Breast Cancer Patient", test_patient_breast_cancer),
    ]:
        print("\n" + "=" * 60)
        print(f"TEST PATIENT: {label}")
        print("=" * 60)

        result = rag_pipeline(patient)

        print(f"\nCANDIDATES FROM VECTOR SEARCH ({len(result['candidates_before_rerank'])}):")
        for t in result["candidates_before_rerank"]:
            print(f"  - [{t.get('nct_number', 'N/A')}] {t.get('study_title', 'N/A')}")

        print(f"\nAFTER RERANKING (top {RERANK_TOP_K}):")
        for t in result["retrieved_trials"]:
            print(f"  - [{t.get('nct_number', 'N/A')}] {t.get('study_title', 'N/A')}")

        print("\nLLM RECOMMENDATION:")
        print(result["recommendation"])