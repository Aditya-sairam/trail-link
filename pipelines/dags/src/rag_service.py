# pipelines/dags/src/mock_rag_service.py

"""
Mock RAG Service with Real Firestore - TrialLink
=================================================
Pipeline:
  1. Embed patient summary       (real Vertex AI text-embedding-005)
  2. Mock vector search          (cosine on in-memory index — no Vertex AI index yet)
  3. Fetch trials from Firestore (real Firestore — clinical_trials_diabetes / clinical_trials_breast_cancer)
  4. Rerank                      (real Vertex AI Ranking API)
  5. Generate recommendation     (real Gemini)

Only mocked:
  - _build_mock_index(): embeds real Firestore trials in memory instead of reading from Vertex AI index
  - query_vector_search(): cosine similarity instead of Vertex AI index endpoint call

Production swap (when Vertex AI Vector Search index is ready):
  - Delete _build_mock_index() and _MOCK_TRIALS, _MOCK_EMBEDDINGS
  - Replace query_vector_search() with the real implementation (see comment inside function)
  - Everything else stays identical
"""

from __future__ import annotations

import os
import logging
import numpy as np
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from google.cloud import firestore
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

# ── Init Vertex AI ─────────────────────────────────────────────────────────────
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID  = os.getenv("GCP_PROJECT_ID")
FIRESTORE_DB    = os.getenv("FIRESTORE_DB", "patient-db-sai")
EMBEDDING_MODEL = "text-embedding-005"
LLM_MODEL       = "gemini-2.0-flash"
RETRIEVAL_TOP_K = 20   # candidates retrieved from vector search (broad net)
RERANK_TOP_K    = 5    # final trials kept after reranking
CONDITIONS      = ["diabetes", "breast_cancer"]  # matches Firestore collection suffixes


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EMBED TEXT
# REAL + MOCK: Identical in both, no changes needed for production
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """
    Embed any text using Vertex AI text-embedding-005.

    Args:
        text     : Text to embed
        task_type: "RETRIEVAL_QUERY"    → patient summaries
                   "RETRIEVAL_DOCUMENT" → clinical trial documents

    Returns:
        768-dimensional embedding vector
    """
    try:
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
        embeddings = model.get_embeddings(inputs)
        return embeddings[0].values
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


def trial_to_text(trial: dict) -> str:
    """
    Convert a Firestore trial document to plain text for embedding and reranking.
    Field names match normalized snake_case columns from firestore_upload.py.

    Original CSV columns → Firestore fields (after normalize_column_name):
        NCT Number         → nct_number
        Study Title        → study_title
        Recruitment Status → recruitment_status
        Brief Summary      → brief_summary
        Eligibility Criteria → eligibility_criteria
        Phase              → phase
        Conditions         → conditions
        Keywords           → keywords
        Interventions      → interventions
        disease            → disease
    """
    return (
        f"Title: {trial.get('study_title') or trial.get('title', '')}. "
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
# MOCK ONLY — BUILD IN-MEMORY INDEX FROM REAL FIRESTORE DATA
# Fetches all trials from real Firestore and embeds them in memory.
# Simulates what Vertex AI Vector Search index holds in production.
# DELETE this entire section when real Vertex AI index is ready.
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_all_trials_for_index() -> list[dict]:
    """
    Fetch ALL trial documents from Firestore across all conditions.
    Used ONLY to build the mock in-memory index.

    Real Firestore collections:
        - clinical_trials_diabetes
        - clinical_trials_breast_cancer

    Returns:
        List of all trial dicts with _doc_id field added
    """
    db = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB)
    all_trials = []

    for condition in CONDITIONS:
        collection_name = f"clinical_trials_{condition}"
        try:
            docs = db.collection(collection_name).stream()
            count = 0
            for doc in docs:
                trial = doc.to_dict()
                trial["_doc_id"] = doc.id  # store Firestore doc ID (NCT number)
                all_trials.append(trial)
                count += 1
            logger.info(f"Fetched {count} trials from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to fetch from {collection_name}: {e}")

    logger.info(f"Total trials fetched for mock index: {len(all_trials)}")
    return all_trials


def _build_mock_index() -> tuple[list[dict], list[list[float]]]:
    """
    Embed all real Firestore trials in memory.
    Imitates what Vertex AI Vector Search index stores in production.

    Returns:
        trials    : list of trial dicts (real Firestore data)
        embeddings: corresponding list of 768-dim embedding vectors
    """
    logger.info("Building mock in-memory index from real Firestore data...")
    trials = _fetch_all_trials_for_index()

    embeddings = []
    for i, trial in enumerate(trials):
        text = trial_to_text(trial)
        embedding = embed_text(text, task_type="RETRIEVAL_DOCUMENT")
        embeddings.append(embedding)
        if (i + 1) % 10 == 0:
            logger.info(f"  Embedded {i + 1}/{len(trials)} trials...")

    logger.info(f"Mock index ready — {len(trials)} trials embedded")
    return trials, embeddings


# Build once at module load time (imitates a deployed, static index)
_MOCK_TRIALS, _MOCK_EMBEDDINGS = _build_mock_index()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QUERY VECTOR SEARCH
# MOCK   : cosine similarity against in-memory index of real Firestore trials
# REAL   : Vertex AI Vector Search index endpoint call (see swap comment below)
# ══════════════════════════════════════════════════════════════════════════════

def query_vector_search(
    patient_embedding: list[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=os.getenv("GCP_REGION","us-central1")
    )
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=os.getenv("VECTOR_SEARCH_ENDPOOINT_ID"))
    fetch_k = top_k*3
    query = np.array(patient_embedding)
    
    logger.info("Querying Vertx AI vector search..")
    
    results = index_endpoint.find_neighbors(
        deployed_index_id=os.getev("DEPLOYED_INDEX_ID"),
        queries=[patient_embedding],
        num_neighbors=fetch_k
    )
    
    matches = results[0]
    
    seen_nct_ids = {}
    for match in matches:
        nct_id = match.id.rsplit("_",1)[0]
        score = match.distance
        if nct_id and  nct_id not in seen_nct_ids or score < seen_nct_ids[nct_id]:
            seen_nct_ids[nct_id] = score 
    
    sorted_trials = sorted(seen_nct_ids.items(),key=lambda x:x[1])
    top_nct_ids = [nct_id for nct_id,_ in sorted_trials[:top_k]]

    logger.info(f"Vector search retrieved top {top_k}: {top_nct_ids}")
    return top_nct_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FETCH MATCHED TRIALS FROM REAL FIRESTORE
# REAL + MOCK: Identical in both, no changes needed for production
# ══════════════════════════════════════════════════════════════════════════════

def fetch_trials_from_firestore(nct_ids: list[str]) -> list[dict]:
    """
    Fetch specific trial documents by NCT ID from real Firestore.
    Searches across all condition collections since a patient query
    may match trials from multiple conditions (e.g. diabetes + breast_cancer).

    Firestore structure:
        Collection : clinical_trials_{condition}   e.g. clinical_trials_diabetes
        Document ID: {nct_number}                  e.g. NCT01234567

    Args:
        nct_ids: NCT numbers returned by vector search

    Returns:
        List of trial dicts fetched from Firestore (deduped by nct_number)
    """
    db = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB)
    trials     = []
    seen_ids   = set()

    for condition in CONDITIONS:
        collection_name = f"clinical_trials_{condition}"
        for nct_id in nct_ids:
            if nct_id in seen_ids:
                continue
            try:
                doc = db.collection(collection_name).document(nct_id).get()
                if doc.exists:
                    trial = doc.to_dict()
                    trial["_doc_id"] = doc.id
                    trials.append(trial)
                    seen_ids.add(nct_id)
            except Exception as e:
                logger.warning(f"Could not fetch {nct_id} from {collection_name}: {e}")

    logger.info(f"Fetched {len(trials)} trials from Firestore")
    return trials


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3.5 — RERANK USING VERTEX AI RANKING API
# REAL + MOCK: Identical in both, no changes needed for production
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary: str,
    trials: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """
    Rerank retrieved trials using Vertex AI Ranking API.
    Narrows top 20 candidates from vector search → top 5 most relevant.

    The Ranking API performs a deeper semantic comparison between the patient
    query and each trial document — significantly more accurate than cosine
    similarity alone, especially for clinical terminology like HbA1c, HER2+.

    Args:
        patient_summary : Original patient text used as the ranking query
        trials          : Candidate trials from vector search + Firestore fetch
        top_k           : How many trials to keep after reranking

    Returns:
        Reranked and trimmed list of trial dicts.
        Falls back to original vector search order if Ranking API fails.
    """
    try:
        client = discoveryengine.RankServiceClient()

        ranking_config = client.ranking_config_path(
            project=GCP_PROJECT_ID,
            location=os.getenv("GCP_REGION", "us-central1"),
            ranking_config="default_ranking_config"
        )

        # Each trial becomes a RankingRecord
        records = [
            discoveryengine.RankingRecord(
                id=str(t.get("nct_number") or t.get("_doc_id", "")),
                title=str(t.get("study_title") or t.get("title", "")),
                content=trial_to_text(t)
            )
            for t in trials
            if t.get("nct_number") or t.get("_doc_id")  # skip records without ID
        ]

        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model="semantic-ranker-512@latest",
            top_n=top_k,
            query=patient_summary,
            records=records
        )

        response = client.rank(request=request)

        # Map reranked record IDs back to full trial dicts
        trial_map = {
            str(t.get("nct_number") or t.get("_doc_id", "")): t
            for t in trials
        }
        reranked = [
            trial_map[record.id]
            for record in response.records
            if record.id in trial_map
        ]

        logger.info(
            f"Reranked {len(trials)} → top {len(reranked)}: "
            f"{[t.get('nct_number') for t in reranked]}"
        )
        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}. Falling back to vector search order.")
        return trials[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE RECOMMENDATION WITH GEMINI
# REAL + MOCK: Identical in both, no changes needed for production
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(
    patient_summary: str,
    retrieved_trials: list[dict],
) -> str:
    """
    Generate clinical trial recommendations using Gemini.
    Passes reranked trials as context alongside the patient profile.

    Args:
        patient_summary  : Patient profile plain text
        retrieved_trials : Reranked top-k trials from Firestore

    Returns:
        LLM-generated recommendation string
    """
    context = "\n\n".join([
        f"Trial {i + 1}:\n"
        f"  NCT ID       : {t.get('nct_number', 'N/A')}\n"
        f"  Title        : {t.get('study_title') or t.get('title', 'N/A')}\n"
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
    model = GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# REAL + MOCK: Identical orchestration in both
# ══════════════════════════════════════════════════════════════════════════════

def rag_pipeline(patient_summary: str) -> dict:
    """
    End-to-end RAG pipeline:
        embed → mock vector search → real Firestore → rerank → Gemini

    Args:
        patient_summary: Plain text description of the patient profile

    Returns:
        {
            "patient_summary"           : str,
            "candidates_before_rerank"  : list[dict],  # top 20 from vector search
            "retrieved_trials"          : list[dict],  # top 5 after reranking
            "recommendation"            : str
        }
    """
    logger.info("=" * 60)
    logger.info("Starting TrialLink RAG Pipeline")
    logger.info("=" * 60)

    # Step 1 — Embed patient summary
    logger.info("Step 1: Embedding patient summary...")
    patient_embedding = embed_text(patient_summary, task_type="RETRIEVAL_QUERY")
    logger.info(f"  Embedding dimensions: {len(patient_embedding)}")

    # Step 2 — Mock vector search → top 20 NCT IDs
    logger.info(f"Step 2: Mock vector search (retrieving top {RETRIEVAL_TOP_K} candidates)...")
    candidate_nct_ids = query_vector_search(patient_embedding, top_k=RETRIEVAL_TOP_K)

    # Step 3 — Fetch full trial docs from real Firestore
    logger.info("Step 3: Fetching matched trials from Firestore...")
    candidates = fetch_trials_from_firestore(candidate_nct_ids)
    logger.info(f"  Fetched {len(candidates)} trial documents")

    # Step 3.5 — Rerank → top 5
    logger.info(f"Step 3.5: Reranking {len(candidates)} candidates → top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials(patient_summary, candidates, top_k=RERANK_TOP_K)

    # Step 4 — Generate recommendation
    logger.info("Step 4: Generating recommendation with Gemini...")
    recommendation = generate_recommendation(patient_summary, reranked_trials)

    logger.info("RAG Pipeline complete")
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
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )

    # Test Patient 1 — Diabetic female
    test_patient_diabetes = """
    45-year-old female with Type 2 diabetes diagnosed 3 years ago.
    HbA1c: 8.2%, BMI: 28, no prior insulin therapy.
    Currently on Metformin. No cardiovascular disease. Non-smoker.
    """

    # Test Patient 2 — Breast cancer patient
    test_patient_breast_cancer = """
    52-year-old female diagnosed with HER2-positive breast cancer, stage II.
    Post-menopausal. No prior targeted therapy. ECOG performance status 0.
    """

    # Run pipeline
    print("\n" + "=" * 60)
    print("TEST PATIENT: Diabetic Female")
    print("=" * 60)

    result = rag_pipeline(test_patient_diabetes)

    print(f"\nCANDIDATES FROM VECTOR SEARCH ({len(result['candidates_before_rerank'])}):")
    for t in result["candidates_before_rerank"]:
        print(f"  - [{t.get('nct_number', 'N/A')}] {t.get('study_title') or t.get('title', 'N/A')}")

    print(f"\nAFTER RERANKING (top {RERANK_TOP_K}):")
    for t in result["retrieved_trials"]:
        print(f"  - [{t.get('nct_number', 'N/A')}] {t.get('study_title') or t.get('title', 'N/A')}")

    print("\nLLM RECOMMENDATION:")
    print(result["recommendation"])