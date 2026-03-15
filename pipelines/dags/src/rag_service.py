# pipelines/dags/src/rag_service.py

"""
RAG Service - TrialLink
========================
Pipeline:
  1. Embed patient summary       (Vertex AI text-embedding-005)
  2. Query Vertex AI Vector Search (real index — chunks from embed.py)
  3. Fetch matched trials from Firestore (clinical_trials_diabetes / clinical_trials_breast_cancer)
  4. Rerank using Vertex AI Ranking API
  5. Generate recommendation using MedGemma (deployed on datapipeline-infra project)

Embedding pipeline (embed.py) stores vectors as:
    {NCT_ID}_{chunk_index}  e.g. NCT01234567_0, NCT01234567_1
query_vector_search() dedupes chunks back to unique NCT IDs before Firestore fetch.

Env vars:
    GCP_PROJECT_ID          e.g. "mlops-test-project-486922"   → Firestore + Vector Search
    MODEL_PROJECT_ID        e.g. "datapipeline-infra"          → MedGemma endpoint
    MEDGEMMA_ENDPOINT_ID    e.g. "4966380223210717184"
    GCP_REGION              e.g. "us-central1"
    FIRESTORE_DATABASE      e.g. "clinical-trials-db"
    VECTOR_SEARCH_ENDPOINT_ID
    DEPLOYED_INDEX_ID
"""

from __future__ import annotations

import os
import logging
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import firestore
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

# ── Init Vertex AI (for embeddings — uses GCP_PROJECT_ID) ─────────────────────
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
MODEL_PROJECT_ID     = os.getenv("MODEL_PROJECT_ID", "datapipeline-infra")
MEDGEMMA_ENDPOINT_ID = os.getenv("MEDGEMMA_ENDPOINT_ID", "4966380223210717184")
FIRESTORE_DB         = os.getenv("FIRESTORE_DATABASE", "clinical-trials-db")
EMBEDDING_MODEL      = "text-embedding-005"
RETRIEVAL_TOP_K      = 20   # candidates from vector search (broad net for reranker)
RERANK_TOP_K         = 5    # final trials after reranking
CONDITIONS           = ["diabetes", "breast_cancer"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EMBED TEXT
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """
    Embed text using Vertex AI text-embedding-005.

    Args:
        text     : Text to embed
        task_type: "RETRIEVAL_QUERY"    → patient summaries
                   "RETRIEVAL_DOCUMENT" → clinical trial documents (used by embed.py)

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
    Convert a Firestore trial document to plain text for reranking.
    Matches build_full_text() in embed.py exactly — same fields, same order.
    This ensures the reranker operates on the same text representation
    that was used to build the vector index.
    """
    def _get(*keys) -> str:
        for k in keys:
            v = trial.get(k)
            if v and str(v).strip() not in ("", "nan", "None"):
                return str(v).strip()
        return ""

    title         = _get("study_title", "title")
    conditions    = _get("conditions")
    disease       = _get("disease")
    phase         = _get("phase")
    status        = _get("recruitment_status")
    interventions = _get("interventions")
    keywords      = _get("keywords")
    summary       = _get("brief_summary")
    eligibility   = _get("eligibility_criteria")

    return (
        f"Title: {title}. "
        f"Condition: {conditions}. "
        f"Disease: {disease}. "
        f"Keywords: {keywords}. "
        f"Phase: {phase}. "
        f"Status: {status}. "
        f"Interventions: {interventions}. "
        f"Eligibility: {eligibility}. "
        f"Summary: {summary}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QUERY VERTEX AI VECTOR SEARCH
# embed.py stores chunks as {NCT_ID}_{chunk_index} e.g. NCT01234567_0
# We fetch top_k * 3 chunks then dedupe back to unique NCT IDs
# ══════════════════════════════════════════════════════════════════════════════

def query_vector_search(
    patient_embedding: list[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    """
    Query Vertex AI Vector Search index and return top_k unique NCT IDs.

    Since embed.py stores one vector per chunk (NCT01234567_0, NCT01234567_1...),
    we fetch top_k * 3 neighbors then dedupe by NCT ID, keeping the best
    scoring chunk per trial.

    Args:
        patient_embedding : 768-dim vector from embed_text()
        top_k             : Number of unique trials to return

    Returns:
        List of unique nct_number strings ordered by best chunk score
    """
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=os.getenv("GCP_REGION", "us-central1")
    )

    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv("VECTOR_SEARCH_ENDPOINT_ID")
    )

    # Fetch more neighbors than needed to account for multiple chunks per trial
    fetch_k = top_k * 3

    logger.info(f"Querying Vertex AI Vector Search (fetching {fetch_k} chunks)...")

    results = index_endpoint.find_neighbors(
        deployed_index_id=os.getenv("DEPLOYED_INDEX_ID"),
        queries=[patient_embedding],
        num_neighbors=fetch_k
    )

    matches = results[0]

    # Dedupe chunks → unique NCT IDs, keep best (lowest) distance per trial
    # chunk ID format: NCT01234567_0 → rsplit("_", 1)[0] → NCT01234567
    SIMILARITY_THRESHOLD = 0.7

    seen_nct_ids = {}
    for match in matches:
        nct_id = match.id.rsplit("_", 1)[0]
        score  = match.distance
        if score < SIMILARITY_THRESHOLD:
            if nct_id not in seen_nct_ids or score < seen_nct_ids[nct_id]:
                seen_nct_ids[nct_id] = score

    if not seen_nct_ids:
        logger.warning("No trials above similarity threshold — condition may not be supported")
        return []

    # Sort by best score → take top_k unique trials
    sorted_trials = sorted(seen_nct_ids.items(), key=lambda x: x[1])
    top_nct_ids   = [nct_id for nct_id, _ in sorted_trials[:top_k]]

    logger.info(f"Vector search → top {top_k} unique trials: {top_nct_ids}")
    return top_nct_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FETCH MATCHED TRIALS FROM FIRESTORE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_trials_from_firestore(nct_ids: list[str]) -> list[dict]:
    """
    Fetch specific trial documents by NCT ID from Firestore.
    Searches across all condition collections since a patient query
    may match trials from multiple conditions.

    Firestore structure:
        Database  : clinical-trials-db
        Collection: clinical_trials_{condition}  e.g. clinical_trials_diabetes
        Doc ID    : {nct_number}                 e.g. NCT01234567

    Args:
        nct_ids: NCT numbers returned by vector search

    Returns:
        List of trial dicts (deduped by NCT ID)
    """
    db       = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB)
    trials   = []
    seen_ids = set()

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
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary: str,
    trials: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
    """
    Rerank retrieved trials using Vertex AI Ranking API.
    Narrows top 20 candidates → top 5 most relevant.

    Args:
        patient_summary : Patient profile text (used as ranking query)
        trials          : Candidate trials from vector search + Firestore
        top_k           : How many trials to keep after reranking

    Returns:
        Reranked list of trials.
        Falls back to original vector search order if Ranking API fails.
    """
    try:
        client = discoveryengine.RankServiceClient()

        ranking_config = client.ranking_config_path(
            project=GCP_PROJECT_ID,
            location=os.getenv("GCP_REGION", "us-central1"),
            ranking_config="default_ranking_config"
        )

        records = [
            discoveryengine.RankingRecord(
                id=str(t.get("nct_number") or t.get("_doc_id", "")),
                title=str(t.get("study_title") or t.get("title", "")),
                content=trial_to_text(t)
            )
            for t in trials
            if t.get("nct_number") or t.get("_doc_id")
        ]

        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model="semantic-ranker-512@latest",
            top_n=top_k,
            query=patient_summary,
            records=records
        )

        response = client.rank(request=request)

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
# STEP 4 — GENERATE RECOMMENDATION WITH MEDGEMMA
# Uses MedGemma deployed on datapipeline-infra project endpoint
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(
    patient_summary: str,
    retrieved_trials: list[dict],
) -> str:
    """
    Generate clinical trial recommendations using MedGemma.
    MedGemma is deployed as a Vertex AI endpoint on datapipeline-infra project.

    Args:
        patient_summary  : Patient profile plain text
        retrieved_trials : Reranked top-k trials from Firestore

    Returns:
        MedGemma-generated recommendation string
    """
    context = "\n\n".join([
        f"Trial {i + 1}:\n"
        f"  NCT ID        : {t.get('nct_number', 'N/A')}\n"
        f"  Title         : {t.get('study_title') or t.get('title', 'N/A')}\n"
        f"  Condition     : {t.get('conditions', 'N/A')}\n"
        f"  Disease       : {t.get('disease', 'N/A')}\n"
        f"  Phase         : {t.get('phase', 'N/A')}\n"
        f"  Status        : {t.get('recruitment_status', 'N/A')}\n"
        f"  Eligibility   : {t.get('eligibility_criteria', 'N/A')}\n"
        f"  Min Age       : {t.get('min_age', 'N/A')}\n"
        f"  Max Age       : {t.get('max_age', 'N/A')}\n"
        f"  Sex           : {t.get('sex', 'N/A')}\n"
        f"  Interventions : {t.get('interventions', 'N/A')}\n"
        f"  Summary       : {t.get('brief_summary', 'N/A')}\n"
        f"  URL           : {t.get('study_url', 'N/A')}"
        for i, t in enumerate(retrieved_trials)
    ])

    prompt = f"""You are a clinical trial matching assistant for TrialLink, an MLOps platform
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

    try:
        region = os.getenv("GCP_REGION", "us-central1")

        # Init aiplatform with MedGemma's project (datapipeline-infra)
        aiplatform.init(project=MODEL_PROJECT_ID, location=region)

        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{MODEL_PROJECT_ID}/locations/{region}/endpoints/{MEDGEMMA_ENDPOINT_ID}"
        )

        response = endpoint.predict(
            instances=[{
                "prompt": prompt
            }]
        )

        # MedGemma returns predictions as a list — extract first result
        result = response.predictions[0]

        # Handle both string and dict response formats
        if isinstance(result, dict):
            return result.get("generated_text") or result.get("outputs") or str(result)
        return str(result)

    except Exception as e:
        logger.error(f"MedGemma generation failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def rag_pipeline(patient_summary: str) -> dict:
    """
    End-to-end RAG pipeline:
        embed → Vertex AI Vector Search → Firestore → rerank → MedGemma

    Args:
        patient_summary: Plain text description of the patient profile

    Returns:
        {
            "patient_summary"          : str,
            "candidates_before_rerank" : list[dict],  # top 20 from vector search
            "retrieved_trials"         : list[dict],  # top 5 after reranking
            "recommendation"           : str
        }
    """
    logger.info("=" * 60)
    logger.info("Starting TrialLink RAG Pipeline")
    logger.info("=" * 60)

    # Step 1 — Embed patient summary
    logger.info("Step 1: Embedding patient summary...")
    patient_embedding = embed_text(patient_summary, task_type="RETRIEVAL_QUERY")
    logger.info(f"  Embedding dimensions: {len(patient_embedding)}")

    # Step 2 — Vertex AI Vector Search → top 20 unique NCT IDs
    logger.info(f"Step 2: Querying Vertex AI Vector Search (top {RETRIEVAL_TOP_K})...")
    candidate_nct_ids = query_vector_search(patient_embedding, top_k=RETRIEVAL_TOP_K)

    if not candidate_nct_ids:
        logger.warning("No supported condition found for this patient")
        return {
            "patient_summary"          : patient_summary,
            "candidates_before_rerank" : [],
            "retrieved_trials"         : [],
            "recommendation"           : "No clinical trials found for this condition. TrialLink currently supports diabetes and breast cancer trials only.",
        }

    # Step 3 — Fetch full trial docs from Firestore
    logger.info("Step 3: Fetching matched trials from Firestore...")
    candidates = fetch_trials_from_firestore(candidate_nct_ids)
    logger.info(f"  Fetched {len(candidates)} trial documents")

    # Step 3.5 — Rerank → top 5
    logger.info(f"Step 3.5: Reranking {len(candidates)} candidates → top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials(patient_summary, candidates, top_k=RERANK_TOP_K)

    # Step 4 — Generate recommendation with MedGemma
    logger.info("Step 4: Generating recommendation with MedGemma...")
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
# PATIENT FETCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PATIENT_DB = os.getenv("PATIENT_DB", "patient-db-dev")


def get_patient_summary(patient_id: str) -> str:
    """
    Fetch patient from Firestore by ID and convert to text summary.

    Args:
        patient_id: UUID of patient in patient-db-dev

    Returns:
        Plain text summary ready for embedding
    """
    import sys
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../sdk/patient_package")
    ))
    from data_models import Patient

    db  = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    doc = db.collection("patients").document(patient_id).get()

    if not doc.exists:
        raise ValueError(f"Patient {patient_id} not found in Firestore")

    patient = Patient(**doc.to_dict())
    return patient.to_text_summary()


def rag_pipeline_for_patient(patient_id: str) -> dict:
    """
    Fetch patient from Firestore and run full RAG pipeline.

    Args:
        patient_id: UUID of patient in patient-db-dev

    Returns:
        Same structure as rag_pipeline()
    """
    logger.info(f"Fetching patient {patient_id} from Firestore...")
    summary = get_patient_summary(patient_id)
    logger.info(f"Patient summary: {summary}")
    return rag_pipeline(summary)


if __name__ == "__main__":
    import json
    from datetime import datetime

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )

    os.makedirs("test_results/patients", exist_ok=True)

    # Fetch real patients from Firestore and run RAG pipeline
    db   = firestore.Client(project=GCP_PROJECT_ID, database=PATIENT_DB)
    docs = list(db.collection("patients").limit(5).stream())

    if not docs:
        logger.warning("No patients found in Firestore")
    else:
        for doc in docs:
            patient_id = doc.id
            logger.info(f"\nRunning RAG pipeline for patient: {patient_id}")
            try:
                result = rag_pipeline_for_patient(patient_id)
                logger.info(f"Matched trials: {len(result['retrieved_trials'])}")
                logger.info(f"Recommendation:\n{result['recommendation']}")

                # Save results
                with open(f"test_results/patients/{patient_id}.json", "w") as f:
                    json.dump({
                        "patient_id"               : patient_id,
                        "patient_summary"          : result["patient_summary"],
                        "candidates_before_rerank" : result["candidates_before_rerank"],
                        "retrieved_trials"         : result["retrieved_trials"],
                        "recommendation"           : result["recommendation"],
                        "timestamp"                : datetime.utcnow().isoformat()
                    }, f, indent=2, default=str)

                logger.info(f"Saved → test_results/patients/{patient_id}.json")

            except Exception as e:
                logger.error(f"Failed for patient {patient_id}: {e}")