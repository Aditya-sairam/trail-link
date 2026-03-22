"""
RAG Service - TrialLink Cloud Function
=======================================
HTTP-triggered Cloud Function that runs the full RAG pipeline for a patient.

Pipeline:
  1. Embed patient summary       (Vertex AI text-embedding-005)
  2. Query Vertex AI Vector Search
  3. Fetch matched trials from Firestore
  4. Rerank using Vertex AI Ranking API
  5. Generate recommendation using MedGemma

Trigger:
    POST https://<function-url>/
    Body: { "patient_id": "uuid-here" }

Response:
    {
        "patient_id"               : str,
        "patient_summary"          : str,
        "candidates_before_rerank" : list[dict],
        "retrieved_trials"         : list[dict],
        "recommendation"           : str
    }
"""

from __future__ import annotations

import os
import json
import logging
import functions_framework
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import firestore
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import aiplatform
import numpy as np
import flask
from datetime import datetime


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── Init Vertex AI ─────────────────────────────────────────────────────────────
vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID", "mlops-test-project-486922")
MODEL_PROJECT_ID     = os.getenv("MODEL_PROJECT_ID", "mlops-test-project-486922")
MEDGEMMA_ENDPOINT_ID = os.getenv("MEDGEMMA_ENDPOINT_ID", "mg-endpoint-bb15ba35-9f1b-4101-acda-037a1c2d3de0")
FIRESTORE_DB         = os.getenv("FIRESTORE_DATABASE", "clinical-trials-db")
PATIENT_DB           = os.getenv("PATIENT_DB", "patient-db-dev")
EMBEDDING_MODEL      = "text-embedding-005"
RETRIEVAL_TOP_K      = 20
RERANK_TOP_K         = 5
CONDITIONS           = ["diabetes", "breast_cancer"]


# ══════════════════════════════════════════════════════════════════════════════
# CLOUD FUNCTION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

@functions_framework.cloud_event
def run_rag_pipeline(cloud_event):
    """Pub/Sub triggered Cloud Function entry point"""
    import base64
    import json

    db = firestore.Client(
            project=os.getenv("GCP_PROJECT_ID"),
            database=os.getenv("TRAIL_SUGGESTIONS_STORE", "")
        )

    # Decode Pub/Sub message
    data = base64.b64decode(
        cloud_event.data["message"]["data"]
    ).decode("utf-8")
    message = json.loads(data)

    patient_id = message.get("patient_id")
    if not patient_id:
        logger.error("No patient_id in Pub/Sub message")
        return
    logger.info(f"Running RAG pipeline for patient: {patient_id}")
    result = rag_pipeline_for_patient(patient_id)
    logger.info(f"Pipeline complete: {result['recommendation'][:100]}...")
    db.collection("trial_suggestions").document(patient_id).set({
        "status"          : "completed",
        "patient_id"      : patient_id,
        "recommendation"  : result["recommendation"],
        "retrieved_trials": result["retrieved_trials"],
        "generated_at"    : datetime.utcnow().isoformat()
    })

   


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — EMBED TEXT
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    try:
        model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
        inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
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
        f"Keywords: {_get('keywords')}. "
        f"Phase: {_get('phase')}. "
        f"Status: {_get('recruitment_status')}. "
        f"Interventions: {_get('interventions')}. "
        f"Eligibility: {_get('eligibility_criteria')}. "
        f"Summary: {_get('brief_summary')}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — QUERY VERTEX AI VECTOR SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def query_vector_search(
    patient_embedding: list[float],
    top_k: int = RETRIEVAL_TOP_K,
) -> list[str]:
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=os.getenv("GCP_REGION", "us-central1"),
        api_transport="grpc"
    )
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name="1573491299300933632"
    )
    fetch_k = top_k * 3
    logger.info("Querying Vertex AI Vector Search...")

    results = index_endpoint.find_neighbors(
        deployed_index_id="clinical_trials_dev",
        queries=[patient_embedding],
        num_neighbors=fetch_k
    )

    matches = results[0]
    seen_nct_ids = {}
    for match in matches:
        nct_id = match.id.rsplit("_", 1)[0]
        score  = match.distance
        if nct_id and (nct_id not in seen_nct_ids or score < seen_nct_ids[nct_id]):
            seen_nct_ids[nct_id] = score

    sorted_trials = sorted(seen_nct_ids.items(), key=lambda x: x[1])
    top_nct_ids   = [nct_id for nct_id, _ in sorted_trials[:top_k]]
    logger.info(f"Vector search → top {top_k}: {top_nct_ids}")
    return top_nct_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FETCH MATCHED TRIALS FROM FIRESTORE
# ══════════════════════════════════════════════════════════════════════════════

def fetch_trials_from_firestore(nct_ids: list[str]) -> list[dict]:
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
# STEP 3.5 — RERANK
# ══════════════════════════════════════════════════════════════════════════════

def rerank_trials(
    patient_summary: str,
    trials: list[dict],
    top_k: int = RERANK_TOP_K,
) -> list[dict]:
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
        response  = client.rank(request=request)
        trial_map = {
            str(t.get("nct_number") or t.get("_doc_id", "")): t
            for t in trials
        }
        reranked = [
            trial_map[record.id]
            for record in response.records
            if record.id in trial_map
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
    patient_summary: str,
    retrieved_trials: list[dict],
) -> str:
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

    system_prompt = (
        "You are a clinical trial matching assistant for TrialLink. "
        "Evaluate each trial's eligibility criteria strictly against the patient profile. "
        "If any hard exclusion criterion is not met, state clearly the patient is ineligible and why. "
        "Only recommend trials where the patient meets all inclusion criteria and none of the exclusion criteria. "
        "Be concise and clinically precise."
    )

    user_prompt = f"""Patient Profile:
{patient_summary}

Top Matching Clinical Trials:
{context}

For each trial:
1. State ELIGIBLE or INELIGIBLE
2. Explain why — check condition, age, sex, stage, prior treatments
3. Describe the intervention if eligible
4. Flag any disqualifying criteria if ineligible"""

    try:
        region = os.getenv("GCP_REGION", "us-central1")
        aiplatform.init(project=MODEL_PROJECT_ID, location=region)
        endpoint = aiplatform.Endpoint(
            endpoint_name=f"projects/{MODEL_PROJECT_ID}/locations/{region}/endpoints/{MEDGEMMA_ENDPOINT_ID}"
        )
        response = endpoint.predict(instances=[{
            "@requestFormat": "chatCompletions",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user",   "content": [{"type": "text", "text": user_prompt}]}
            ],
            "max_tokens": 2048
        }])
        return response.predictions["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
# FULL RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def rag_pipeline(patient_summary: str) -> dict:
    logger.info("=" * 60)
    logger.info("Starting TrialLink RAG Pipeline")
    logger.info("=" * 60)

    logger.info("Step 1: Embedding patient summary...")
    patient_embedding = embed_text(patient_summary, task_type="RETRIEVAL_QUERY")
    logger.info(f"  Embedding dimensions: {len(patient_embedding)}")

    logger.info(f"Step 2: Querying Vector Search (top {RETRIEVAL_TOP_K})...")
    candidate_nct_ids = query_vector_search(patient_embedding, top_k=RETRIEVAL_TOP_K)

    if not candidate_nct_ids:
        logger.warning("No supported condition found for this patient")
        return {
            "patient_summary"          : patient_summary,
            "candidates_before_rerank" : [],
            "retrieved_trials"         : [],
            "recommendation"           : "No clinical trials found. TrialLink currently supports diabetes and breast cancer trials only.",
        }

    logger.info("Step 3: Fetching matched trials from Firestore...")
    candidates = fetch_trials_from_firestore(candidate_nct_ids)
    logger.info(f"  Fetched {len(candidates)} trial documents")

    logger.info(f"Step 3.5: Reranking {len(candidates)} → top {RERANK_TOP_K}...")
    reranked_trials = rerank_trials(patient_summary, candidates, top_k=RERANK_TOP_K)

    logger.info("Step 4: Generating recommendation...")
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
# PATIENT FETCH HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_patient_summary(patient_id: str) -> str:
    from data_models import Patient

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