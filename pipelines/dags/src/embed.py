"""
Embedding Pipeline
==================
Reads clinical trial documents from Firestore, builds rich text chunks,
embeds them using Vertex AI text-embedding-005 (same model as rag_service.py),
and upserts vectors into Vertex AI Vector Search index.

Marks each Firestore doc with embedded=True after success so
weekly re-runs only process new trials — never re-embeds old ones.

Env vars (set in datapipelineStack.py by GCP teammate):
    GCP_PROJECT_ID          e.g. "trial-link"
    GCP_REGION              e.g. "us-central1"
    VECTOR_SEARCH_INDEX_ID  from pulumi stack output  ← teammate fills this
    FIRESTORE_DATABASE      "clinical-trials-db"
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID      = os.getenv("GCP_PROJECT_ID", "")
REGION          = os.getenv("GCP_REGION", "us-central1")
INDEX_ID        = os.getenv("VECTOR_SEARCH_INDEX_ID", "")
FIRESTORE_DB    = os.getenv("FIRESTORE_DATABASE", "clinical-trials-db")

EMBEDDING_MODEL = "text-embedding-005"   # must match rag_service.py
TASK_TYPE       = "RETRIEVAL_DOCUMENT"   # document side of RAG
BATCH_SIZE      = 100                    # safe limit (Vertex AI max is 250)
SLEEP_SECS      = 1.0                    # pause between batches to stay within quota
CONDITIONS      = ["diabetes", "breast_cancer"]  # matches Firestore collection names


# ── Build text chunk from a Firestore trial doc ───────────────────────────────

def build_chunk(doc: dict) -> str:
    """
    Combine the most medically relevant fields into one text string.
    This is what gets embedded — must be rich enough for MedGemma to
    retrieve the right trials for a patient query.

    Field names are in snake_case because firestore_upload.py normalizes them.
    Matches the same fields used in trial_to_text() in rag_service.py.
    """
    def _get(*keys) -> str:
        for k in keys:
            v = doc.get(k)
            if v and str(v).strip() not in ("", "nan", "None"):
                return str(v).strip()
        return ""

    title         = _get("study_title", "title")
    conditions    = _get("conditions")
    disease       = _get("disease")
    disease_type  = _get("disease_type")
    phase         = _get("phase")
    status        = _get("recruitment_status")
    interventions = _get("interventions")
    keywords      = _get("keywords")
    summary       = _get("brief_summary")
    eligibility   = _get("eligibility_criteria")

    # Truncate very long fields to keep chunk under ~2000 tokens
    summary     = summary[:1500]     if len(summary) > 1500     else summary
    eligibility = eligibility[:1000] if len(eligibility) > 1000 else eligibility

    chunk = (
        f"Title: {title}. "
        f"Condition: {conditions}. "
        f"Disease: {disease}. "
        f"Keywords: {keywords}. "
        f"Eligibility: {eligibility}. "
        f"Phase: {phase}. "
        f"Status: {status}. "
        f"Interventions: {interventions}. "
        f"Summary: {summary}."
    )
    return chunk.strip()


# ── Embed a batch of texts ────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Vertex AI text-embedding-005.
    Returns a list of 768-dim float vectors, one per input text.
    """
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

    vertexai.init(project=PROJECT_ID, location=REGION)
    model   = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    inputs  = [TextEmbeddingInput(text=t, task_type=TASK_TYPE) for t in texts]
    results = model.get_embeddings(inputs)
    return [r.values for r in results]


# ── Upsert vectors into Vertex AI Vector Search ───────────────────────────────

def upsert_to_vector_search(datapoints: list[dict], index_id: str) -> None:
    """
    Push embeddings into Vertex AI Vector Search using streaming upsert.
    No index rebuild needed — vectors are available immediately.

    datapoints: [{"id": "NCT01234567", "embedding": [0.1, 0.2, ...]}, ...]
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types import IndexDatapoint

    aiplatform.init(project=PROJECT_ID, location=REGION)
    index = aiplatform.MatchingEngineIndex(index_name=index_id)

    datapoint_objs = [
        IndexDatapoint(
            datapoint_id=dp["id"],
            feature_vector=dp["embedding"],
        )
        for dp in datapoints
    ]

    index.upsert_datapoints(datapoints=datapoint_objs)
    log.info(f"  ✓ Upserted {len(datapoint_objs)} vectors to Vector Search")


# ── Embed one condition ───────────────────────────────────────────────────────

def embed_condition(
    condition: str,
    project_id: str,
    index_id: str,
    force_reembed: bool = False,
) -> int:
    """
    Embed all unembedded trials for one condition e.g. "diabetes".

    1. Read Firestore collection clinical_trials_{condition}
    2. Skip docs already marked embedded=True (unless force_reembed)
    3. Build text chunk per trial
    4. Embed in batches of 100
    5. Upsert to Vector Search
    6. Mark Firestore doc embedded=True + embedded_at timestamp

    Returns number of docs newly embedded.
    """
    from google.cloud import firestore

    db              = firestore.Client(project=project_id, database=FIRESTORE_DB)
    collection_name = f"clinical_trials_{condition}"
    log.info(f"Starting embedding for: {collection_name}")

    # Fetch only unembedded docs (saves cost on weekly runs)
    if force_reembed:
        docs = list(db.collection(collection_name).stream())
    else:
        docs = list(
            db.collection(collection_name)
            .where("embedded", "!=", True)
            .stream()
        )

    log.info(f"  Found {len(docs)} unembedded documents")
    if not docs:
        return 0

    # Build chunks
    records = []
    skipped = 0
    for doc in docs:
        data   = doc.to_dict()
        nct_id = data.get("nct_number") or data.get("nctnumber") or doc.id
        chunk  = build_chunk(data)

        if not chunk.strip():
            log.warning(f"  Skipping {nct_id}: empty chunk")
            skipped += 1
            continue

        records.append({"id": str(nct_id), "chunk": chunk, "ref": doc.reference})

    log.info(f"  Built {len(records)} chunks ({skipped} skipped)")

    # Embed + upsert in batches
    embedded_count = 0
    for batch_start in range(0, len(records), BATCH_SIZE):
        batch     = records[batch_start : batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        log.info(f"  Batch {batch_num}: embedding {len(batch)} texts...")

        texts   = [r["chunk"] for r in batch]
        vectors = embed_texts(texts)

        datapoints = [
            {"id": r["id"], "embedding": vec}
            for r, vec in zip(batch, vectors)
        ]
        upsert_to_vector_search(datapoints, index_id=index_id)

        # Mark docs as embedded in Firestore
        now = datetime.utcnow().isoformat()
        for r in batch:
            r["ref"].update({"embedded": True, "embedded_at": now})

        embedded_count += len(batch)
        log.info(f"  Batch {batch_num} done — total embedded so far: {embedded_count}")

        if batch_start + BATCH_SIZE < len(records):
            time.sleep(SLEEP_SECS)

    log.info(f"✓ [{condition}] Newly embedded: {embedded_count} | Skipped: {skipped}")
    return embedded_count


# ── Top-level entry point called from the DAG ─────────────────────────────────

def embed_conditions(
    conditions: list[str],
    project_id: str,
    index_id: Optional[str] = None,
    force_reembed: bool = False,
) -> dict[str, int]:
    """
    Embed all conditions. Called from the Airflow DAG task.

    Args:
        conditions:    ["diabetes", "breast_cancer"]
        project_id:    GCP project ID
        index_id:      Vertex AI Vector Search index ID (falls back to env var)
        force_reembed: True = re-embed everything, False = only new trials

    Returns:
        {"diabetes": 42, "breast_cancer": 17}  — newly embedded per condition
    """
    _index_id = index_id or INDEX_ID

    if not _index_id:
        raise ValueError(
            "VECTOR_SEARCH_INDEX_ID env var not set. "
            "GCP teammate needs to run pulumi up and add the index ID."
        )

    results = {}
    for condition in conditions:
        count = embed_condition(
            condition=condition,
            project_id=project_id,
            index_id=_index_id,
            force_reembed=force_reembed,
        )
        results[condition] = count

    total = sum(results.values())
    log.info(f"✓ Embedding complete — total: {total} | breakdown: {results}")
    return results
