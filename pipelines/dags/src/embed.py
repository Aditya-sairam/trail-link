"""
Embedding Pipeline
==================
Reads clinical trial documents from Firestore, builds rich text chunks,
embeds them using Vertex AI text-embedding-005 (same model as rag_service.py),
and upserts vectors into Vertex AI Vector Search index.

For long trials, the text is split into overlapping chunks — each chunk
gets its own vector with ID format: {NCT_ID}_{chunk_index}
e.g. NCT01234567_0, NCT01234567_1, NCT01234567_2

Marks each Firestore doc with embedded=True after success so
weekly re-runs only process new trials — never re-embeds old ones.

Env vars:
    GCP_PROJECT_ID          e.g. "trial-link"
    GCP_REGION              e.g. "us-central1"
    VECTOR_SEARCH_INDEX_ID  from pulumi stack output
    FIRESTORE_DATABASE      "clinical-trials-db"
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google.cloud import firestore
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID      = os.getenv("GCP_PROJECT_ID", "")
REGION          = os.getenv("GCP_REGION", "us-central1")
INDEX_ID        = os.getenv("VECTOR_SEARCH_INDEX_ID", "")
FIRESTORE_DB    = os.getenv("FIRESTORE_DATABASE", "clinical-trials-db")

EMBEDDING_MODEL = "text-embedding-005" 
TASK_TYPE       = "RETRIEVAL_DOCUMENT"
BATCH_SIZE      = 20   # reduced — 20 chunks * ~900 tokens = ~18k tokens, safely under 20k limit
SLEEP_SECS      = 1.0
CONDITIONS      = ["diabetes", "breast_cancer"]

# ── Chunking config ───────────────────────────────────────────────────────────
# ~1200 chars ≈ 900 tokens, well within the 3072 token limit
# 200 char overlap ensures context continuity between chunks
CHUNK_SIZE      = 1200
CHUNK_OVERLAP   = 200
MAX_TOKENS_PER_CALL = 18000  # stay safely under 20k limit
CHARS_PER_TOKEN     = 1.4    # conservative estimate (more chars per token = smaller batches)

# ── Build full text from a Firestore trial doc ────────────────────────────────

def build_full_text(doc: dict) -> str:
    """
    Combine all medically relevant fields into one text string.
    This full text is then split into chunks before embedding.
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
    phase         = _get("phase")
    status        = _get("recruitment_status")
    interventions = _get("interventions")
    keywords      = _get("keywords")
    summary       = _get("brief_summary")
    eligibility   = _get("eligibility_criteria")

    # Build full text without any truncation — chunking handles length
    text = (
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
    return text.strip()


# ── Split text into overlapping chunks ────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of roughly chunk_size characters.

    Overlap ensures that context at chunk boundaries isn't lost.
    For example with chunk_size=1200, overlap=200:
        chunk_0: text[0:1200]
        chunk_1: text[1000:2200]
        chunk_2: text[2000:3200]
        ...

    If the full text fits in one chunk, returns a single-element list.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap

    return chunks


# ── Embed a batch of texts ────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Vertex AI text-embedding-005.
    Automatically splits into sub-batches if total tokens would exceed
    the 20k per-call limit. ~0.75 tokens per character is a safe estimate.
    Returns a list of 768-dim float vectors, one per input text.
    """
    

    vertexai.init(project=PROJECT_ID, location=REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    all_vectors = []
    sub_batch   = []
    sub_tokens  = 0

    for text in texts:
        estimated_tokens = len(text) / CHARS_PER_TOKEN

        # If adding this text would exceed the limit, flush current sub-batch first
        if sub_batch and (sub_tokens + estimated_tokens) > MAX_TOKENS_PER_CALL:
            log.info(f"  Sub-batch: embedding {len(sub_batch)} texts (~{int(sub_tokens)} tokens)")
            inputs  = [TextEmbeddingInput(text=t, task_type=TASK_TYPE) for t in sub_batch]
            results = model.get_embeddings(inputs)
            all_vectors.extend([r.values for r in results])
            sub_batch  = []
            sub_tokens = 0

        sub_batch.append(text)
        sub_tokens += estimated_tokens

    # Flush remaining texts
    if sub_batch:
        log.info(f"  Sub-batch: embedding {len(sub_batch)} texts (~{int(sub_tokens)} tokens)")
        inputs  = [TextEmbeddingInput(text=t, task_type=TASK_TYPE) for t in sub_batch]
        results = model.get_embeddings(inputs)
        all_vectors.extend([r.values for r in results])

    return all_vectors


# ── Upsert vectors into Vertex AI Vector Search ───────────────────────────────

def upsert_to_vector_search(datapoints: list[dict], index_id: str) -> None:
    """
    Push embeddings into Vertex AI Vector Search using streaming upsert.

    datapoints: [{"id": "NCT01234567_0", "embedding": [0.1, 0.2, ...]}, ...]
    """
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
    Embed all unembedded trials for one condition.

    Each trial may produce multiple chunk vectors:
        NCT01234567_0, NCT01234567_1, NCT01234567_2 ...

    Returns number of trials (not chunks) newly embedded.
    """
    db              = firestore.Client(project=project_id, database=FIRESTORE_DB)
    collection_name = f"clinical_trials_{condition}"
    log.info(f"Starting embedding for: {collection_name}")

    if force_reembed:
        docs = list(db.collection(collection_name).stream())
    else:
        # Fetch all and filter in Python — handles missing field, False, and null
        all_docs = list(db.collection(collection_name).stream())
        docs = [
            doc for doc in all_docs
            if doc.to_dict().get("embedded") is not True
        ]

    log.info(f"  Found {len(docs)} unembedded documents")
    if not docs:
        return 0

    # Build chunks for all trials
    records = []
    skipped = 0

    for doc in docs:
        data   = doc.to_dict()
        nct_id = data.get("nct_number") or data.get("nctnumber") or doc.id
        text   = build_full_text(data)

        if not text.strip():
            log.warning(f"  Skipping {nct_id}: empty text")
            skipped += 1
            continue

        # Split into chunks — short trials get 1 chunk, long trials get multiple
        chunks = chunk_text(text)
        log.info(f"  {nct_id}: {len(chunks)} chunk(s) from {len(text)} chars")

        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{nct_id}_{i}",   # e.g. NCT01234567_0, NCT01234567_1
                "chunk": chunk,
                "ref": doc.reference,
                "nct_id": str(nct_id),
                "is_last_chunk": i == len(chunks) - 1,  # used to mark doc as embedded
            })

    total_chunks = len(records)
    log.info(f"  Built {total_chunks} total chunks from {len(docs) - skipped} trials ({skipped} skipped)")

    # Embed + upsert in batches
    embedded_trials = set()
    now = datetime.utcnow().isoformat()

    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch     = records[batch_start : batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        log.info(f"  Batch {batch_num}: embedding {len(batch)} chunks...")

        texts   = [r["chunk"] for r in batch]
        vectors = embed_texts(texts)

        datapoints = [
            {"id": r["id"], "embedding": vec}
            for r, vec in zip(batch, vectors)
        ]
        upsert_to_vector_search(datapoints, index_id=index_id)

        # Mark Firestore doc as embedded only after its LAST chunk is processed
        for r, _ in zip(batch, vectors):
            if r["is_last_chunk"]:
                r["ref"].update({"embedded": True, "embedded_at": now})
                embedded_trials.add(r["nct_id"])

        log.info(f"  Batch {batch_num} done")

        if batch_start + BATCH_SIZE < total_chunks:
            time.sleep(SLEEP_SECS)

    log.info(f"✓ [{condition}] Trials embedded: {len(embedded_trials)} | Chunks: {total_chunks} | Skipped: {skipped}")
    return len(embedded_trials)


# ── Top-level entry point called from the DAG ─────────────────────────────────

def embed_conditions(
    conditions: list[str],
    project_id: str,
    index_id: Optional[str] = None,
    force_reembed: bool = False,
) -> dict[str, int]:
    """
    Embed all conditions. Called from the Airflow DAG task.

    Returns:
        {"diabetes": 42, "breast_cancer": 17} — newly embedded trials per condition
    """
    _index_id = index_id or INDEX_ID

    if not _index_id:
        raise ValueError(
            "VECTOR_SEARCH_INDEX_ID env var not set. "
            "Run pulumi up and add the index ID."
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
    log.info(f"✓ Embedding complete — total trials: {total} | breakdown: {results}")
    return results