"""
RAG Retrieval Analysis - TrialLink
====================================
Compares three retrieval strategies using MedGemma as the judge.
No hardcoded keywords or disease-specific rules.

MedGemma scores each retrieved trial 0-2:
  0 = clearly irrelevant or wrong subtype
  1 = partially relevant
  2 = strongly relevant and eligible

Strategies compared:
  A: Weighted vector search, top 5 directly       (no reranker)
  B: Simple cosine top 50, rerank to top 5
  C: Weighted cosine top 50, rerank to top 5      (current)

Run:
    export GCP_PROJECT_ID="datapipeline-infra"
    export MEDGEMMA_ENDPOINT_ID="mg-endpoint-..."
    python rag_analysis.py
"""

from __future__ import annotations

import os
import re
import json
import logging
import numpy as np
import pandas as pd
import vertexai

from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import discoveryengine_v1alpha as discoveryengine
from google.cloud import aiplatform

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

vertexai.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION", "us-central1")
)

# ── Config ─────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID")
GCP_REGION           = os.getenv("GCP_REGION", "us-central1")
EMBEDDING_MODEL      = "text-embedding-005"
MEDGEMMA_ENDPOINT_ID = os.getenv(
    "MEDGEMMA_ENDPOINT_ID",
    "mg-endpoint-645b70e1-a108-4645-adfa-ccc7f14c9de0"
)

CACHE_TITLE       = "embeddings_title.npy"
CACHE_CONDITION   = "embeddings_condition.npy"
CACHE_ELIGIBILITY = "embeddings_eligibility.npy"
TRIALS_CACHE      = "trials_cache.pkl"

WEIGHT_TITLE       = 0.2
WEIGHT_CONDITION   = 0.4
WEIGHT_ELIGIBILITY = 0.4
TOP_K_RETRIEVAL    = 50
TOP_K_FINAL        = 5

# ── Test patients — just free text, no keywords ────────────────────────────────
TEST_PATIENTS = {
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def embed_text(text: str, task_type: str = "RETRIEVAL_QUERY") -> np.ndarray:
    model  = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    inputs = [TextEmbeddingInput(text=text, task_type=task_type)]
    return np.array(model.get_embeddings(inputs)[0].values)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q     = query / (np.linalg.norm(query) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return (matrix / norms) @ q


def load_cache():
    for f in [CACHE_TITLE, CACHE_CONDITION, CACHE_ELIGIBILITY, TRIALS_CACHE]:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"Cache '{f}' not found. Run rag_only_swarali.py first."
            )
    title_embs = np.load(CACHE_TITLE)
    cond_embs  = np.load(CACHE_CONDITION)
    elig_embs  = np.load(CACHE_ELIGIBILITY)
    trials     = pd.read_pickle(TRIALS_CACHE).to_dict(orient="records")
    logger.info(f"Loaded {len(trials)} trials from cache.")
    return trials, title_embs, cond_embs, elig_embs


def call_reranker(query: str, trials: list[dict], top_k: int) -> list[dict]:
    try:
        client         = discoveryengine.RankServiceClient()
        ranking_config = client.ranking_config_path(
            project=GCP_PROJECT_ID, location=GCP_REGION,
            ranking_config="default_ranking_config"
        )
        records = [
            discoveryengine.RankingRecord(
                id      = str(t.get("nct_number", "")),
                title   = str(t.get("study_title", "")),
                content = (
                    f"Condition: {t.get('conditions', '')}. "
                    f"Eligibility: {t.get('eligibility_criteria', '')}. "
                    f"Disease: {t.get('disease', '')}."
                )
            )
            for t in trials if t.get("nct_number")
        ]
        response  = client.rank(request=discoveryengine.RankRequest(
            ranking_config = ranking_config,
            model          = "semantic-ranker-512@latest",
            top_n          = top_k,
            query          = query,
            records        = records
        ))
        trial_map = {str(t.get("nct_number", "")): t for t in trials}
        return [trial_map[r.id] for r in response.records if r.id in trial_map]
    except Exception as e:
        logger.error(f"Reranker failed: {e}")
        return trials[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# MEDGEMMA AS JUDGE
# ══════════════════════════════════════════════════════════════════════════════

def judge_with_medgemma(patient_summary: str, trials: list[dict]) -> list[dict]:
    """
    Ask MedGemma to score each trial for this patient.
    Returns trials with a 'relevance_score' (0, 1, or 2) and 'judge_reason' added.

    Scoring:
        2 = strongly relevant, patient likely eligible
        1 = partially relevant, some eligibility concerns
        0 = clearly irrelevant or patient ineligible

    No hardcoded disease rules — MedGemma uses its own medical knowledge.
    """
    trial_list = "\n\n".join([
        f"Trial {i+1} [{t.get('nct_number')}]:\n"
        f"  Title      : {t.get('study_title', '')}\n"
        f"  Condition  : {t.get('conditions', '')}\n"
        f"  Eligibility: {t.get('eligibility_criteria', '')[:300]}\n"
        f"  Min Age    : {t.get('min_age', '')}\n"
        f"  Max Age    : {t.get('max_age', '')}\n"
        f"  Sex        : {t.get('sex', '')}"
        for i, t in enumerate(trials)
    ])

    system_prompt = (
        "You are a clinical trial relevance judge. "
        "Given a patient profile and a list of clinical trials, "
        "score each trial's relevance for this specific patient. "
        "Use only the information provided. Do not assume anything not stated."
    )

    user_prompt = f"""Patient Profile:
{patient_summary}

Trials to Score:
{trial_list}

Score each trial:
  2 = strongly relevant — patient condition matches, likely eligible
  1 = partially relevant — related condition but eligibility concerns
  0 = not relevant — wrong condition, wrong subtype, or clearly ineligible

Respond ONLY with valid JSON, no explanation outside the JSON:
{{
  "scores": [
    {{"nct_number": "NCT...", "score": 2, "reason": "one sentence"}},
    ...
  ]
}}"""

    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    endpoint = aiplatform.Endpoint(MEDGEMMA_ENDPOINT_ID)

    response = endpoint.predict(instances=[{
        "@requestFormat": "chatCompletions",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]}
        ],
        "max_tokens": 1024
    }])

    raw = response.predictions["choices"][0]["message"]["content"]

    # Parse JSON from response
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
        scores_map = {
            s["nct_number"]: {"score": s["score"], "reason": s["reason"]}
            for s in parsed["scores"]
        }
    except Exception as e:
        logger.warning(f"Could not parse MedGemma judge response: {e}\nRaw: {raw}")
        scores_map = {}

    # Attach scores back to trial dicts
    scored_trials = []
    for t in trials:
        nct = str(t.get("nct_number", ""))
        s   = scores_map.get(nct, {"score": -1, "reason": "no score returned"})
        scored_trials.append({
            **t,
            "relevance_score": s["score"],
            "judge_reason"   : s["reason"],
        })

    return scored_trials


# ══════════════════════════════════════════════════════════════════════════════
# THREE STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def strategy_a(summary, trials, title_embs, cond_embs, elig_embs) -> list[dict]:
    """Weighted vector search, top 5 directly. No reranker."""
    q = embed_text(summary, "RETRIEVAL_QUERY")
    scores  = (
        WEIGHT_TITLE       * cosine_scores(q, title_embs) +
        WEIGHT_CONDITION   * cosine_scores(q, cond_embs)  +
        WEIGHT_ELIGIBILITY * cosine_scores(q, elig_embs)
    )
    top_idx = np.argsort(scores)[::-1][:TOP_K_FINAL]
    return [trials[i] for i in top_idx]


def strategy_b(summary, trials, title_embs, cond_embs, elig_embs) -> list[dict]:
    """Simple cosine top 50, rerank to top 5."""
    q       = embed_text(summary, "RETRIEVAL_QUERY")
    scores  = cosine_scores(q, cond_embs)
    top_idx = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
    return call_reranker(summary, [trials[i] for i in top_idx], TOP_K_FINAL)


def strategy_c(summary, trials, title_embs, cond_embs, elig_embs) -> list[dict]:
    """Weighted cosine top 50, rerank to top 5. Current approach."""
    q = embed_text(summary, "RETRIEVAL_QUERY")
    scores  = (
        WEIGHT_TITLE       * cosine_scores(q, title_embs) +
        WEIGHT_CONDITION   * cosine_scores(q, cond_embs)  +
        WEIGHT_ELIGIBILITY * cosine_scores(q, elig_embs)
    )
    top_idx    = np.argsort(scores)[::-1][:TOP_K_RETRIEVAL]
    candidates = [trials[i] for i in top_idx]
    return call_reranker(summary, candidates, TOP_K_FINAL)


# ══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_analysis():
    trials, title_embs, cond_embs, elig_embs = load_cache()

    strategies = {
        "A — Vector Only (no reranker)"  : strategy_a,
        "B — Simple Vector + Reranker"   : strategy_b,
        "C — Weighted Vector + Reranker" : strategy_c,
    }

    summary_table = []

    for patient_name, summary in TEST_PATIENTS.items():
        print("\n" + "═" * 70)
        print(f"PATIENT: {patient_name}")
        print("═" * 70)

        for strat_name, strat_fn in strategies.items():
            print(f"\n── {strat_name} ──")
            results = strat_fn(summary, trials, title_embs, cond_embs, elig_embs)

            # Judge with MedGemma
            scored  = judge_with_medgemma(summary, results)

            total_score = sum(t["relevance_score"] for t in scored if t["relevance_score"] >= 0)
            max_score   = TOP_K_FINAL * 2
            pct         = round(total_score / max_score * 100)

            print(f"  MedGemma Score: {total_score}/{max_score} ({pct}%)")
            for t in scored:
                icon = "✅" if t["relevance_score"] == 2 else ("⚠️ " if t["relevance_score"] == 1 else "❌")
                print(f"  {icon} [{t.get('nct_number')}] score={t['relevance_score']}  {t.get('study_title','')[:60]}")
                print(f"       → {t['judge_reason']}")

            summary_table.append({
                "patient"  : patient_name,
                "strategy" : strat_name,
                "score"    : total_score,
                "max"      : max_score,
                "pct"      : pct,
            })

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n\n" + "═" * 70)
    print("FINAL SUMMARY — MedGemma Relevance Score (higher = better)")
    print("═" * 70)
    df = pd.DataFrame(summary_table)
    pivot = df.pivot(index="strategy", columns="patient", values="pct")
    pivot["Average"] = pivot.mean(axis=1).round(0).astype(int)
    pivot = pivot.sort_values("Average", ascending=False)
    print(pivot.to_string())
    print("\nScores are % of max possible (10 points per patient, 2 per trial).")
    print("Best strategy to use in rag_only.py is the one with highest Average.")


if __name__ == "__main__":
    run_analysis()