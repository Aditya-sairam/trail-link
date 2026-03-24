# TrialLink — Model Development & Evaluation

## Table of Contents

1. [Overview](#1-overview)
2. [Infrastructure Setup](#2-infrastructure-setup)
3. [Model Development and ML Code](#3-model-development-and-ml-code)
4. [Vectorization](#4-vectorization)
5. [Guardrails](#5-guardrails)
6. [Model Validation and Evaluation](#6-model-validation-and-evaluation)
7. [Model Bias Detection](#7-model-bias-detection)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Sensitivity Analysis](#9-model-sensitivity-analysis)
10. [Experiment Tracking and Results](#10-experiment-tracking-and-results)
11. [CI/CD Pipeline Automation](#11-cicd-pipeline-automation)
12. [Testing](#12-testing)
13. [Code Implementation](#13-code-implementation)
14. [Replication Guide](#14-replication-guide)

---

## 1. Overview

TrialLink is a clinical trial matching system that uses Retrieval-Augmented Generation (RAG) to match patients with relevant clinical trials from ClinicalTrials.gov. The system automates initial screening by embedding patient summaries, retrieving candidate trials via vector search, reranking them for clinical relevance, and generating eligibility recommendations using MedGemma.

Since TrialLink uses a **pre-trained large language model (MedGemma 4B-IT)** rather than training from scratch, the model development focus is on **pipeline validation, evaluation , retrieval quality, bias detection, sensitivity analysis, and automated guardrails** 

---

### RAG Pipeline Architecture
```
Patient Summary (text)
       │
       ▼
┌──────────────────────────┐
│  Input Guardrails         │  PII redaction → structural validation → LLM judge (Gemini)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Step 1: Embed Text       │  Vertex AI text-embedding-005 (768 dims)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Step 2: Vector Search    │  Vertex AI Matching Engine → top-20 candidates
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Step 3: Firestore Fetch  │  Retrieve full trial documents by NCT ID
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Step 3.5: Rerank         │  Vertex AI Ranking API (semantic-ranker-512) → top-5
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Step 4: MedGemma         │  Generate eligibility recommendation
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Output Guardrails        │  Policy checks → grounding checks → LLM judge (Gemini)
└──────────┬───────────────┘
           ▼
       Final Recommendation
```
---
## 2. Infrastructure Setup

> **Location**: `infra/pulumi_stacks/`

Infrastructure is provisioned using **Pulumi** with a GCS backend (`gs://pulumi-state-mlops-test-project-486922`). All resources are deployed to `mlops-test-project-486922` in `us-central1`.

### GCP Services Used

| Service | Purpose |
|---------|---------|
| Cloud Firestore | Trial storage (`clinical-trials-db`) and patient storage (`patient-db-dev`) |
| Vertex AI Matching Engine | Vector search index for trial embeddings |
| Vertex AI Endpoints | MedGemma 4B-IT dedicated endpoint |
| Vertex AI Embedding API | `text-embedding-005` for patient/trial embeddings |
| Vertex AI Ranking API | Semantic reranking (`semantic-ranker-512@latest`) |
| Vertex AI Generative Models | Gemini 2.5 Flash for guardrail judges |
| Cloud Functions v2 | RAG pipeline deployment (`rag-service-cloud-function`) |
| Pub/Sub | Trigger pipeline via `clinical-trial-suggestions-request` topic |
| Cloud Storage | Pipeline data (`triallink-pipeline-data-*`), Pulumi state |

### Pulumi Stack

- **Stack name**: `dev`
- **Backend**: `gs://pulumi-state-mlops-test-project-486922`
- **Deploy command**: `pulumi up --yes --stack dev`
- **State export**: `pulumi stack export --stack dev`

> _Detailed infrastructure documentation to be added by the infra teammate._

---
## 3. Model Development and ML Code

> **PDF Reference**: Section 2 — Model Development and ML Code

Since TrialLink uses a pre-trained model (MedGemma 4B-IT), there is no traditional model training step. Instead, the "model" is the full RAG pipeline — from embedding to recommendation generation. Development focused on optimizing retrieval quality, prompt engineering, and guardrail design.

### 3.1 Loading Data from the Data Pipeline

**PDF Reference**: Section 2.1

Clinical trial data flows through the following pipeline:

1. **Ingestion**: Apache Airflow DAGs (`pipelines/`) fetch trial data from the ClinicalTrials.gov API v2 for two supported conditions — diabetes and breast cancer (~1000 trials each).
2. **Storage**: Fetched trials are stored in Cloud Firestore (`clinical-trials-db`) in per-condition collections (`clinical_trials_diabetes`, `clinical_trials_breast_cancer`).
3. **Embedding**: A separate Airflow DAG converts stored trials into vector embeddings using Vertex AI `text-embedding-005` (768 dimensions).
4. **Indexing**: Embeddings are uploaded to a Vertex AI Matching Engine index for vector search.

At inference time, the RAG pipeline (`models/rag_service.py`) reads from Firestore and the Vector Search index — it does not re-run the data pipeline.

### 3.2 Training and Selecting the Best Model

**PDF Reference**: Section 2.2

Since TrialLink's core task is **clinical eligibility reasoning over structured medical text**, selecting the right language model required a thorough evaluation of models across capability, deployment constraints, and domain fit.

#### Models Considered

| Model | Type | Strengths | Limitations for TrialLink |
|-------|------|-----------|--------------------------|
| **GPT-4o** (OpenAI) | General-purpose LLM | Strong reasoning, broad medical knowledge | Proprietary API — no GCP-native deployment, data privacy concerns for PHI |
| **Gemini 1.5 Pro** (Google) | General-purpose LLM | Native Vertex AI support, long context | Not fine-tuned on clinical/biomedical data; weaker on eligibility criterion parsing |
| **BioMistral-7B** | Biomedical fine-tuned | Open-source, biomedical pretraining | Limited instruction-following for structured output |
| **Med-PaLM 2** (Google) | Medical LLM | Strong clinical benchmarks | Restricted access, not available for standard deployment |
| **Llama 3.1 8B Instruct** (Meta) | General instruction-tuned | Open-source, deployable on Vertex AI | No medical domain specialization; poor performance on eligibility criterion matching in testing |
| **MedGemma 4B-IT** (Google DeepMind) | Medical-domain fine-tuned | Purpose-built for clinical reasoning | Smaller context window than GPT-4o |

#### Why MedGemma 4B-IT Was Chosen

After evaluating the above candidates, **MedGemma 4B-IT** was selected as the recommendation model for the following reasons:

1. **Clinical domain specialization** — MedGemma is explicitly fine-tuned on medical literature, clinical notes, and biomedical text, making it significantly more accurate at parsing eligibility criteria (inclusion/exclusion logic, drug interactions, diagnosis codes) compared to general-purpose models.

2. **Instruction-following on structured prompts** — TrialLink's prompt requires the model to go criterion-by-criterion and output structured eligibility verdicts. MedGemma 4B-IT consistently followed this structure in testing, whereas general-purpose models of similar size tended to produce generic or inconsistent outputs.

3. **PHI-safe deployment** — Deploying on a dedicated Vertex AI endpoint ensures patient data never leaves the GCP project boundary, satisfying data governance requirements for healthcare applications.

4. **Cost-performance balance** — The 4B parameter variant delivers clinically sound reasoning at a fraction of the cost of GPT-4o or Gemini 1.5 Pro, making it viable for per-patient inference at scale.

#### Deployment Configuration

| Parameter | Value |
|-----------|-------|
| Model | MedGemma 4B-IT |
| Machine Type | g2-standard-8 |
| Accelerator | NVIDIA L4 × 1 |
| Request Format | `@requestFormat: chatCompletions` |
| Max Tokens | 2048 |

### 3.3 Model Validation

**PDF Reference**: Section 2.3

Model validation is performed in `tests/test_rag_pipeline.py` by running the full RAG pipeline against 13 diverse test patients across three categories:

- **Diabetes** (5 patients): Type 1, Type 2, prediabetes — varying age, sex, medications, lab values
- **Breast Cancer** (4 patients): HER2+, TNBC, HR+/HER2−, metastatic — varying stage, treatments, menopausal status
- **Out-of-Distribution** (4 patients): COPD, Alzheimer's, hypertension, obesity — tests system behavior on unsupported conditions

Validation checks run automatically after each pipeline execution:
```python
# Threshold check — fails CI if metrics drop below minimum
THRESHOLD_SUCCESS_RATE = 0.7    # at least 70% of patients get results
THRESHOLD_AVG_MATCHED  = 3.0    # at least 3 trials matched on average
```

All metrics are logged to MLflow on Databricks per run.

### 3.4 Model Bias Detection

**PDF Reference**: Section 2.4

Bias detection is covered in detail in [Section 7: Model Bias Detection](#7-model-bias-detection).

### 3.5 Pushing to Model/Artifact Registry

**PDF Reference**: Section 2.6

MedGemma is deployed from the Vertex AI Model Garden and versioned through Vertex AI's model registry. The current deployed model:
```
Project:    mlops-test-project-486922
Endpoint:   mg-endpoint-833ffeb4-d9e8-42e3-ae54-2a1a22a5777e
Model:      google-medgemma-medgemma-4b-it-1774290346 (v1)
Region:     us-central1
```

Pipeline artifacts (test results, per-patient JSONs, summary reports) are logged to Databricks MLflow after each CI run using `mlflow.log_artifacts()`.


Add here why this model , why best 

---

## 4. Vectorization

> **PDF Reference**: Supports Section 2.1 (Loading Data) and Section 8.1 (RAG Format)

Vectorization converts clinical trial documents and patient summaries into numerical embeddings for semantic similarity search.

### 4.1 Embedding Model

| Parameter | Value |
|-----------|-------|
| Model | Vertex AI `text-embedding-005` |
| Dimensions | 768 |
| Task Type (trials) | `RETRIEVAL_DOCUMENT` |
| Task Type (patients) | `RETRIEVAL_QUERY` |

### 4.2 Trial Text Representation

Each trial is converted to a single text string for embedding using `trial_to_text()` in `models/rag_service.py`:
```
Title: {study_title}. Condition: {conditions}. Disease: {disease}.
Keywords: {keywords}. Phase: {phase}. Status: {recruitment_status}.
Interventions: {interventions}. Eligibility: {eligibility_criteria}.
Summary: {brief_summary}.
```

### 4.3 Vector Search Index

| Parameter | Value |
|-----------|-------|
| Service | Vertex AI Matching Engine |
| Index Endpoint | `1573491299300933632` |
| Deployed Index ID | `clinical_trials_dev` |
| Distance Metric | Cosine similarity |
| Total Trials Indexed | ~2000 (diabetes + breast cancer) |
| Fetch Strategy | Retrieve `top_k × 3` neighbors, deduplicate by NCT ID, return `top_k` |

### 4.4 Query Flow

At inference time, the patient summary is embedded and used to query the vector index:
```
Patient Summary
       │
       ▼
  embed_text(summary, task_type="RETRIEVAL_QUERY")
       │
       ▼
  768-dim embedding
       │
       ▼
  query_vector_search(embedding, top_k=20)
       │
       ▼
  Fetch top_k × 3 = 60 neighbors
       │
       ▼
  Deduplicate by NCT ID (keep best score per trial)
       │
       ▼
  Return top-20 candidate NCT IDs
```

> _Detailed documentation on embedding generation, indexing pipeline, and re-indexing strategy to be added by the vectorization teammate._

---
## 5. Guardrails

> **PDF Reference**: Supports Section 2.3 (Model Validation) and Section 8.4 (Model Validation Code)
> **Location**: Integrated in `models/rag_service.py`

TrialLink implements a multi-layered guardrail system to ensure patient safety, prevent hallucination, and block prompt injection. Guardrails run both before (input) and after (output) the core RAG pipeline.

### 5.1 Input Guardrails

Three checks run sequentially before the pipeline processes a patient summary:

**Step 0A — PII Redaction**

Regex-based detection and redaction of personally identifiable information:

| PII Type | Pattern | Replacement |
|----------|---------|-------------|
| Email | `user@domain.com` | `[REDACTED_EMAIL]` |
| Phone | `(123) 456-7890` | `[REDACTED_PHONE]` |
| SSN | `123-45-6789` | `[REDACTED_SSN]` |
| Date of Birth | `DOB: 01/15/1980` | `[REDACTED_DOB]` |

PII hit counts are logged in the guardrail metadata for audit purposes.

**Step 0B — Structural Validation**

Rule-based checks on the patient summary:

- Input must not be empty
- Input must not exceed 12,000 characters
- Prompt injection patterns are blocked (e.g., "ignore previous instructions", "reveal system prompt")
- Input must contain at least 2 of 4 clinical signals: age, sex, condition/diagnosis, clinical data (labs/medications)

**Step 0C — LLM Input Judge (Gemini 2.5 Flash)**

An LLM-based classifier categorizes the input into one of five categories:

| Category | Action |
|----------|--------|
| `valid_supported_clinical_summary` | Proceed with pipeline |
| `valid_but_unsupported_condition` | Block — return guardrail response |
| `non_medical_or_irrelevant` | Block |
| `unsafe_or_prompt_injection` | Block |
| `missing_required_clinical_information` | Block |

If the LLM judge fails (e.g., rate limit, malformed response), the pipeline continues with structural checks only and the run is flagged.

**Configurable via environment variables:**
```
ENABLE_INPUT_LLM_GUARDRAIL=true|false
ENABLE_OUTPUT_LLM_GUARDRAIL=true|false
MAX_INPUT_CHARS=12000
```

### 5.2 Output Guardrails

Three checks run after MedGemma generates a recommendation:

**Step 5A — Policy Checks**

- Recommendation must not be empty
- Must not exceed 16,000 characters
- Banned patterns are flagged (dosage amounts, prescribing language like "take 500mg", "start medication", "I recommend starting")
- A medical disclaimer must be present — if missing, it is automatically appended

**Step 5B — Grounding Checks**

- Any NCT IDs mentioned in the recommendation must exist in the retrieved trials (prevents hallucinated trial references)
- Trial titles referenced must match the retrieved context
- Detects ungrounded trial references in structured output

**Step 5C — LLM Output Judge (Gemini 2.5 Flash)**

Evaluates the final recommendation for:

| Check | Criteria |
|-------|----------|
| `is_safe` | No medication dosage, prescribing advice, or treatment decisions |
| `is_grounded` | No trials outside the retrieved context, no invented evidence |

If the output fails any check, the recommendation is replaced with a safe fallback message directing the user to review trials manually, and the guardrail status is set to `flagged`.

### 5.3 Guardrail Metadata

Every pipeline run returns a `guardrail` object in the result:
```json
{
  "status": "passed | flagged | blocked",
  "stage": "completed | input_structure | input_llm_judge | output_policy | ...",
  "reason": "description of why guardrail triggered",
  "pii_hits": {"email": 0, "phone": 0, "ssn": 0, "dob": 0},
  "flag_reasons": [],
  "llm_input_judgment": { ... },
  "llm_output_judgment": { ... }
}
```

This metadata is captured in test results and logged to MLflow for tracking guardrail behavior across runs.

---

## 6. Model Validation and Evaluation

> **PDF Reference**: Section 2.3 (Model Validation) and Section 8.4 (Code for Model Validation)

Validation is performed at two levels: automated pipeline validation (quantitative metrics) and LLM-as-Judge evaluation (qualitative assessment).

### 6.1 Automated Pipeline Validation
**Location**: `tests/test_rag_pipeline.py`

The test pipeline runs the full RAG pipeline against 13 diverse test patients and validates results against predefined thresholds.

**Test Patient Design:**

| Slice | Count | Purpose |
|-------|-------|---------|
| Diabetes | 5 | Type 1, Type 2, prediabetes — varying age, sex, medications, HbA1c |
| Breast Cancer | 4 | HER2+, TNBC, HR+/HER2−, metastatic — varying stage, treatments |
| Out-of-Distribution | 4 | COPD, Alzheimer's, hypertension, obesity — tests unsupported conditions |

**Validation Thresholds:**

| Metric | Threshold | Description |
|--------|-----------|-------------|
| `success_rate` | ≥ 0.70 | At least 70% of patients must receive results |
| `avg_trials_matched` | ≥ 3.0 | Average of at least 3 matched trials per patient |

If either threshold is not met, CI fails and deployment is blocked.

**Metrics logged to MLflow per run:**

- `success_rate` — % of patients that completed the pipeline
- `avg_candidates_retrieved` — average vector search candidates
- `avg_trials_matched` — average trials after reranking
- `total_failed` / `total_success` — patient counts
- `total_guardrail_blocked` / `total_guardrail_flagged` — guardrail activity
- Per-patient metrics: `{patient_id}_candidates_retrieved`, `{patient_id}_trials_matched`

**Latest CI Results (GitHub Actions):**
```
Overall: success=100% | avg_candidates=18.46 | avg_matched=4.62

Slice metrics:
  avg_trials_matched_condition_diabetes:      5.0
  avg_trials_matched_condition_breast_cancer: 3.75
  avg_trials_matched_condition_ood:           5.0
  avg_trials_matched_age_group_adult:         4.44
  avg_trials_matched_age_group_elderly:       5.0
  avg_trials_matched_sex_female:              4.38
  avg_trials_matched_sex_male:                5.0
```

**Artifacts saved per run:**

- `test_results/rag_output.json` — full results for all 13 patients
- `test_results/rag_summary.json` — summary with top trials and recommendation previews
- `test_results/patients/{patient_id}.json` — per-patient detailed results including guardrail metadata

All artifacts are uploaded to Databricks MLflow and GitHub Actions artifacts.

### 6.2 LLM-as-Judge Evaluation (Gemini)

**Owner**: Teammate
**Location**: `models/evaluate_rag.py`
**Results**: `eval_results/`

A secondary evaluation layer uses Gemini 2.5 Flash as an independent judge to assess the quality of MedGemma's recommendations. This provides qualitative validation beyond the quantitative metrics in Section 6.1.

**Evaluation uses exactly 2 Gemini calls per patient:**

**Call 1 — Per-Trial Verdict:**
For each of the 5 reranked trials, checks patient profile against eligibility criteria and produces:

| Field | Description |
|-------|-------------|
| `verdict` | `ELIGIBLE` / `NOT ELIGIBLE` / `NEEDS REVIEW` |
| `fitness_score` | 1–5 score |
| `reasoning` | Paragraph citing specific criteria and patient data |
| `key_matches` | Criteria the patient meets |
| `key_concerns` | Potential issues |
| `disqualifying_criterion` | Exact exclusion criterion triggered, if any |

**Call 2 — Overall Assessment:**
Aggregates the per-trial verdicts into:

| Field | Description |
|-------|-------------|
| `overall_score` | 1–5 quality score |
| `top_trial` | Best matching NCT ID |
| `summary` | 2–3 sentence recommendation summary |
| `score_reasoning` | Why this score was given |

**Overall Score Guide:**

| Score | Meaning |
|-------|---------|
| 5 | Multiple strong eligible matches, clinically precise reasoning |
| 4 | At least one clear eligible match, minor gaps |
| 3 | Partial matches or borderline eligibility |
| 2 | Mostly ineligible or significant reasoning errors |
| 1 | All ineligible or critically flawed assessment |

**Output:**

- `eval_results/patients/{patient_id}_eval.json` — per-patient verdicts and scores
- `eval_results/summary.json` — aggregate metrics across all evaluated patients

---
## 7. Model Bias Detection

> **PDF Reference**: Section 6 (Model Bias Detection Using Slicing Techniques) and Section 2.4
> **Location**: `tests/test_rag_pipeline.py` — `compute_slice_metrics()` and `check_bias_alert()`

### 7.1 Slicing Strategy

Every test patient is tagged with three demographic attributes used for slicing:

| Slice Dimension | Groups | Purpose |
|----------------|--------|---------|
| `condition` | `diabetes`, `breast_cancer`, `ood` | Does retrieval quality vary by disease? |
| `age_group` | `adult` (18–59), `elderly` (60+) | Are older patients underserved? |
| `sex` | `male`, `female` | Is there gender-based disparity? |

### 7.2 Metrics Tracked Per Slice

For each slice group, the following metric is computed:
```
avg_trials_matched_{slice}_{group} = total matched trials / number of patients in group
```

All slice metrics are logged to MLflow per run, enabling tracking across experiments.

### 7.3 Bias Alert Mechanism

The `check_bias_alert()` function compares each slice's average against the overall average:

| Condition | Action |
|-----------|--------|
| Slice avg is > 2.0 trials below overall avg | **CI fails** — bias check raises `ValueError` |
| Slice avg is > 1.0 trials below overall avg | **Warning logged** — does not block CI |
| Slice avg is within 1.0 of overall avg | **Passes** — no action |
```python
BIAS_THRESHOLD = 2.0  # fails CI
WARNING_THRESHOLD = 1.0  # logs warning
```

### 7.4 Latest Bias Results

From the most recent CI run:
```
Overall avg_matched: 4.62

Slice metrics:
  condition_diabetes:      5.0    (within threshold ✅)
  condition_breast_cancer: 3.75   (within threshold ✅)
  condition_ood:           5.0    (within threshold ✅)
  age_group_adult:         4.44   (within threshold ✅)
  age_group_elderly:       5.0    (within threshold ✅)
  sex_female:              4.38   (within threshold ✅)
  sex_male:                5.0    (within threshold ✅)
```

No significant disparities detected. Breast cancer (3.75) and female (4.38) slices are slightly below average — this is due to the output guardrail flagging some MedGemma recommendations that contained dosage-related language, reducing the effective match count for those patients.

### 7.5 Bias Mitigation

**PDF Reference**: Section 6.3 and 6.4

If bias were detected (any slice > 2.0 below average), the following mitigation strategies would apply:

1. **Investigate root cause** — examine per-patient results in MLflow artifacts to determine whether the issue is retrieval quality, reranker behavior, or guardrail over-triggering
2. **Expand trial coverage** — if a condition has fewer indexed trials, increase the dataset for that condition
3. **Adjust guardrail sensitivity** — if the output guardrail disproportionately flags certain conditions (e.g., oncology trials mentioning dosages), tune the `BANNED_OUTPUT_PATTERNS` to reduce false positives
4. **Adjust reranker top-K** — increase `RERANK_TOP_K` for underperforming slices to give more trials a chance to pass through
5. **Re-run and validate** — after mitigation, re-run the pipeline and verify slice metrics improve without degrading other slices

All mitigation steps and trade-offs are documented in the MLflow run artifacts and GitHub Issue alerts.

---

## 8. Hyperparameter Tuning

> **PDF Reference**: Section 3 — Hyperparameter Tuning

Since TrialLink uses a pre-trained model (MedGemma 4B-IT), traditional hyperparameter tuning (grid search, Bayesian optimization over learning rate, epochs, etc.) does not apply. Instead, tuning focuses on the **RAG pipeline parameters** that control retrieval and recommendation quality.

### 8.1 Tunable Parameters

| Parameter | Default | Search Space | Effect |
|-----------|---------|-------------|--------|
| `RETRIEVAL_TOP_K` | 20 | [10, 20, 50] | Number of candidates from vector search |
| `RERANK_TOP_K` | 5 | [3, 5, 10] | Number of trials kept after reranking |
| `SIMILARITY_THRESHOLD` | 0.7 | [0.5, 0.7, 0.9] | Minimum cosine similarity score to include a trial |
| `MAX_TOKENS` | 2048 | — | MedGemma output length (fixed) |
| `EMBEDDING_MODEL` | `text-embedding-005` | — | Embedding model (fixed) |

### 8.2 Tuning Approach

We use a **one-at-a-time (OAT)** variation strategy: vary one parameter while holding others at their defaults. This isolates the effect of each parameter on pipeline performance.

Each configuration is run against all 13 test patients and logged as a separate MLflow run with:

- The varied parameter name and value
- Per-patient candidate and matched trial counts
- Aggregate metrics (avg_candidates, avg_matched, success_rate)

### 8.3 Selected Configuration

Based on the sensitivity analysis results (see [Section 9](#9-model-sensitivity-analysis)):

| Parameter | Selected Value | Rationale |
|-----------|---------------|-----------|
| `RETRIEVAL_TOP_K` | 20 | No improvement beyond 20 — reranker is the bottleneck |
| `RERANK_TOP_K` | 5 | Balances quality and coverage — 3 is too few, 10 adds noise |
| `SIMILARITY_THRESHOLD` | 0.7 | 0.5 filters out everything, 0.9 works but 0.7 is the safe floor |

Detailed results are in Section 9.

---

## 9. Model Sensitivity Analysis

> **PDF Reference**: Section 5 — Model Sensitivity Analysis
> **Location**: `tests/test_sensitivity_analysis.py`, `tests/test_feature_ablation.py`

Sensitivity analysis determines how the pipeline's performance changes with respect to different hyperparameters and input features. Two analyses were performed:

### 9.1 Hyperparameter Sensitivity

**Location**: `tests/test_sensitivity_analysis.py`
**MLflow Experiment**: `triallink-sensitivity-analysis`

Each parameter was varied one-at-a-time across 13 patients. Each configuration was logged as a separate MLflow run.

**Results:**
```
Run                                            Avg Candidates  Avg Matched    Avg OOD   Success
--------------------------------------------------------------------------------
RETRIEVAL_TOP_K_ret10_rer5_thr07                         10.0          5.0        5.0       1.0
RETRIEVAL_TOP_K_ret20_rer5_thr07                         20.0          5.0        5.0       1.0
RETRIEVAL_TOP_K_ret50_rer5_thr07                         50.0          5.0        5.0       1.0
RERANK_TOP_K_ret20_rer3_thr07                            20.0          3.0        3.0       1.0
RERANK_TOP_K_ret20_rer5_thr07                            20.0          5.0        5.0       1.0
RERANK_TOP_K_ret20_rer10_thr07                           20.0         10.0       10.0       1.0
SIMILARITY_THRESHOLD_ret20_rer5_thr05                     0.0          0.0        0.0       1.0
SIMILARITY_THRESHOLD_ret20_rer5_thr07                    20.0          5.0        5.0       1.0
SIMILARITY_THRESHOLD_ret20_rer5_thr09                    20.0          5.0        5.0       1.0
```

**Key Findings:**

- **`RETRIEVAL_TOP_K` has no effect on final output** — candidates increase (10→20→50) but matched trials stay at 5.0 because the reranker always caps at `RERANK_TOP_K`. This confirms the reranker is the bottleneck, not retrieval breadth.
- **`RERANK_TOP_K` directly controls output** — 3→5→10 maps linearly to matched trials. This is the most impactful parameter for controlling how many trials a patient sees.
- **`SIMILARITY_THRESHOLD=0.5` kills all results** — all vector search distances are above 0.5, so this threshold filters out everything. The threshold of 0.7 is the safe lower bound. 0.7 and 0.9 produce identical results, meaning all relevant trials have similarity scores between 0.5 and 0.7.

### 9.2 Feature Importance Analysis (Ablation Study)

**PDF Reference**: Section 5 — Feature Importance Analysis
**Location**: `tests/test_feature_ablation.py`
**MLflow Experiment**: `triallink-feature-ablation`

Since SHAP and LIME are designed for traditional ML models with structured inputs, we use an **ablation study** as the equivalent for our RAG pipeline. For each patient, we run the pipeline with the full summary (baseline), then re-run with one feature masked at a time and measure how much the retrieved trials change.

**Method:**

1. Run pipeline with full patient summary → baseline trial set
2. Remove one feature (e.g., diagnosis) from summary → re-run pipeline
3. Compare retrieved trials against baseline using overlap ratio
4. `importance = 1 - overlap_ratio` (higher = feature matters more)

**Features tested** (masked via regex patterns):

| Feature | What's Removed | Example |
|---------|---------------|---------|
| `diagnosis` | Primary condition | "Type 2 Diabetes", "HER2-positive breast cancer" |
| `age` | Patient age | "45-year-old" |
| `medications` | Current medications | "Metformin 500mg, Empagliflozin" |
| `lab_values` | Clinical observations | "HbA1c: 8.2%; BMI: 28", "BP 158/96 mmHg" |
| `smoking` | Smoking status | "Never smoker", "Former smoker, 40 pack-years" |

**Test patients**: 6 patients (2 per condition) to balance cost and coverage.

**LLM guardrails are disabled** during ablation (`ENABLE_INPUT_LLM_GUARDRAIL=false`) because ablated summaries are intentionally incomplete and would be blocked by the input guardrail.

**Results:**
```
Rank   Feature          Avg Importance
----------------------------------------
  1    diagnosis               0.900  ██████████████████
  2    lab_values              0.850  █████████████████
  3    age                     0.767  ███████████████
  4    medications             0.700  ██████████████
  5    smoking                 0.600  ████████████
```

**Key Findings:**

- **Diagnosis is the most important feature (0.900)** — removing the primary condition changes 90% of retrieved trials. This is clinically expected — the disease name is the primary signal for trial matching.
- **Lab values rank second (0.850)** — trial eligibility criteria contain specific thresholds (HbA1c > 7%, BMI > 30, eGFR ranges). Removing these significantly changes which trials match.
- **Age ranks third (0.767)** — most trials have age-based inclusion/exclusion criteria, making age a strong retrieval signal.
- **Medications (0.700)** — current medications affect exclusion criteria matching (e.g., "no prior insulin therapy").
- **Smoking (0.600)** — least important but still meaningful. Higher than expected, suggesting the embedding model treats smoking status as a meaningful signal even though most trial eligibility criteria don't heavily filter on it.

All five features have importance above 0.5, meaning the pipeline uses information from every part of the patient summary — no single feature is ignored.

---

## 10. Experiment Tracking and Results

> **PDF Reference**: Section 4 — Experiment Tracking and Results
> **Tracking Platform**: MLflow on Databricks
> **Databricks URL**: `https://dbc-b74b3877-4d11.cloud.databricks.com`

### 10.1 Tracking Setup

All experiments are logged to Databricks MLflow via the `MLFLOW_TRACKING_URI=databricks` environment variable. Authentication uses a Databricks Personal Access Token (PAT).

**MLflow Experiments:**

| Experiment | Purpose | Trigger |
|------------|---------|---------|
| `triallink-rag-pipeline` | Main evaluation — validation, bias, rollback | Every CI push + local runs |
| `triallink-sensitivity-analysis` | Hyperparameter sensitivity (9 configs) | Manual trigger |
| `triallink-feature-ablation` | Feature importance ablation (6 patients × 6 runs) | Manual trigger |

### 10.2 What Each Run Logs

**RAG Pipeline Run (`test_rag_pipeline.py`):**

| Category | What's Logged |
|----------|--------------|
| **Params** | `model`, `embedding_model`, `retrieval_top_k`, `rerank_top_k`, `total_test_patients`, `gcp_project`, `medgemma_endpoint` |
| **Metrics** | `success_rate`, `avg_candidates_retrieved`, `avg_trials_matched`, `total_failed`, `total_success`, `total_guardrail_blocked`, `total_guardrail_flagged` |
| **Slice Metrics** | `avg_trials_matched_condition_{diabetes,breast_cancer,ood}`, `avg_trials_matched_age_group_{adult,elderly}`, `avg_trials_matched_sex_{male,female}` |
| **Per-Patient Metrics** | `{patient_id}_candidates_retrieved`, `{patient_id}_trials_matched`, `{patient_id}_guardrail_blocked`, `{patient_id}_guardrail_flagged` |
| **Artifacts** | `test_results/rag_output.json`, `test_results/rag_summary.json`, `test_results/patients/{patient_id}.json` (13 files) |

**Sensitivity Analysis Run (`test_sensitivity_analysis.py`):**

| Category | What's Logged |
|----------|--------------|
| **Params** | `retrieval_top_k`, `rerank_top_k`, `similarity_threshold`, `varied_param`, `total_patients` |
| **Metrics** | `avg_candidates_retrieved`, `avg_trials_matched`, `avg_ood_trials_matched`, `success_rate` |
| **Per-Patient Metrics** | `{patient_id}_candidates`, `{patient_id}_matched` |
| **Artifacts** | `sensitivity/{run_name}/run_summary.json`, `sensitivity/{run_name}/patient_results.json` |

**Feature Ablation Run (`test_feature_ablation.py`):**

| Category | What's Logged |
|----------|--------------|
| **Params** | `features_tested`, `total_patients`, `total_runs` |
| **Metrics** | `avg_importance_{diagnosis,lab_values,age,medications,smoking}` |
| **Per-Patient Metrics** | `{patient_id}_{feature}_overlap`, `{patient_id}_{feature}_rank_shift`, `{patient_id}_{feature}_importance` |
| **Artifacts** | `ablation/ablation_summary.json` |

### 10.3 Model Selection

The current pipeline configuration was selected based on experiment tracking results across multiple runs:

1. **Retrieval strategy**: Vector search (top-20) + semantic reranker (top-5) was selected after comparing retrieval quality across configurations in the sensitivity analysis
2. **Validation**: Configuration must pass `success_rate ≥ 0.70` and `avg_matched ≥ 3.0`
3. **Bias check**: Configuration must not show > 2.0 trial disparity across any demographic slice
4. **Rollback check**: Configuration must not degrade > 20% from previous run

All selection decisions are traceable through MLflow run comparisons on Databricks.

### 10.4 Viewing Results

**Databricks MLflow UI:**
```
https://dbc-b74b3877-4d11.cloud.databricks.com/ml/experiments
```

**GitHub Actions Artifacts:**
Test results are also uploaded as GitHub Actions artifacts on every CI run and are downloadable from the Actions tab for 30 days.

---

## 11. CI/CD Pipeline Automation

> **PDF Reference**: Section 7 — CI/CD Pipeline Automation for Model Development
> **Owner**: Swarali (pipeline + alerts), Teammate (Pulumi deployment)
> **Location**: `.github/workflows/rag_ci.yml`

### 11.1 Pipeline Overview

The CI/CD pipeline is triggered on every push to `sanika-swarali-rag` and follows a **test-first, deploy-second** strategy. Deployment only proceeds if all evaluation checks pass.
```
Push to sanika-swarali-rag
       │
       ▼
┌──────────────────────────┐
│  Job 1: RAG Evaluation    │
│                           │
│  1. Run 13 test patients  │
│  2. Validation threshold  │──── FAIL ──┐
│  3. Bias detection        │            │
│  4. Rollback check        │            ▼
│  5. Log to MLflow         │    ┌───────────────┐
│  6. Upload artifacts      │    │ Job 3: Alert   │
└──────────┬───────────────┘    │                │
           │                     │ GitHub Issue    │
         PASS                    │ @mention team   │
           │                     │ Assign author   │
           ▼                     └───────────────┘
┌──────────────────────────┐
│  Job 2: Deploy            │
│                           │
│  Pulumi up --stack dev    │
└──────────────────────────┘
```

### 11.2 CI/CD Setup

**PDF Reference**: Section 7.1

| Setting | Value |
|---------|-------|
| CI Platform | GitHub Actions |
| Trigger | Push to `sanika-swarali-rag` branch |
| Runner | `ubuntu-latest` |
| Python | 3.11 |
| GCP Auth | Service account key (`GCP_SA_KEY` secret) |
| MLflow Backend | Databricks (`MLFLOW_TRACKING_URI=databricks`) |

### 11.3 Automated Model Validation

**PDF Reference**: Section 7.2

After the pipeline runs all 13 patients, three automated checks execute in sequence:

**Check 1 — Validation Threshold:**
```python
THRESHOLD_SUCCESS_RATE = 0.7   # 70% of patients must get results
THRESHOLD_AVG_MATCHED  = 3.0   # average 3+ trials matched
```
Fails CI if either metric is below threshold.

**Check 2 — Bias Detection:**
```python
BIAS_THRESHOLD = 2.0   # fails CI if any slice is >2.0 below average
```
Compares `avg_trials_matched` per slice (condition, age_group, sex) against overall average.

**Check 3 — Rollback:**
```python
DEGRADATION_TOLERANCE = 0.2   # 20% tolerance
```
Compares current run against the previous FINISHED MLflow run. If `success_rate` or `avg_matched` dropped more than 20%, CI fails.

If any check fails → deployment is blocked → alert is triggered.

### 11.4 Model Deployment

**PDF Reference**: Section 7.4

Deployment is handled by Pulumi in Job 2 and only runs if Job 1 (evaluation) passes. The RAG service is deployed as a Cloud Function v2 on GCP.
```yaml
- name: Deploy with Pulumi
  working-directory: ./infra/pulumi_stacks
  run: pulumi up --yes --stack dev
```

> _Pulumi deployment configuration is managed by the infra teammate in `infra/pulumi_stacks/`._

### 11.5 Notifications and Alerts

**PDF Reference**: Section 7.5

On failure, a GitHub Issue is automatically created with:

- Commit SHA and link
- Author assigned and `@mentioned`
- Links to CI logs and MLflow experiments
- Context-aware messaging:
  - If evaluation failed → "Deployment Blocked — Evaluation Failed"
  - If deployment failed → "Evaluation Passed but Deploy Failed"
- Possible causes listed (validation, bias, rollback, guardrails, infrastructure)
- Action items for resolution

All 5 team members are `@mentioned` in every failure issue, ensuring GitHub sends email notifications to everyone:
```
cc @Aditya-sairam @ItSwara @Sannn7 @Baidyam @Vaishnavi0805
```

Labels applied: `ci-failure`, `rag-pipeline`

### 11.6 Rollback Mechanism

**PDF Reference**: Section 7.6

Rollback operates at two levels:

**Level 1 — Metric-based rollback (pre-deployment):**

The `check_rollback()` function in `test_rag_pipeline.py` compares current metrics against the previous FINISHED MLflow run. If performance degraded more than 20%, CI fails before deployment is attempted. The previous production version remains live.
```python
def check_rollback(current_success_rate, current_avg_matched):
    # Query previous FINISHED run from MLflow
    # Compare with 20% degradation tolerance
    # Raise ValueError if degraded → blocks deployment
```

If no previous run exists, hardcoded baselines are used:
```python
BASELINE_SUCCESS_RATE = 0.7
BASELINE_AVG_MATCHED  = 3.0
```

**Level 2 — Infrastructure rollback (post-deployment failure):**

If the Pulumi deployment itself fails after evaluation passes, the previous infrastructure state remains unchanged since Pulumi only applies changes on success. No manual rollback is needed.

### 11.7 Environment Variables

All environment variables required for CI:

**GitHub Actions Variables (Settings → Variables):**

| Variable | Value |
|----------|-------|
| `GCP_PROJECT_ID` | `mlops-test-project-486922` |
| `MODEL_PROJECT_ID` | `mlops-test-project-486922` |
| `MODEL_PROJECT_NUMBER` | `903943936563` |
| `GCP_REGION` | `us-central1` |
| `FIRESTORE_DATABASE` | `clinical-trials-db` |
| `VECTOR_SEARCH_ENDPOINT_ID` | `1573491299300933632` |
| `DEPLOYED_INDEX_ID` | `clinical_trials_dev` |
| `MEDGEMMA_DEDICATED_DNS` | `mg-endpoint-*.prediction.vertexai.goog` |
| `DATABRICKS_HOST` | `https://dbc-b74b3877-4d11.cloud.databricks.com` |

**GitHub Actions Secrets (Settings → Secrets):**

| Secret | Description |
|--------|-------------|
| `GCP_SA_KEY` | GCP service account JSON key |
| `MEDGEMMA_ENDPOINT_ID` | MedGemma dedicated endpoint ID |
| `DATABRICKS_TOKEN` | Databricks PAT with files scope |
| `PULUMI_CONFIG_PASSPHRASE` | Pulumi stack encryption passphrase |

---

## 12. Testing
> **Location**: `tests/`

### 12.1 Test Files Overview

| File | Purpose | Trigger |
|------|---------|---------|
| `test_rag_pipeline.py` | End-to-end RAG evaluation — 13 patients, validation, bias, rollback | CI (every push) + local |
| `test_sensitivity_analysis.py` | Hyperparameter sensitivity — 9 configs × 13 patients | Manual (local or workflow_dispatch) |
| `test_feature_ablation.py` | Feature importance ablation — 6 patients × 6 runs | Manual (local or workflow_dispatch) |
| `test_guardrails.py` | Unit tests for guardrail functions | Local |
| `test_guardrails_e2e_live.py` | End-to-end guardrail tests against live GCP services | Local |
| `test_output_guardrails.py` | Output guardrail-specific tests | Local |
| `test_bias.py` | Bias detection unit tests | Local |
| `test_quality.py` | Retrieval quality checks | Local |
| `test_validate.py` | Validation logic tests | Local |
| `test_validity.py` | Input validity tests | Local |
| `test_schema.py` | Data schema validation | Local |
| `test_stats.py` | Statistical metric tests | Local |
| `test_ingest.py` | Data ingestion tests | Local |

### 12.2 Running Tests Locally

**Prerequisites:**

1. Python 3.11+ with a virtual environment
2. `.env` file in repo root (see [Replication Guide](#13-replication-guide))
3. GCP authentication: `gcloud auth application-default login`
4. MedGemma endpoint deployed and running
5. Vector Search index deployed and running

**Run the main evaluation pipeline:**
```bash
python tests/test_rag_pipeline.py
```

Expected output: 13 patients processed, validation/bias/rollback checks pass, results saved to `tests/test_results/` and logged to Databricks MLflow.

**Run sensitivity analysis:**
```bash
python tests/test_sensitivity_analysis.py
```

Expected runtime: ~45–60 minutes (9 configs × 13 patients = 117 pipeline calls). Results saved to `tests/test_results/sensitivity/`.

**Run feature ablation:**
```bash
python tests/test_feature_ablation.py
```

Expected runtime: ~30–45 minutes (6 patients × 6 runs = 36 pipeline calls). Results saved to `tests/test_results/ablation/`.

### 12.3 Test Output Structure
```
tests/
├── test_results/
│   ├── rag_output.json              # Full results for all 13 patients
│   ├── rag_summary.json             # Summary with top trials per patient
│   ├── patients/
│   │   ├── test_diabetes_001.json   # Per-patient detailed results
│   │   ├── test_diabetes_002.json
│   │   ├── ...
│   │   └── test_ood_obesity_001.json
│   ├── sensitivity/
│   │   ├── sensitivity_summary.json
│   │   ├── RETRIEVAL_TOP_K_ret10_rer5_thr07/
│   │   │   ├── run_summary.json
│   │   │   └── patient_results.json
│   │   ├── ...
│   │   └── SIMILARITY_THRESHOLD_ret20_rer5_thr09/
│   └── ablation/
│       └── ablation_summary.json
```

### 12.4 Running Tests in CI

The main evaluation (`test_rag_pipeline.py`) runs automatically on every push via `.github/workflows/rag_ci.yml`.

Sensitivity analysis and feature ablation are available as manual-trigger workflows:

- `.github/workflows/sensitivity-analysis.yml` — trigger from GitHub Actions tab → "Run workflow"
- `.github/workflows/feature-ablation.yml` — trigger from GitHub Actions tab → "Run workflow"

### 12.5 Test Patients

The 13 test patients are designed to cover diverse demographics and conditions:

**Diabetes (5 patients):**

| ID | Age | Sex | Diagnosis | Key Details |
|----|-----|-----|-----------|-------------|
| `test_diabetes_001` | 45 | F | Type 2 Diabetes | Metformin 500mg, HbA1c 8.2% |
| `test_diabetes_002` | 62 | M | Type 2 Diabetes + Hypertension | Metformin + Lisinopril, HbA1c 9.1% |
| `test_diabetes_003` | 55 | F | Type 2 Diabetes + Obesity | Metformin + Empagliflozin, HbA1c 7.8%, BMI 35 |
| `test_diabetes_004` | 38 | M | Type 1 Diabetes | Insulin pump, HbA1c 7.5%, CGM |
| `test_prediabetes_001` | 50 | M | Prediabetes + Metabolic Syndrome | No meds, HbA1c 6.1%, current smoker |

**Breast Cancer (4 patients):**

| ID | Age | Sex | Diagnosis | Key Details |
|----|-----|-----|-----------|-------------|
| `test_breast_cancer_001` | 52 | F | HER2+ stage II | Post-lumpectomy, no prior targeted therapy |
| `test_breast_cancer_002` | 44 | F | TNBC stage III | Post-mastectomy, 4 cycles chemo, no immunotherapy |
| `test_breast_cancer_003` | 61 | F | HR+/HER2− stage II | On Letrozole, no CDK4/6 inhibitor |
| `test_breast_cancer_004` | 48 | F | Metastatic HER2+ | Prior Trastuzumab + Pertuzumab, on Capecitabine |

**Out-of-Distribution (4 patients):**

| ID | Age | Sex | Diagnosis | Purpose |
|----|-----|-----|-----------|---------|
| `test_ood_copd_001` | 67 | M | COPD stage 3 | Tests unsupported respiratory condition |
| `test_ood_alzheimers_001` | 74 | F | Early Alzheimer's | Tests unsupported neurological condition |
| `test_ood_hypertension_001` | 58 | M | Resistant Hypertension + CKD | Tests unsupported cardiovascular condition |
| `test_ood_obesity_001` | 41 | F | Morbid Obesity + Sleep Apnea | Tests unsupported metabolic condition |

---

## 13. Code Implementation

> **PDF Reference**: Section 8 — Code Implementation

### 13.1 RAG Format

**PDF Reference**: Section 8.1

The entire model development process is implemented as a RAG system. The pipeline is deployed as a Cloud Function v2 triggered via Pub/Sub, ensuring reproducibility and portability.
```
models/
├── rag_service.py           # Full RAG pipeline — guardrails, embed, search, rerank, generate
├── evaluate_rag.py          # LLM-as-Judge evaluation using Gemini
├── data_models.py           # Patient data model
├── main.py                  # Entry point
├── alert_function/          # Alert Cloud Function
├── requirements.txt         # Python dependencies
└── __init__.py
```

### 13.2 Code for Loading Data from Data Pipeline

**PDF Reference**: Section 8.2

Trial data is loaded from Firestore at inference time via `fetch_trials_from_firestore()`:
```python
# models/rag_service.py
def fetch_trials_from_firestore(nct_ids: list[str]) -> list[dict]:
    db = firestore.Client(project=GCP_PROJECT_ID, database=FIRESTORE_DB)
    for condition in CONDITIONS:  # ["diabetes", "breast_cancer"]
        collection_name = f"clinical_trials_{condition}"
        for nct_id in nct_ids:
            doc = db.collection(collection_name).document(nct_id).get()
            ...
```

Patient data is loaded via `get_patient_summary()` which reads from the `patient-db-dev` Firestore database and converts to a text summary using the `Patient` data model.

### 13.3 Code for Model Validation

**PDF Reference**: Section 8.4

Validation is implemented in `tests/test_rag_pipeline.py` with three automated checks:
```python
# Threshold validation
def check_validation_threshold(success_rate, avg_matched):
    THRESHOLD_SUCCESS_RATE = 0.7
    THRESHOLD_AVG_MATCHED  = 3.0
    # Raises ValueError if below threshold → fails CI

# Bias detection
def check_bias_alert(slice_metrics, avg_matched):
    BIAS_THRESHOLD = 2.0
    # Raises ValueError if any slice > 2.0 below average → fails CI

# Rollback check
def check_rollback(current_success_rate, current_avg_matched):
    DEGRADATION_TOLERANCE = 0.2
    # Queries previous FINISHED MLflow run
    # Raises ValueError if degradation > 20% → fails CI
```

### 13.4 Code for Bias Checking

**PDF Reference**: Section 8.5

Bias is computed via `compute_slice_metrics()` which groups results by condition, age group, and sex:
```python
# tests/test_rag_pipeline.py
def compute_slice_metrics(results: list[dict]) -> dict:
    slices = {
        "condition" : {},   # diabetes, breast_cancer, ood
        "age_group" : {},   # adult, elderly
        "sex"       : {},   # male, female
    }
    # Computes avg_trials_matched per slice group
    # Returns dict logged to MLflow
```

### 13.5 Code for Sensitivity Analysis

**PDF Reference**: Section 8.3 (extended to RAG parameters)

Hyperparameter sensitivity is in `tests/test_sensitivity_analysis.py`:
```python
# Patches rag_service constants at runtime
rag_service.RETRIEVAL_TOP_K = retrieval_top_k
rag_service.RERANK_TOP_K    = rerank_top_k

# Monkeypatches query_vector_search for similarity threshold
def patched_query(patient_embedding, top_k=retrieval_top_k):
    # Applies similarity_threshold filter
    ...
```

Feature ablation is in `tests/test_feature_ablation.py`:
```python
# Masks one feature at a time via regex
FEATURE_MASKS = {
    "diagnosis":   [r"Active diagnoses:\s*[^.]+\."],
    "age":         [r"\d+-year-old"],
    "medications": [r"Current medications:\s*[^.]+\."],
    "lab_values":  [r"Recent observations:\s*[^.]+\.", ...],
    "smoking":     [r"Smoking status:\s*[^.]+\."],
}

def mask_feature(summary, feature_name):
    for pattern in FEATURE_MASKS[feature_name]:
        masked = re.sub(pattern, "", masked)
    return masked
```

### 13.6 Code for Model Selection after Bias Checking

**PDF Reference**: Section 8.6

Model selection follows a sequential gate in CI:

1. Validation threshold must pass → otherwise CI fails
2. Bias check must pass → otherwise CI fails
3. Rollback check must pass → otherwise CI fails
4. Only if all three pass → deployment proceeds

This is enforced by the job dependency in `.github/workflows/rag_ci.yml`:
```yaml
deploy:
  needs: rag-evaluation
  if: success()  # only runs if evaluation job passed
```

### 13.7 Code to Push Model to Registry

**PDF Reference**: Section 8.7

MedGemma is deployed from the Vertex AI Model Garden and managed through Vertex AI's built-in model registry:
```
Endpoint:  mg-endpoint-833ffeb4-d9e8-42e3-ae54-2a1a22a5777e
Model:     google-medgemma-medgemma-4b-it-1774290346 (v1)
```

Pipeline artifacts are pushed to Databricks MLflow after each run:
```python
# tests/test_rag_pipeline.py
mlflow.log_artifacts(RESULTS_DIR, artifact_path="test_results")
```

Deployment of the RAG service to Cloud Functions is handled via Pulumi:
```yaml
# .github/workflows/rag_ci.yml — Job 2
- name: Deploy with Pulumi
  working-directory: ./infra/pulumi_stacks
  run: pulumi up --yes --stack dev
```
---

## 14. Replication Guide

This section provides step-by-step instructions to replicate the TrialLink model development environment from scratch.

### 14.1 Prerequisites

- Python 3.11+
- Google Cloud SDK (`gcloud`) installed and configured
- GCP project with billing enabled
- Databricks account (free community edition works for MLflow)
- GitHub account with access to the repository

### 14.2 Clone and Setup
```bash
git clone https://github.com/Aditya-sairam/trail-link.git
cd trail-link
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r models/requirements.txt
pip install python-dotenv mlflow databricks-sdk
```

### 14.3 GCP Authentication

**Local development:**
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project mlops-test-project-486922
gcloud auth application-default set-quota-project mlops-test-project-486922
```

**CI (GitHub Actions):**

A service account JSON key is used via the `GCP_SA_KEY` secret. The service account requires the following IAM roles on `mlops-test-project-486922`:

| Role | Purpose |
|------|---------|
| `roles/aiplatform.user` | Vertex AI embeddings, Vector Search, MedGemma endpoint |
| `roles/datastore.user` | Firestore read/write |
| `roles/discoveryengine.editor` | Vertex AI Ranking API |
| `roles/iam.serviceAccountUser` | Deploy Cloud Functions as a service account |
| `roles/compute.viewer` | Pulumi requires region listing |
| `roles/cloudfunctions.developer` | Deploy Cloud Functions v2 |
| `roles/storage.admin` | Pulumi state in GCS |

### 14.4 Environment Variables

Create a `.env` file in the repo root:
```bash
# GCP
GCP_PROJECT_ID=mlops-test-project-486922
MODEL_PROJECT_ID=mlops-test-project-486922
MODEL_PROJECT_NUMBER=903943936563
GCP_REGION=us-central1
FIRESTORE_DATABASE=clinical-trials-db

# MedGemma
MEDGEMMA_ENDPOINT_ID=mg-endpoint-833ffeb4-d9e8-42e3-ae54-2a1a22a5777e
MEDGEMMA_DEDICATED_DNS=mg-endpoint-833ffeb4-d9e8-42e3-ae54-2a1a22a5777e.us-central1-903943936563.prediction.vertexai.goog

# Vector Search
VECTOR_SEARCH_ENDPOINT_ID=1573491299300933632
DEPLOYED_INDEX_ID=clinical_trials_dev

# MLflow / Databricks
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://dbc-b74b3877-4d11.cloud.databricks.com
DATABRICKS_TOKEN=<your-databricks-pat-token>

# Guardrails (set to false to skip LLM guardrails during testing)
ENABLE_INPUT_LLM_GUARDRAIL=true
ENABLE_OUTPUT_LLM_GUARDRAIL=true
```

**Important:** Add `.env` to `.gitignore` — it contains secrets.

### 14.5 GCP Services Setup

The following GCP services must be enabled on the project:
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  discoveryengine.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  pubsub.googleapis.com \
  storage.googleapis.com \
  --project=mlops-test-project-486922
```

### 14.6 MedGemma Deployment

MedGemma is deployed from the Vertex AI Model Garden:
```bash
# List available models
gcloud ai models list --region=us-central1 --project=mlops-test-project-486922

# Deploy to existing endpoint
gcloud ai endpoints deploy-model <ENDPOINT_ID> \
  --region=us-central1 \
  --project=mlops-test-project-486922 \
  --model=<MODEL_ID> \
  --display-name=medgemma-4b-it \
  --traffic-split=0=100 \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1

# Get the dedicated DNS for .env
gcloud ai endpoints describe <ENDPOINT_ID> \
  --region=us-central1 \
  --project=mlops-test-project-486922 \
  --format="value(dedicatedEndpointDns)"
```

**Note:** MedGemma incurs GPU costs while deployed. Undeploy when not in use:
```bash
gcloud ai endpoints undeploy-model <ENDPOINT_ID> \
  --region=us-central1 \
  --project=mlops-test-project-486922 \
  --deployed-model-id=<DEPLOYED_MODEL_ID>
```

### 14.7 Databricks MLflow Setup

1. Create a Databricks account at https://databricks.com
2. Generate a Personal Access Token: **Settings → Developer → Access tokens**
3. Ensure the token has **Workspace access** (includes files scope for artifact uploads)
4. Update `DATABRICKS_HOST` and `DATABRICKS_TOKEN` in your `.env`

MLflow experiments are created automatically on first run.

### 14.8 Running the Pipeline

**Step 1 — Verify GCP access:**
```bash
# Check authentication
gcloud auth list
gcloud config get-value project

# Verify Firestore access
python -c "
from google.cloud import firestore
db = firestore.Client(project='mlops-test-project-486922', database='clinical-trials-db')
docs = list(db.collection('clinical_trials_diabetes').limit(1).stream())
print(f'Firestore OK: {len(docs)} doc(s)')
"

# Verify MedGemma endpoint
gcloud ai endpoints describe <ENDPOINT_ID> \
  --region=us-central1 \
  --format="json(deployedModels)"
```

**Step 2 — Run the main evaluation:**
```bash
python tests/test_rag_pipeline.py
```

Expected: 13 patients processed, all checks pass, results logged to Databricks MLflow.

**Step 3 — Run sensitivity analysis (optional):**
```bash
python tests/test_sensitivity_analysis.py
```

Expected: 9 configurations × 13 patients = 117 pipeline calls. ~45–60 minutes.

**Step 4 — Run feature ablation (optional):**
```bash
python tests/test_feature_ablation.py
```

Expected: 6 patients × 6 runs = 36 pipeline calls. ~30–45 minutes.

### 14.9 GitHub Actions Setup

To replicate the CI pipeline, configure the following in your GitHub repo:

**Settings → Secrets and variables → Actions → Secrets:**

| Secret | Value |
|--------|-------|
| `GCP_SA_KEY` | Full JSON content of GCP service account key |
| `MEDGEMMA_ENDPOINT_ID` | MedGemma endpoint ID |
| `DATABRICKS_TOKEN` | Databricks PAT token |
| `PULUMI_CONFIG_PASSPHRASE` | Pulumi stack encryption passphrase |

**Settings → Secrets and variables → Actions → Variables:**

| Variable | Value |
|----------|-------|
| `GCP_PROJECT_ID` | `mlops-test-project-486922` |
| `MODEL_PROJECT_ID` | `mlops-test-project-486922` |
| `MODEL_PROJECT_NUMBER` | `903943936563` |
| `GCP_REGION` | `us-central1` |
| `FIRESTORE_DATABASE` | `clinical-trials-db` |
| `VECTOR_SEARCH_ENDPOINT_ID` | `1` |
| `DEPLOYED_INDEX_ID` | `clinical_trials_dev` |
| `MEDGEMMA_DEDICATED_DNS` | `mg-endpoint-*.prediction.vertexai.goog` |
| `DATABRICKS_HOST` | `https://dbc-b74b3877-4d11.cloud.databricks.com` |

**Settings → Labels:**

Create two labels: `ci-failure` and `rag-pipeline`

### 14.10 Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Permission denied on resource project` | Wrong GCP project in vertexai.init | Check `GCP_PROJECT_ID` in `.env` and run `gcloud auth application-default set-quota-project` |
| `Failed to resolve 'none'` | `MEDGEMMA_DEDICATED_DNS` not set | Add the dedicated DNS to `.env` |
| `invalid_scope` | Service account missing OAuth scopes | Ensure `google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])` in `generate_recommendation()` |
| `Rollback triggered` | Current metrics below previous run | Run again to establish new baseline, or increase `DEGRADATION_TOLERANCE` |
| `Input lacks sufficient clinical structure` | Guardrail rejecting valid input | Check that patient summary contains at least 2 of: age, sex, condition, clinical data |
| `Recommendation exceeds max length` | MedGemma generating verbose output | Not a blocker — guardrail replaces with safe fallback message |
| MLflow artifacts not uploading | Databricks PAT missing files scope | Regenerate PAT with Workspace access |
| `CONSUMER_INVALID` | Embedding call going to wrong project | Run `gcloud auth application-default set-quota-project mlops-test-project-486922` |











