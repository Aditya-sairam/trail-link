# TrialLink Data Pipeline — Airflow + GCP + Infrastructure as Code

The TrialLink Data Pipeline is a Dockerized Apache Airflow DAG that ingests clinical trial data from ClinicalTrials.gov, processes and batches it, and uploads structured outputs to Google Cloud Storage.

This pipeline demonstrates:

- Airflow-based DAG orchestration  
- Infrastructure as Code using Pulumi  
- Secure GCP integration with service accounts  
- Reproducible Docker-based local execution  
- Cloud-native storage architecture  

It is designed as part of the broader TrialLink MLOps platform for scalable clinical trial matching.

## Architecture

The workflow is organized into the following logical stages:

1. **Data Acquisition**
   - Fetches clinical trial data from the ClinicalTrials.gov API.
   - Handles pagination and rate limiting.
   - Writes raw data to structured CSV.

2. **Data Preprocessing**
   - Cleans and normalizes text fields.
   - Enriches data using condition-specific classifiers.
   - Applies business rule validation.

3. **Schema Validation**
   - Generates and validates dataset schema.
   - Runs in “warn” mode for raw data.
   - Runs in “enforce” mode for processed data.

4. **Quality & Anomaly Detection**
   - Detects missing values, outliers, and inconsistencies.
   - Produces structured anomaly reports.

5. **Statistics Generation**
   - Computes dataset summaries (distribution, counts, enrollment metrics).

6. **Bias Detection (Data Slicing)**
   - Performs demographic slicing (age, gender, geography).
   - Flags underrepresented subgroups.

7. **Cloud Integration**
   - Uploads outputs to Google Cloud Storage and Firestore.

The DAG ensures:
- Deterministic task ordering
- Parallel execution where applicable
- Retry logic and failure visibility
- Idempotent task behavior


## Project Structure
```
trail-link/
└── pipelines/
    ├── Dockerfile                  # Dockerized Airflow environment
    ├── requirements.txt            # Pipeline dependencies
    │
    └── dags/
        └── src/
            ├── clinical_trials_dag.py   # Main Airflow DAG definition
            ├── ingest.py                # Data acquisition (ClinicalTrials API)
            ├── schema.py                # Schema generation & validation
            ├── validate.py              # Business rule validation
            ├── quality.py               # Data cleaning & anomaly detection
            ├── stats.py                 # Statistical summaries
            ├── bias.py                  # Bias detection (data slicing)
            ├── gcs_upload.py            # Upload to Google Cloud Storage
            ├── firestore_upload.py      # Upload to Firestore
            │
            └── conditions/
                └── registry.py          # Condition-specific classifiers
```

## Data Acquisition
The pipeline fetches clinical trial data from the ClinicalTrials.gov API (v2) using a condition-driven query.

A pipeline run is triggered with a configuration parameter such as:

```json
{"condition": "diabetes"}
```

The pipeline only fetches the clinical trials which are actively recruiting 

### Ingestion Behavior

* Uses condition-specific search queries

* Handles API pagination (up to 1000 records per page)

* Applies controlled delays to avoid API throttling

* Flattens nested JSON responses into structured CSV format

* Standardizes key clinical trial attributes

The ingestion module writes raw output to:

```
data/<condition>/raw/clinical_trials.csv
```

The ingestion logic is implemented in:
```code
pipelines/dags/src/ingest.py
```
---

## Data Preprocessing

After ingestion, the pipeline performs structured preprocessing and validation through modular stages.

### 1. Raw Schema Validation

- Generates a baseline schema from the raw dataset  
- Validates required columns  
- Runs in **warn mode** to allow evolving API fields  
- Produces a schema validation report  

---

### 2. Data Enrichment

The pipeline enriches raw data using a condition-specific classifier defined in a registry pattern.

- Adds derived fields:
  - `disease`
  - `disease_type`
- Supports extensibility for new medical conditions without modifying DAG logic  

Enriched output is written to:

```
data/<condition>/enriched/enriched_trials.csv
```

---

### 3. Processed Schema Enforcement

- Runs schema validation in **enforce mode**
- Ensures enriched fields are present
- Fails the pipeline if structural violations occur

---

### 4. Business Logic Validation

- Detects logical inconsistencies (e.g., recruitment status conflicts)
- Generates structured validation reports

---

### 5. Data Cleaning & Quality Checks

The pipeline performs automated normalization and anomaly detection:

- Whitespace normalization  
- HTML entity decoding  
- Removal of invalid characters  
- Medical terminology normalization  
- Duplicate word cleanup  

It also detects:

- High null-percentage columns  
- Enrollment outliers  
- Status-date inconsistencies  

Quality and anomaly reports are generated for every pipeline run.

All preprocessing modules are implemented under:

```
pipelines/dags/src/
```

Each component follows a single-responsibility design to ensure modularity and testability.


## Statistics & Reporting

After preprocessing and validation, the pipeline generates structured statistical summaries for the enriched dataset.

### Generated Metrics

- Total number of trials
- Total number of columns
- Recruitment status distribution
- Disease subtype distribution
- Sex/gender distribution
- Enrollment statistics:
  - Mean
  - Median
  - Minimum
  - Maximum
  - Total enrollment

These statistics are computed from the enriched dataset and written to:

```
data/<condition>/reports/stats.json -> where is this ?
```


A consolidated pipeline summary is also generated to aggregate metadata and validation results.

Statistical computation is implemented in:
```
pipelines/dags/src/stats.py
```
This stage provides transparency into dataset structure and distribution before downstream usage or cloud upload.

---

## Bias Detection (Data Slicing)

To evaluate representation and fairness, the pipeline performs demographic slicing on the enriched dataset.

### Evaluated Slices

- Age groups:
  - Pediatric
  - Adult
  - Elderly
- Gender distribution
- Disease subtype distribution
- Geographic distribution (country-level)

### Bias Detection Rules

The pipeline flags warnings when:

- Pediatric representation is below 5%
- A gender group is underrepresented (< 10%)
- Geographic distribution is significantly imbalanced

Bias metrics and warnings are written to:

```
data/<condition>/reports/bias_report.json - where is this file ?
```

Bias detection logic is implemented in:
```
pipelines/dags/src/bias.py
```

This stage ensures demographic transparency and helps identify potential representation gaps in clinical trial participation.

---

## DAG Orchestration & Execution Flow

The pipeline is orchestrated using Apache Airflow through a single DAG:
```
clinical_trials_data_pipeline
```
### Execution Flow
```
fetch_raw
↓
schema_raw
↓
enrich
↓
schema_processed
↓
validate
↓
quality
↓
stats ─┐
bias ─┤→ save_reports
anomaly ─┘


NEED TO MENTION ON SCHEDULE 
```

---
## Testing & Reproducibility

The pipeline is designed to be reproducible, modular, and independently testable.

### Modular Testability

Each pipeline stage is implemented as a standalone module under:

```
pipelines/dags/src/
```

### Reproducibility

The pipeline ensures consistent behavior across environments through:

* Dockerized Airflow execution

* Environment-variable–driven configuration

* Deterministic task ordering

* Idempotent file outputs

* Infrastructure as Code (Pulumi) for cloud provisioning
---
## Data Versioning (DVC)

The project uses Data Version Control (DVC) to track and manage dataset versions independently of source code.

### Purpose

- Version raw and processed datasets
- Ensure reproducibility of pipeline outputs
- Maintain history of data changes
- Enable rollback to previous dataset states

### Tracked Artifacts

DVC tracks:

- Raw datasets
- Enriched datasets
- Generated reports (when required)
- Any intermediate pipeline outputs

DVC metadata files (`.dvc`) are stored alongside tracked data, while actual data is stored in configured remote storage.


---
## Tracking, Logging & Error Handling

The pipeline incorporates structured logging and failure handling at both the task and workflow levels.

### Airflow Task Monitoring

- Each task execution is logged through Airflow’s built-in logging system.
- Logs are accessible via the Airflow UI.
- Task retries are configured to handle transient failures.
- Failed tasks are clearly visible in Graph and Tree views.

---

### Structured Logging

Each module logs:

- Execution start and completion
- Number of records processed
- Validation results
- Detected anomalies
- Upload success or skip status

This ensures traceability across pipeline stages.

---

### Schema & Validation Failures

The pipeline differentiates between:

- **Warn Mode (Raw Schema)**  
  Logs violations but allows pipeline continuation.

- **Enforce Mode (Processed Schema)**  
  Raises errors and fails the pipeline if structural requirements are not met.

This prevents invalid enriched datasets from propagating downstream.

---

### Cloud Upload Safety

Before executing cloud upload tasks, the pipeline:

- Verifies required environment variables
- Checks credential availability
- Skips upload tasks safely if configuration is incomplete

This prevents accidental runtime failures in local mode.

---

## Alerts & Failure Handling

The pipeline implements execution-level alerts and validation-based safeguards.

### Validation-Based Failures

- Processed schema validation runs in **enforce mode**.
- Structural violations raise exceptions and fail the DAG.
- Business rule validation can short-circuit downstream tasks.
- Quality checks can stop execution when critical anomalies are detected.

These mechanisms prevent invalid datasets from propagating downstream.

---

### Airflow Execution Alerts

Airflow provides built-in visibility for failures:

- Failed tasks are clearly marked in the UI.
- Logs are accessible per task.
- Retry behavior is configurable.
- Execution timeouts are enforced per task.

---

### Anomaly Reporting

Detected anomalies are written to:
```
data/<condition>/reports/anomalies.json
```

---

## Pipeline Optimization & Parallelization

The pipeline is optimized for parallel execution and performance visibility.

### Parallel Condition Execution

Each medical condition (e.g., diabetes, breast cancer) runs in its own `TaskGroup`.

This enables:

- Independent execution per condition
- Parallel processing across conditions
- Isolated failure handling
- Scalable extension for new conditions

---

### Parallel Analytical Tasks

After validation and quality checks, the following tasks execute in parallel:

- `stats`
- `anomaly`
- `bias`

This reduces total runtime by allowing analytical stages to run concurrently.

---

### Controlled Short-Circuit Execution

The pipeline uses `ShortCircuitOperator` for:

- Validation checks
- Quality checks
- GCP configuration checks

This prevents unnecessary downstream execution when conditions are not met.

---

### Task Timeouts & Retries

Each task includes:

- Explicit execution timeouts
- Retry configuration with delay

This prevents long-running hangs and improves resilience against transient failures.

---

### Performance Monitoring

Airflow provides:

- Graph View (dependency visualization)
- Tree View (task state overview)
- Gantt View (runtime performance analysis)

The Gantt View is used to identify bottlenecks and optimize slow tasks when necessary.

---

This design ensures scalable execution, efficient resource usage, and clear observability across pipeline runs.




---

### Execution Environment

- Airflow runs locally via Docker Compose
- Infrastructure (GCS, IAM, Firestore) is provisioned via Pulumi
- GCP credentials are securely mounted into the Airflow container

The DAG ensures deterministic task execution and dependency management.

## Pipeline Output

```
gs://<bucket>/raw/YYYY-MM-DD/batch_001/YYYY-MM-DD_batch_001.csv
gs://<bucket>/raw/YYYY-MM-DD/batch_002/YYYY-MM-DD_batch_002.csv
gs://<bucket>/raw/YYYY-MM-DD/batch_003/YYYY-MM-DD_batch_003.csv
```

Each CSV contains up to 1000 clinical trials with fields: NCT_ID, Title, Status, Conditions, Eligibility_Criteria, Interventions, Location info, Sponsor info, and more.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (allocate at least 4GB memory)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed
- [Pulumi CLI](https://www.pulumi.com/docs/install/) installed
- Python 3.11+
- A GCP project with billing enabled

---

## Setup Guide

### Step 1: Get Repository Access

1. Ask the repo owner to add you as a **collaborator** with write access
2. Accept the invitation at [GitHub Notifications](https://github.com/notifications)
3. Clone and switch to the working branch:

```bash
git clone https://github.com/Aditya-sairam/trail-link.git
cd trail-link
git checkout datapipeline-infra
```

### Step 2: Create a GCP Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (e.g., `datapipeline-yourname`)
3. Enable billing on the project
4. Note your **Project ID** (the string, not the number)

### Step 3: Authenticate gcloud

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <YOUR_PROJECT_ID>
```

### Step 4: Deploy Infrastructure with Pulumi

This creates a GCS bucket and service account in your GCP project.

```bash
cd infra/pulumi_stacks

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize Pulumi
pulumi login --local
pulumi stack init <YOUR_STACK_NAME>
pulumi config set gcp:project <YOUR_PROJECT_ID>
pulumi config set gcp:region us-central1

# Enable required APIs
gcloud services enable \
  storage.googleapis.com \
  firestore.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  --project=<YOUR_PROJECT_ID>

# Deploy
pulumi preview
pulumi up
```

This creates:
- **GCS Bucket** (`triallink-pipeline-data-<YOUR_PROJECT_ID>`) — stores pipeline data
- **Service Account** (`pipeline-sa-<stack>`) — with GCS write access
- **Patient Infrastructure** — Firestore, Artifact Registry (for patient API)

### Step 5: Update Pipeline Config

Edit `pipelines/dags/src/clinical_trials_dag.py`:

1. Update the bucket name:
```python
BUCKET_NAME = os.getenv("GCS_BUCKET", "triallink-pipeline-data-<YOUR_PROJECT_ID>")
```

2. Update the GCS client project:
```python
client = storage.Client(project="<YOUR_PROJECT_ID>")
```

### Step 6: Configure Docker Compose for GCP Access

The `docker-compose.yaml` needs your local GCP credentials mounted so Airflow can upload to GCS.

Verify these are set in `docker-compose.yaml` under `x-airflow-common`:

**Volumes** (mounts your local GCP credentials into the container):
```yaml
volumes:
  - ${AIRFLOW_PROJ_DIR:-.}/dags:/opt/airflow/dags
  - ${AIRFLOW_PROJ_DIR:-.}/logs:/opt/airflow/logs
  - ${AIRFLOW_PROJ_DIR:-.}/config:/opt/airflow/config
  - ${AIRFLOW_PROJ_DIR:-.}/plugins:/opt/airflow/plugins
  - ~/.config/gcloud:/home/airflow/.config/gcloud:ro   # ← GCP credentials
```

**Environment** (add these to `&airflow-common-env`):
```yaml
GOOGLE_APPLICATION_CREDENTIALS: /home/airflow/.config/gcloud/application_default_credentials.json
GOOGLE_CLOUD_PROJECT: <YOUR_PROJECT_ID>
```

Make sure you've already run `gcloud auth application-default login` (Step 3) — this creates the credentials file that gets mounted.

### Step 7: Run Airflow Locally

```bash
cd pipelines

# Create required directories
mkdir -p logs plugins config

# Set Airflow user
echo "AIRFLOW_UID=$(id -u)" > .env

# Initialize the database (takes a couple minutes)
docker compose up airflow-init

# Start Airflow
docker compose up
```

Wait until you see:
```
airflow-webserver-1  | "GET /health HTTP/1.1" 200
```

### Step 7: Access Airflow UI

1. Open [localhost:8080](http://localhost:8080)
2. Login with `airflow2` / `airflow2`
3. Find `clinical_trials_pipeline` DAG
4. Toggle the switch to unpause it
5. Click the play button to trigger

Or via CLI:
```bash
docker compose exec airflow-scheduler airflow dags unpause clinical_trials_pipeline
docker compose exec airflow-scheduler airflow dags trigger clinical_trials_pipeline
```

### Step 8: Verify Data

```bash
gcloud storage ls -r gs://triallink-pipeline-data-<YOUR_PROJECT_ID>/raw/
```

### Step 9: Stop Airflow

```bash
docker compose down
```

---

## Project Structure

```
trail-link/
├── .github/workflows/
│   └── pipeline-build.yml          # GitHub Actions CI/CD workflow
├── env-mapping.json                 # GitHub username → variable prefix mapping
├── setup_github_oidc.sh             # One-time GCP setup for CI/CD
├── infra/
│   └── pulumi_stacks/
│       ├── __main__.py              # Pulumi entry point
│       ├── patientStack.py          # Patient API infrastructure
│       ├── datapipelineStack.py     # Data pipeline infrastructure
│       ├── Pulumi.yaml              # Project config
│       └── requirements.txt         # Pulumi dependencies
├── pipelines/
│   ├── docker-compose.yaml          # Local Airflow setup
│   ├── .env                         # Airflow user config
│   ├── dags/
│   │   ├── airflow.py               # DAG definition (tasks + dependencies)
│   │   └── src/
│   │       ├── __init__.py
│   │       └── clinical_trials_dag.py  # Pipeline functions
│   ├── logs/                        # Airflow logs (auto-generated)
│   ├── plugins/                     # Airflow plugins
│   └── config/                      # Airflow config
├── sdk/                             # Patient API
├── models/                          # ML models (future)
└── tests/                           # Test suite (future)
```

## DAG Tasks

```
fetch_trials → save_to_csv → upload_to_gcs → cleanup
```

| Task | Description |
|---|---|
| `fetch_trials` | Calls ClinicalTrials.gov API v2, paginates through all diabetes trials (RECRUITING, ACTIVE_NOT_RECRUITING, ENROLLING_BY_INVITATION) |
| `save_to_csv` | Flattens JSON into 33 columns, splits into batches of 1000 |
| `upload_to_gcs` | Uploads each batch to `raw/YYYY-MM-DD/batch_NNN/` in GCS |
| `cleanup` | Removes temporary local files |


## Useful Commands

```bash
# Airflow
docker compose up                # Start Airflow
docker compose down              # Stop Airflow
docker compose logs -f           # View all logs
docker compose exec airflow-scheduler airflow dags list          # List DAGs
docker compose exec airflow-scheduler airflow dags trigger <id>  # Trigger DAG

# GCS
gcloud storage ls -r gs://triallink-pipeline-data-<PROJECT>/raw/   # List data
gcloud storage cp gs://<path> .                                     # Download file

# Pulumi (from infra/pulumi_stacks/)
pulumi preview    # Dry run
pulumi up         # Deploy
pulumi refresh    # Sync state
pulumi destroy    # Tear down all resources
```

## Troubleshooting

**DAG not showing in UI?**
- Check import errors: `docker compose exec airflow-scheduler airflow dags list-import-errors`
- Restart scheduler: `docker compose restart airflow-scheduler`

**GCS upload failing?**
- Verify credentials are mounted: check `~/.config/gcloud/application_default_credentials.json` exists
- Verify project ID in `clinical_trials_dag.py`: `storage.Client(project="YOUR_PROJECT")`
- Re-authenticate: `gcloud auth application-default login`

**Pulumi errors?**
- Enable required APIs: `gcloud services enable storage.googleapis.com ...`
- Check project ID: `pulumi config get gcp:project`

## Cost Notes

- **GCS**: Pay per storage + operations. Minimal for this data volume.
- **Airflow**: Runs locally — no cloud cost.
- **Pulumi**: Free for individual use.