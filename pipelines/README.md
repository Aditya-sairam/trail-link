# TrialLink Data Pipeline

Automated data pipeline that fetches clinical trial data from ClinicalTrials.gov, processes it into batched CSVs, and stores it in Google Cloud Storage. The pipeline runs locally via Apache Airflow in Docker containers.

## Architecture

```
Local Docker Compose → Airflow (localhost:8080) → DAG runs locally →
  1. Fetches data from ClinicalTrials.gov API
  2. Processes into batched CSVs (1000 trials per batch)
  3. Uploads to GCS bucket
  4. Cleans up local temp files
```

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