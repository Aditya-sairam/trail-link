# TrialLink - Clinical Trial Matching Platform

MLOps-powered platform for connecting patients with relevant clinical trials using RAG (Retrieval Augmented Generation) and automated data pipelines.

## Data Pipeline Assignment 

For the Data Pipeline assignment we have a local Airflow running on Docker compose which fetches data from the Clinical Trials API , for the diabetes / breast cancer clinical trials which are recruiting paitents currently. 

The main Data Processing , fetching code is written in :

```
trail-link/pipelines/
```

The folder has a README.d with more deatils on the bias , anamolies and alerts

GCP Resources are defined as infra in 

```
trail-link/infra/datapipelineStack.py
```

Details on how to execute the pipeline are in this ReadME.md

---

## Table of Contents
- [Running Locally](#running-locally)
- [Running on Google Cloud](#running-on-google-cloud)
- [Unit Tests](#unit-tests)
- [Project Structure](#project-structure)
  

---
## Embedding Model
We use `text-embedding-005`, Vertex AI's most recent stable embedding model, which outputs 768-dimensional vectors — a balance between retrieval quality and storage/compute efficiency. It is natively supported by Vertex AI Vector Search, keeping the entire pipeline within GCP with no cross-cloud API calls. The `RETRIEVAL_DOCUMENT` task type is optimized for asymmetric search (short query → long document), which is exactly the pattern a clinical trial RAG system requires. It is also compatible with MedGemma, making it a natural fit for the medical domain.

> **Note on alternatives:** We evaluated `text-embedding-large-exp-03-07` (3072-dim), but adopting it would require re-embedding all trials, rebuilding the Vertex AI Vector Search index for 3072 dimensions, and it is not yet a stable release — so we defer it as a future upgrade path.

## Running Locally

Follow these steps to run the clinical trials data pipeline locally using Docker and Airflow. **No GCP account required** for local testing.

### Prerequisites

- Docker installed and running
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aditya-sairam/trail-link
cd trial-link
```

### Step 2: Build the Airflow Docker Image

```bash
# Navigate to the airflow directory
cd pipelines

# Build the Docker image
docker build -t airflow-pipeline .
```

### Step 3: Run the Airflow Container

```bash
#docker command to run the mailhog service - If any anomolies are detected, a mail will be sent to the mailhog endpoint.
docker run -d --rm --name mailhog -p 1025:1025 -p 8025:8025 mailhog/mailhog

docker run -d -p 8081:8081 \
  --name airflow-local \
  -v $(pwd)/data:/opt/airflow/repo/data \
  airflow-pipeline
```

**What this does:**
- Runs Airflow on port 8081
- Mounts data directory to persist outputs locally
- Runs without GCP - all data stays in container

### Step 4: Access Airflow UI

Open your browser and navigate to:
```
http://localhost:8081
```

**Login credentials:**
- Username: `admin`
- Password: `admin`

### Step 5: Trigger the Pipeline
1. Navigate to the DAGs page
2. Find `clinical_trials_data_pipeline`
3. Click the **play button** (▶) on the right
4. Click **"Trigger DAG w/ config"**

### Step 6: Monitor Pipeline Execution

1. In the Airflow UI, click on the DAG run
2. View the **Graph** view to see task progress
3. Click individual tasks to view logs
4. Wait for all tasks to complete (green checkmarks)

**Task Groups:**
- `diabetes_pipeline` - Complete processing pipeline for diabetes trials
- `breast_cancer_pipeline` - Complete processing pipeline for breast cancer trials

**Tasks per condition (executed in parallel):**
- `fetch_raw` - Fetch clinical trials from ClinicalTrials.gov for the specific condition
- `schema_raw` - Validate raw data schema against baseline
- `enrich` - Process and enrich trials data with eligibility parsing and classification
- `schema_processed` - Validate processed data schema compliance
- `validate` - Validate data quality (short-circuits if validation fails)
- `quality` - Run quality checks for anomalies (short-circuits if critical issues found)
- `stats` - Generate statistics report (total trials, distributions)
- `anomaly` - Detect and log data anomalies
- `notify_anomaly_email` - sends anomoly details to mailhog endpoint
- `bias` - Analyze demographic and geographic bias
- `save_reports` - Consolidate and save all reports locally
- `check_gcp_config` - Verify GCP credentials and configuration (short-circuits uploads if missing)
- `upload_raw_files_gcs` - Upload raw trial data to Google Cloud Storage
- `upload_reports_gcs` - Upload all reports to Google Cloud Storage
- `upload_firestore` - Upload enriched trials to Firestore database

### Step 7: Retrieve Generated Files

Once the pipeline completes successfully, copy the output files from the container to your local machine:

```bash
# Copy all generated data for a specific condition
docker cp <airflow-container-namme>:/opt/airflow/repo/data/diabetes <to a folder of your choice>
```

**Verify the files:**

```bash
# View directory structure
ls -R diabetes-output/

# Expected output:
# diabetes-output/
# ├── raw/
# │   └── clinical_trials.csv
# ├── enriched/
# │   └── enriched_trials.csv
# ├── schema/
# │   ├── raw_schema.json
# │   └── processed_schema.json
# └── reports/
#     ├── pipeline_summary.json
#     ├── stats.json
#     ├── quality_stats.json
#     ├── anomalies.json
#     ├── bias_report.json
#     ├── schema_raw_report.json
#     └── schema_processed_report.json

# View the pipeline summary
cat diabetes-output/reports/pipeline_summary.json
```

**Note:** The upload tasks (`task_upload_gcs` and `task_upload_firestore`) will show as **skipped** in the Airflow UI when running locally without GCP configuration. This is expected behavior.

### Step 8: Stop and Clean Up

```bash
# Stop the container
docker stop airflow-local

# Remove the container
docker rm airflow-local

# (Optional) Remove the image
docker rmi airflow-pipeline

# (Optional) Clean up downloaded data
rm -rf pipeline-output/
```

---

## Troubleshooting (Local Setup)

### Issue: "Port 8081 already in use"

**Solution:**
```bash
# Use a different port
docker run -p 8082:8081 ...
# Then access: http://localhost:8082
```

### Issue: DAGs not appearing in UI

**Solution:**
- Wait 30-60 seconds for Airflow to scan DAGs
- Check container logs: `docker logs airflow-local`
- Verify DAGs directory is mounted correctly

### Issue: Container crashes or exits immediately

**Solution:**
```bash
# Check logs
docker logs airflow-local

# Common issue: Memory - increase Docker memory limit to 4GB
```

### Issue: Cannot copy files from container

**Solution:**
```bash
# Verify container is running
docker ps

# If stopped, start it
docker start airflow-local

# Then copy files
docker cp airflow-local:/opt/airflow/repo/data ./
```

---

## Expected Output

After successful pipeline execution, you should have:

```
data/
└── diabetes/
    ├── raw/
    │   └── clinical_trials.csv          (Fetched trials)
    ├── enriched/
    │   └── enriched_trials.csv          (Processed trials)
    └── reports/
        ├── pipeline_summary.json        (Overall summary)
        ├── stats.json                   (Data statistics)
        ├── quality_stats.json           (Quality metrics)
        ├── anomalies.json               (Detected anomalies)
        └── bias_report.json             (Bias analysis)
```

---
## Available Conditions

You can trigger the pipeline for different medical conditions:
- `diabetes`
- `breast_cancer`
---

# Option 2:

## Running on Google Cloud Platform

This guide walks through deploying the complete infrastructure and running the clinical trials pipeline on GCP. 
**Pulumi** (Infrastructure as Code tool using Python) automates the creation of all required cloud resources.

---

## Prerequisites

- Google Cloud account with billing enabled
- Terminal/command line access

---

## Step 1: Install Google Cloud CLI

**macOS:**
```bash
brew install google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Windows:**  
Download from https://cloud.google.com/sdk/docs/install

**Verify installation:**
```bash
gcloud version
```

---

## Step 2: Authenticate to Google Cloud

```bash
# Login to your Google account
gcloud auth login

# Set up application default credentials (needed for Pulumi, Docker, and applications)
gcloud auth application-default login

# Set your GCP project ID (replace with your actual project ID)
gcloud config set project YOUR_PROJECT_ID

# Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## Step 3: Enable Required GCP APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
```

---

## Step 4: Install Pulumi

**macOS:**
```bash
brew install pulumi/tap/pulumi
```

**Linux:**
```bash
curl -fsSL https://get.pulumi.com | sh
```

**Windows:**
```bash
choco install pulumi
```

**Verify installation:**
```bash
pulumi version
```

---

## Step 5: Configure Pulumi State Storage

Pulumi stores infrastructure state in a backend. We'll use Google Cloud Storage:

```bash
# Create GCS bucket for Pulumi state files
gcloud storage buckets create gs://pulumi-state-YOUR_PROJECT_ID --location=US

# Configure Pulumi to use this bucket
pulumi login gs://pulumi-state-YOUR_PROJECT_ID

# Set encryption passphrase for secrets (remember this!)
export PULUMI_CONFIG_PASSPHRASE="your-secure-passphrase"
```

### Infrastructure Overview

The infrastructure is managed using Pulumi (Infrastructure as Code) with Python, organized into modular stacks.

### Directory Structure
```
infra/
├── pulumi_stacks/
│   ├── dataPipelineStack.py    # Defines GCP resources for data pipeline (Composer, GCS, Firestore, etc.)
│   └── patientStack.py          # Defines patient-facing app infrastructure (not in scope for data pipeline submission)
├── Pulumi.yaml                  # Project configuration (project name, runtime, description)
├── Pulumi.dev.yaml              # Environment-specific config for 'dev' stack (GCP project ID, region, secrets)
└── requirements.txt             # Python dependencies for Pulumi (pulumi, pulumi-gcp, etc.)
```

### File Descriptions

**`dataPipelineStack.py`**
- Defines all GCP resources for the clinical trials data pipeline
- Creates Cloud run environment (managed Airflow) - *optional, for production deployments (currently disabled)*
- Provisions Cloud Storage buckets for raw data , dvc and reports
- Sets up Firestore database for enriched trial storage
- Configures IAM roles and permissions

**`patientStack.py`**
- Defines infrastructure for patient-facing matching application
- *(Not in scope for data pipeline submission)*

**`Pulumi.yaml`**
- Project metadata (name, runtime, description)
- Specifies Python as the runtime language

**`Pulumi.dev.yaml`**
- Environment-specific configuration for the `dev` stack
- Contains GCP project ID, region, bucket names, Firestore database name
- Keeps secrets and environment variables separate from code

**`requirements.txt`**
- Lists Pulumi dependencies (`pulumi>=3.0.0`, `pulumi-gcp>=7.0.0`)

---

## Step 6: Install Pulumi Dependencies

```bash
# Navigate to Pulumi infrastructure directory
cd infra/pulumi_stacks

# Install Python dependencies
pip install -r requirements.txt
```

---

## Step 7: Deploy Infrastructure with Pulumi

```bash
# Preview what will be created (optional but recommended)
pulumi preview

# Deploy all infrastructure
pulumi up

# Type 'yes' when prompted to confirm deployment
```

**What Pulumi creates:**
- Artifact Registry repository (stores Docker images)
- Firestore database (stores clinical trials and patient data)
- Cloud Storage buckets (stores raw clinical trial data)
- Cloud Run service (Patient CRUD API)
- Service accounts with IAM permissions

**Note:** First deployment takes 5-10 minutes.

---

## Step 8: Verify Infrastructure Deployment

```bash
# View all deployed resources and their outputs
pulumi stack output
```

**Verify in GCP Console:**
- Navigate to https://console.cloud.google.com
- Check **Cloud Run** → Should see `patient-api-dev` service
- Check **Firestore** → Should see database created
- Check **Cloud Storage** → Should see buckets created

---

## Step 9: Export Environment Variables

Export Pulumi outputs to environment variables (needed for Airflow):

```bash
# IMPORTANT: Run these commands in the SAME terminal where you'll run Docker

export BUCKET_NAME=$(pulumi stack output RAW_CLINICAL_TRIALS_STORAGE)
export FIRESTORE_DB=$(pulumi stack output CLINICAL_TRIALS_FIRESTORE)
export PROJECT_ID=$(pulumi config get gcp:project)

# Verify exports
echo "Bucket: $BUCKET_NAME"
echo "Firestore: $FIRESTORE_DB"
echo "Project: $PROJECT_ID"
```

**⚠️ Important:** These variables only exist in your current terminal session. Keep this terminal open!

---

## Step 10: Build and Run Airflow with GCP Configuration

**In the SAME terminal where you exported variables:**

```bash
# Navigate to airflow directory
cd ../../pipelines

# Build Airflow Docker image
docker build -t airflow-pipeline .

# Run Airflow with GCP credentials and configuration
docker run  -p 8081:8081 \
  --name airflow-gcp \
  -e RAW_CLINICAL_TRIALS_STORAGE=$BUCKET_NAME \
  -e GCP_PROJECT_ID=$PROJECT_ID \
  -e CLINICAL_TRIALS_FIRESTORE=$FIRESTORE_DB \
  -e GOOGLE_APPLICATION_CREDENTIALS=/home/airflow/.config/gcloud/application_default_credentials.json \
  -v ~/.config/gcloud/application_default_credentials.json:/home/airflow/.config/gcloud/application_default_credentials.json:ro \
  -v $(pwd)/data:/opt/airflow/repo/data \
  airflow-pipeline

```
---

## Step 11: Trigger the Data Pipeline

**Option A: Via Airflow UI**
1. Open http://localhost:8081
2. Login with username: `admin`, password: `admin`
3. Find the `clinical_trials_data_pipeline` DAG
4. Click the **play button** (▶)
5. Click **"Trigger DAG w/ config"**
6. Enter: `{"condition": "diabetes"}`
7. Click **Trigger**
---

## Step 12: Monitor Pipeline Execution

**In Airflow UI:**
1. Click on the running DAG instance
2. View **Graph** to see task progress
3. Tasks should complete in order:
   - Fetch → Schema Check → Enrich → Validate → Quality → Stats/Anomaly/Bias → Save Reports → **GCP Upload** ✅

**Key difference from local run:** The `task_upload_gcs` and `task_upload_firestore` tasks should show as **succeeded** (green), not skipped!

---

## Step 13: Verify Data in Google Cloud

After pipeline completes successfully:

### **Check Cloud Storage:**

```bash
# List uploaded files
gsutil ls -r gs://$BUCKET_NAME/

# View diabetes trial data
gsutil cat gs://$BUCKET_NAME/diabetes/raw_clinical_trials.json | head -20
```

**Or via GCP Console:**
- Navigate to https://console.cloud.google.com/storage
- Find your bucket (e.g., `triallink-pipeline-data-dev-...`)
- Browse to `diabetes/` folder
- Verify `raw_clinical_trials.json` exists

---

### **Check Firestore:**

```bash
# List Firestore collections
gcloud firestore collections list --database=$FIRESTORE_DB

# View documents in clinical_trials collection
gcloud firestore documents list diabetes_trials --database=$FIRESTORE_DB --limit 5
```

**Or via GCP Console:**
- Navigate to https://console.cloud.google.com/firestore
- Select your database
- Browse collections to see uploaded trial documents

---

## Step 14: Clean Up Resources

**⚠️ Important:** Destroy resources when done to avoid ongoing charges.

```bash
# Stop and remove Airflow container
docker stop airflow-gcp
docker rm airflow-gcp

# Navigate to Pulumi directory
cd infra/pulumi_stacks

# Preview what will be destroyed
pulumi destroy --preview

# Destroy all cloud resources
pulumi destroy

# Confirm by typing 'yes'
```

**This deletes:**
- All Cloud Run services
- Firestore databases
- Cloud Storage buckets (and their contents)
- Artifact Registry repositories
- Service accounts

---

## Alternative: All-in-One Deployment Script

For easier deployment, create `deploy-to-gcp.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Deploying infrastructure with Pulumi..."
cd infra/pulumi_stacks
export PULUMI_CONFIG_PASSPHRASE="your-passphrase"
pulumi up --yes

echo "📝 Exporting configuration..."
export BUCKET_NAME=$(pulumi stack output dev_clinical_trials_bucket)
export FIRESTORE_DB=$(pulumi stack output dev_firestore_db)
export PROJECT_ID=$(pulumi config get gcp:project)

echo "✅ Infrastructure deployed:"
echo "  Bucket: $BUCKET_NAME"
echo "  Firestore: $FIRESTORE_DB"
echo "  Project: $PROJECT_ID"

echo "🐳 Building and running Airflow..."
cd ../../airflow
docker build -t airflow-pipeline -f Dockerfile.airflow .

docker run -d -p 8081:8081 \
  --name airflow-gcp \
  -e RAW_CLINICAL_TRIALS_STORAGE=$BUCKET_NAME \
  -e GCP_PROJECT_ID=$PROJECT_ID \
  -e CLINICAL_TRIALS_FIRESTORE=$FIRESTORE_DB \
  -e GOOGLE_APPLICATION_CREDENTIALS=/home/airflow/.config/gcloud/application_default_credentials.json \
  -v ~/.config/gcloud/application_default_credentials.json:/home/airflow/.config/gcloud/application_default_credentials.json:ro \
  -v $(pwd)/data:/opt/airflow/repo/data \
  airflow-pipeline

echo "✅ Setup complete!"
echo "🌐 Airflow UI: http://localhost:8081"
echo "🔍 Verify env vars: docker exec airflow-gcp env | grep BUCKET"
```

**Usage:**
```bash
chmod +x deploy-to-gcp.sh
./deploy-to-gcp.sh
```

---

## Data Version Control (DVC)

Track and version pipeline outputs using DVC. The GCS bucket for DVC storage is automatically provisioned via Pulumi.

### Setup DVC
```bash
# Initialize DVC
dvc init

# Configure GCS remote (bucket already created by Pulumi)
dvc remote add -d gcs gs://dvc-storage-clinical-trials-<your-project-id>

# Commit DVC config
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Track Pipeline Outputs
```bash
# Track generated data
dvc add data/diabetes/enriched/enriched_trials.csv
dvc add data/diabetes/reports/

dvc add data/breast_cancer/enriched/enriched_trials.csv
dvc add data/breast_cancer/reports/

# Commit DVC metadata
git add data/**/*.dvc .gitignore
git commit -m "Track pipeline outputs with DVC"

# Push data to GCS
dvc push
```

### Pull Data (For Team Members)
```bash
# Clone repo
git clone <repo-url>
cd trail-link

# Pull data from DVC remote
dvc pull
```

### Common Commands
```bash
dvc status          # Check what's changed
dvc pull            # Download tracked data
dvc push            # Upload tracked data
dvc checkout        # Restore data to tracked version
```
---
## Unit Tests

Unit tests are written using **pytest** to validate each component of the data pipeline.

### Test Files

| File | Component Tested |
|---|---|
| `tests/test_ingest.py` | Data fetching, field extraction, enrichment |
| `tests/test_quality.py` | Text cleaning, anomaly detection, quality checks |
| `tests/test_schema.py` | Schema generation, type drift, validation rules |
| `tests/test_bias.py` | Sex/geographic/age bias detection, scoring |
| `tests/test_validate.py` | Row count, NCT format, critical null checks |
| `tests/test_stats.py` | Enrollment stats, sponsor ranking, geography |

### What's Tested

- **Field extraction** — all 32 fields from ClinicalTrials.gov API, missing/malformed keys
- **Text cleaning** — HTML entities, invalid characters, medical abbreviations (T1DM → Type 1 Diabetes Mellitus), whitespace
- **Anomaly detection** — duplicate NCT numbers, high missing columns, enrollment outliers, invalid NCT format
- **Schema validation** — type drift, null thresholds, categorical limits, enforce vs warn modes
- **Bias detection** — sex imbalance ratio, US geographic concentration (>80%), low pediatric representation (<5%), bias scoring (LOW/MEDIUM/HIGH)
- **Validation gates** — empty datasets, missing required columns, disease_type unknown rate
- **Statistics** — enrollment aggregates, top sponsors, country distribution

### Running Tests
```bash
# From repo root
pip install pytest pandas
pytest tests/ -v
```
---

## Troubleshooting

### Environment variables not set in container

**Issue:** Docker container doesn't have the env vars.

**Solution:** Export variables and run Docker in the **same terminal session**, or use the deployment script above.

```bash
# Verify in container
docker exec airflow-gcp env | grep BUCKET
```

### Upload tasks still skipped despite GCP configuration

**Check the logs:**
```bash
docker logs airflow-gcp | grep "GCP configuration"
```

Should show: `✅ GCP configuration complete - uploads to GCS and Firestore enabled`

If it shows incomplete configuration, check that ALL env vars are set correctly.

---
