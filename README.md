# Trial-Link

MLOps platform for clinical trial matching.

This repository contains: - Infrastructure as Code (Pulumi + GCP) - ML
SDK and model modules (future scope) - Production-style data pipelines
(Airflow + DVC) - Modular ETL for clinical trial datasets

Current Implemented Pipeline: - Breast Cancer Clinical Trials Data
Pipeline (Airflow + DVC)

------------------------------------------------------------------------

# Setup

## 1. Clone the repository

## 2. Create and activate virtual environment

``` bash
python -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Project Structure

    trail-link/
    │
    ├── infra/                  # Infrastructure as Code (Pulumi stacks)
    ├── sdk/                    # Python SDK (future ML ops layer)
    ├── models/                 # Model definitions & training scripts (future)
    ├── pipelines/              # Legacy / experimental pipeline folder
    ├── src/                    # Modular production pipeline code
    │   └── pipelines/
    │       ├── breast_cancer/
    │       │   ├── ingest.py
    │       │   ├── quality.py
    │       │   ├── bias.py
    │       │   └── run_local.py
    │       └── diabetes/      
    │
    ├── dags/                   # Airflow DAG definitions
    │   └── breast_cancer_pipeline_dag.py
    │
    ├── data/                   # DVC-tracked artifacts
    │   └── breast_cancer/
    │       ├── raw/
    │       ├── processed/
    │       └── reports/
    │
    ├── tests/                  # Unit tests (pytest)
    ├── docker-compose.yml
    ├── Dockerfile.airflow
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Breast Cancer Data Pipeline (Airflow)

Airflow DAG: `breast_cancer_data_pipeline`

Pipeline Tasks:

1.  fetch_raw
    -   Downloads clinical trials (CSV) from ClinicalTrials.gov API.
2.  enrich
    -   Deduplicates by NCTId\
    -   Adds `disease` column\
    -   Extracts `cancer_type` from Conditions
3.  quality_checks
    -   Generates dataset statistics\
    -   Validates schema\
    -   Detects anomalies
4.  check_anomalies
    -   Conditional branch if anomalies exist
5.  alert_on_anomalies
    -   Sends alert if anomalies detected
6.  bias_slicing
    -   Generates subgroup representation report
7.  dvc_version_data
    -   Versions raw, processed, and report artifacts using DVC

------------------------------------------------------------------------

# Run with Airflow (Docker)

## Build and start services

``` bash
docker compose up -d --build
```

## Open Airflow UI

URL:

    http://localhost:8080

Credentials:

    username: admin
    password: admin

Trigger DAG:

    breast_cancer_data_pipeline

Use Graph View and Gantt View to inspect pipeline flow and bottlenecks.

------------------------------------------------------------------------

# Data Versioning with DVC

Tracked directories: - data/breast_cancer/raw/ -
data/breast_cancer/processed/ - data/breast_cancer/reports/

Common commands:

``` bash
dvc status
dvc push
dvc pull
```

Reproducibility test:

``` bash
rm -rf data/breast_cancer/raw data/breast_cancer/processed data/breast_cancer/reports
dvc pull
```

------------------------------------------------------------------------

# Data Quality & Bias Detection

Generated Reports:

-   stats.json
-   anomalies.json
-   bias_report.json

Bias slicing evaluates representation and missingness across subgroups.

------------------------------------------------------------------------

# How to Setup and Run Pulumi (Infrastructure)

## Step 1: Install Pulumi

MacOS:

``` bash
brew install pulumi/tap/pulumi
```

Linux:

``` bash
curl -fsSL https://get.pulumi.com | sh
```

## Step 2: Login to Pulumi (GCP Bucket Backend)

``` bash
gcloud storage buckets create gs://pulumi-state-YOUR_PROJECT_ID --location=US
pulumi login gs://pulumi-state-YOUR_PROJECT_ID
```

## Step 3: Navigate to Pulumi stacks

``` bash
cd infra/pulumi_stacks
```

## Step 4: Preview and Deploy

``` bash
pulumi preview
pulumi up
```

------------------------------------------------------------------------

# Reproducibility

To fully reproduce the pipeline:

1.  Clone repository\
2.  Install Docker\
3.  Run:

``` bash
docker compose up -d --build
```

4.  Trigger DAG from Airflow UI

All dataset artifacts are versioned with DVC.

------------------------------------------------------------------------


