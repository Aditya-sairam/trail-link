"""
Clinical Trials Data Pipeline DAG
Fetches diabetes clinical trials from ClinicalTrials.gov API v2,
saves as batched CSVs, and uploads to GCS.
"""

import os
import csv
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# ============================================================
# Configuration
# ============================================================
BUCKET_NAME = os.getenv("GCS_BUCKET", "triallink-pipeline-data-datapipeline-infra")
CONDITION = "diabetes"
STATUSES = "RECRUITING|ACTIVE_NOT_RECRUITING|ENROLLING_BY_INVITATION"
PAGE_SIZE = 1000
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
LOCAL_DIR = "/tmp/pipeline"
RUN_DATE = datetime.utcnow().strftime("%Y-%m-%d")

# CSV column order (matches your reference)
CSV_COLUMNS = [
    "NCT_ID", "Title", "Brief_Title", "Status", "Study_Type", "Phase",
    "Enrollment", "Start_Date", "Completion_Date", "Primary_Completion_Date",
    "Last_Update", "Brief_Summary", "Detailed_Description", "Conditions",
    "Keywords", "Allocation", "Intervention_Model", "Primary_Purpose",
    "Masking", "Interventions", "Intervention_Types", "Eligibility_Criteria",
    "Min_Age", "Max_Age", "Sex", "Accepts_Healthy_Volunteers",
    "Location_Countries", "Location_Cities", "Number_of_Locations",
    "Sponsor", "Sponsor_Class", "Collaborators", "Study_URL",
]

log = logging.getLogger(__name__)


# ============================================================
# Helper: Extract fields from a single study JSON
# ============================================================
def extract_study(study):
    """Flatten a single study JSON into a dict matching CSV_COLUMNS."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    desc = proto.get("descriptionModule", {})
    design = proto.get("designModule", {})
    elig = proto.get("eligibilityModule", {})
    sponsor = proto.get("sponsorCollaboratorsModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    arms = proto.get("armsInterventionsModule", {})
    conditions = proto.get("conditionsModule", {})

    # Extract interventions
    interventions = arms.get("interventions", [])
    intervention_names = "; ".join(i.get("name", "") for i in interventions)
    intervention_types = "; ".join(i.get("type", "") for i in interventions)

    # Extract locations
    locations = contacts.get("locations", [])
    countries = list(set(loc.get("country", "") for loc in locations if loc.get("country")))
    cities = list(set(loc.get("city", "") for loc in locations if loc.get("city")))

    # Extract sponsor info
    lead_sponsor = sponsor.get("leadSponsor", {})
    collabs = sponsor.get("collaborators", [])
    collab_names = "; ".join(c.get("name", "") for c in collabs)

    # Extract design info
    design_info = design.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})
    phases = design.get("phases", [])

    # Extract dates
    start = status.get("startDateStruct", {})
    completion = status.get("completionDateStruct", {})
    primary_completion = status.get("primaryCompletionDateStruct", {})

    # Extract keywords
    kw = conditions.get("keywords", [])

    return {
        "NCT_ID": ident.get("nctId", ""),
        "Title": ident.get("officialTitle", ""),
        "Brief_Title": ident.get("briefTitle", ""),
        "Status": status.get("overallStatus", ""),
        "Study_Type": design.get("studyType", ""),
        "Phase": "; ".join(phases) if phases else "",
        "Enrollment": design.get("enrollmentInfo", {}).get("count", ""),
        "Start_Date": start.get("date", ""),
        "Completion_Date": completion.get("date", ""),
        "Primary_Completion_Date": primary_completion.get("date", ""),
        "Last_Update": status.get("lastUpdateSubmitDate", ""),
        "Brief_Summary": desc.get("briefSummary", ""),
        "Detailed_Description": desc.get("detailedDescription", ""),
        "Conditions": "; ".join(conditions.get("conditions", [])),
        "Keywords": "; ".join(kw),
        "Allocation": design_info.get("allocation", ""),
        "Intervention_Model": design_info.get("interventionModel", ""),
        "Primary_Purpose": design_info.get("primaryPurpose", ""),
        "Masking": masking_info.get("masking", ""),
        "Interventions": intervention_names,
        "Intervention_Types": intervention_types,
        "Eligibility_Criteria": elig.get("eligibilityCriteria", ""),
        "Min_Age": elig.get("minimumAge", ""),
        "Max_Age": elig.get("maximumAge", ""),
        "Sex": elig.get("sex", ""),
        "Accepts_Healthy_Volunteers": elig.get("healthyVolunteers", ""),
        "Location_Countries": "; ".join(countries),
        "Location_Cities": "; ".join(cities),
        "Number_of_Locations": len(locations),
        "Sponsor": lead_sponsor.get("name", ""),
        "Sponsor_Class": lead_sponsor.get("class", ""),
        "Collaborators": collab_names,
        "Study_URL": f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}",
    }


# ============================================================
# Task 1: Fetch trials from API
# ============================================================
def fetch_trials(**context):
    """Fetch all trials from ClinicalTrials.gov API v2 with pagination."""
    os.makedirs(LOCAL_DIR, exist_ok=True)

    all_studies = []
    page_token = None
    page_num = 0

    while True:
        page_num += 1
        log.info(f"Fetching page {page_num}...")

        params = {
            "query.cond": CONDITION,
            "filter.overallStatus": STATUSES,
            "pageSize": PAGE_SIZE,
            "format": "json",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies", [])
        all_studies.extend(studies)
        log.info(f"Page {page_num}: fetched {len(studies)} studies (total: {len(all_studies)})")

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        ## Rate limiting: be polite to the API
        time.sleep(0.5)

    # Save raw JSON
    raw_path = os.path.join(LOCAL_DIR, "raw_studies.json")
    with open(raw_path, "w") as f:
        json.dump(all_studies, f)

    log.info(f"Total studies fetched: {len(all_studies)}")
    context["ti"].xcom_push(key="total_studies", value=len(all_studies))
    context["ti"].xcom_push(key="raw_path", value=raw_path)


# ============================================================
# Task 2: Save to batched CSVs
# ============================================================
def save_to_csv(**context):
    """Flatten JSON and save as batched CSV files."""
    raw_path = context["ti"].xcom_pull(key="raw_path")

    with open(raw_path, "r") as f:
        all_studies = json.load(f)

    csv_paths = []
    batch_num = 0

    for i in range(0, len(all_studies), PAGE_SIZE):
        batch_num += 1
        batch = all_studies[i : i + PAGE_SIZE]
        batch_str = str(batch_num).zfill(3)

        # Create batch directory
        batch_dir = os.path.join(LOCAL_DIR, f"batch_{batch_str}")
        os.makedirs(batch_dir, exist_ok=True)

        # CSV file path
        csv_path = os.path.join(batch_dir, f"{RUN_DATE}_batch_{batch_str}.csv")

        rows = [extract_study(s) for s in batch]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        csv_paths.append(csv_path)
        log.info(f"Batch {batch_str}: saved {len(rows)} rows to {csv_path}")

    context["ti"].xcom_push(key="csv_paths", value=csv_paths)
    context["ti"].xcom_push(key="total_batches", value=batch_num)
    log.info(f"Total batches created: {batch_num}")


# ============================================================
# Task 3: Upload to GCS
# ============================================================
def upload_to_gcs(**context):
    """Upload batched CSVs to GCS bucket."""
    from google.cloud import storage

    csv_paths = context["ti"].xcom_pull(key="csv_paths")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for csv_path in csv_paths:
        # Extract batch folder name from path
        # e.g., /tmp/pipeline/batch_001/2026-02-16_batch_001.csv
        parts = csv_path.split("/")
        batch_folder = parts[-2]  # batch_001
        filename = parts[-1]      # 2026-02-16_batch_001.csv

        gcs_path = f"raw/{RUN_DATE}/{batch_folder}/{filename}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(csv_path)
        log.info(f"Uploaded: gs://{BUCKET_NAME}/{gcs_path}")

    log.info(f"All {len(csv_paths)} batches uploaded to GCS")


# ============================================================
# Task 4: Cleanup local files
# ============================================================
def cleanup(**context):
    """Remove local temp files."""
    import shutil

    if os.path.exists(LOCAL_DIR):
        shutil.rmtree(LOCAL_DIR)
        log.info(f"Cleaned up {LOCAL_DIR}")


# ============================================================
# DAG Definition
# ============================================================
default_args = {
    "owner": "triallink",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="clinical_trials_pipeline",
    default_args=default_args,
    description="Fetch diabetes clinical trials and upload to GCS",
    schedule_interval=None,  # Triggered externally by VM startup
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["clinical-trials", "diabetes", "data-pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_trials",
        python_callable=fetch_trials,
    )

    t2 = PythonOperator(
        task_id="save_to_csv",
        python_callable=save_to_csv,
    )

    t3 = PythonOperator(
        task_id="upload_to_gcs",
        python_callable=upload_to_gcs,
    )

    t4 = PythonOperator(
        task_id="cleanup",
        python_callable=cleanup,
    )

    t1 >> t2 >> t3 >> t4