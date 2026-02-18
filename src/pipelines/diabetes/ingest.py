"""
Diabetes trials ingestion
Mirrors: src/pipelines/breast_cancer/ingest.py
"""
import pandas as pd
import requests
import os
import logging

logger = logging.getLogger(__name__)

BASE          = "/opt/airflow/repo"
RAW_PATH      = f"{BASE}/data/diabetes/raw/diabetes_trials_raw.csv"
ENRICHED_PATH = f"{BASE}/data/diabetes/processed/diabetes_trials_enriched.csv"

DIABETES_CONDITIONS = [
    "diabetes",
    "diabetes mellitus",
    "type 1 diabetes",
    "type 2 diabetes",
    "gestational diabetes",
    "prediabetes",
    "diabetes insipidus",
    "neonatal diabetes",
    "monogenic diabetes",
    "latent autoimmune diabetes",
    "maturity onset diabetes",
]


def safe_get_locations(contacts_module):
    """
    Safely extract location names from contactsLocationsModule.
    Handles all cases: list of dicts, list of strings, empty, None.
    """
    try:
        locations = contacts_module.get("locations", [])
        if not locations or not isinstance(locations, list):
            return ""
        names = []
        for loc in locations[:3]:
            if isinstance(loc, dict):
                facility = loc.get("facility", {})
                if isinstance(facility, dict):
                    name = facility.get("name", "")
                elif isinstance(facility, str):
                    name = facility
                else:
                    name = ""
                if name:
                    names.append(name)
            elif isinstance(loc, str):
                if loc:
                    names.append(loc)
        return ", ".join(names)
    except Exception:
        return ""


def download_raw_trials_csv(
    raw_file_path: str = RAW_PATH,
    status: str = "RECRUITING",
    page_size: int = 1000,
    condition_query: str = "diabetes",
):
    """
    Download ALL diabetes trials from ClinicalTrials.gov API.
    Searches every diabetes condition name and deduplicates by NCT Number.
    """
    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

    url = "https://clinicaltrials.gov/api/v2/studies"
    all_trials = []
    seen_nct_ids = set()

    for condition in DIABETES_CONDITIONS:
        logger.info(f"Fetching: '{condition}'...")

        params = {
            "query.cond": condition,
            "filter.overallStatus": status,
            "pageSize": page_size,
            "format": "json",
        }

        next_page_token = None
        condition_count = 0

        while True:
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"  Request failed for '{condition}': {e}")
                break

            studies = data.get("studies", [])

            for study in studies:
                try:
                    proto     = study.get("protocolSection", {})
                    id_module = proto.get("identificationModule", {})
                    nct_id    = id_module.get("nctId", "")

                    if nct_id in seen_nct_ids:
                        continue
                    seen_nct_ids.add(nct_id)

                    status_module      = proto.get("statusModule", {})
                    desc_module        = proto.get("descriptionModule", {})
                    conditions_module  = proto.get("conditionsModule", {})
                    design_module      = proto.get("designModule", {})
                    contacts_module    = proto.get("contactsLocationsModule", {})
                    eligibility_module = proto.get("eligibilityModule", {})
                    sponsor_module     = proto.get("sponsorCollaboratorsModule", {})
                    lead_sponsor       = sponsor_module.get("leadSponsor", {})

                    # Safe interventions extraction
                    interventions = []
                    arms_module = proto.get("armsInterventionsModule", {})
                    for i in arms_module.get("interventions", []):
                        if isinstance(i, dict):
                            interventions.append(i.get("name", ""))

                    trial = {
                        "NCT Number":         nct_id,
                        "Study Title":        id_module.get("briefTitle", ""),
                        "Recruitment Status": status_module.get("overallStatus", ""),
                        "Brief Summary":      desc_module.get("briefSummary", ""),
                        "Conditions":         ", ".join(
                            c for c in conditions_module.get("conditions", [])
                            if isinstance(c, str)
                        ),
                        "Interventions":      ", ".join(interventions),
                        "Sponsor":            lead_sponsor.get("name", "") if isinstance(lead_sponsor, dict) else "",
                        "Enrollment":         design_module.get("enrollmentInfo", {}).get("count", "") if isinstance(design_module.get("enrollmentInfo"), dict) else "",
                        "Age":                ", ".join(
                            a for a in eligibility_module.get("stdAges", [])
                            if isinstance(a, str)
                        ),
                        "Sex":                eligibility_module.get("sex", ""),
                        "Locations":          safe_get_locations(contacts_module),
                    }
                    all_trials.append(trial)
                    condition_count += 1

                except Exception as e:
                    logger.warning(f"  Skipping study due to error: {e}")
                    continue

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        logger.info(f"  ✓ {condition_count} new unique trials for '{condition}'")

    df = pd.DataFrame(all_trials)
    df.to_csv(raw_file_path, index=False)

    logger.info(f"✓ Total unique trials downloaded: {len(df):,}")
    logger.info(f"  Saved → {raw_file_path}")
    return len(df)


def classify_disease_type(conditions: str) -> str:
    """Classify diabetes type from the Conditions field."""
    if pd.isna(conditions) or conditions is None:
        return "Unknown"
    c = str(conditions).lower()
    if "type 1" in c or "t1d" in c or "t1dm" in c or "juvenile" in c or "iddm" in c or "lada" in c:
        return "Type 1 Diabetes"
    elif "type 2" in c or "t2d" in c or "t2dm" in c or "niddm" in c or "non-insulin dependent" in c:
        return "Type 2 Diabetes"
    elif "gestational" in c or "gdm" in c:
        return "Gestational Diabetes"
    elif "prediabetes" in c or "pre-diabetes" in c or "impaired glucose" in c:
        return "Pre-Diabetes"
    elif "insipidus" in c:
        return "Diabetes Insipidus"
    elif "neonatal" in c or "monogenic" in c or "mody" in c:
        return "Rare Diabetes"
    else:
        return "Diabetes (General)"


def enrich_trials_csv(
    raw_file_path: str = RAW_PATH,
    enriched_file_path: str = ENRICHED_PATH,
):
    """Enrich raw trials with disease_type and data_source columns."""
    os.makedirs(os.path.dirname(enriched_file_path), exist_ok=True)

    df = pd.read_csv(raw_file_path)
    df = df.drop_duplicates(subset=["NCT Number"], keep="first")
    df["disease_type"] = df["Conditions"].apply(classify_disease_type)
    df["data_source"]  = "ClinicalTrials.gov"
    df.to_csv(enriched_file_path, index=False)

    logger.info(f"✓ Enriched {len(df):,} trials → {enriched_file_path}")
    for dtype, count in df["disease_type"].value_counts().items():
        logger.info(f"  {dtype:35} {count:,}")

    return df