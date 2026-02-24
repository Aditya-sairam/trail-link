"""
Common ingestion module for clinical trials.
Works for ANY condition via the condition registry.
"""
import os
import time
import logging
import requests
import pandas as pd
from typing import Callable

log = logging.getLogger(__name__)


def extract_study(study: dict) -> dict:
    """Flatten a single study JSON into a flat dict with 32 fields."""
    proto    = study.get("protocolSection", {})
    ident    = proto.get("identificationModule", {})
    status   = proto.get("statusModule", {})
    desc     = proto.get("descriptionModule", {})
    design   = proto.get("designModule", {})
    elig     = proto.get("eligibilityModule", {})
    sponsor  = proto.get("sponsorCollaboratorsModule", {})
    contacts = proto.get("contactsLocationsModule", {})
    arms     = proto.get("armsInterventionsModule", {})
    conds    = proto.get("conditionsModule", {})

    interventions      = arms.get("interventions", [])
    intervention_names = "; ".join(i.get("name", "") for i in interventions if isinstance(i, dict))
    intervention_types = "; ".join(i.get("type", "") for i in interventions if isinstance(i, dict))

    locations = contacts.get("locations", [])
    countries = list(set(
        loc.get("country", "") for loc in locations
        if isinstance(loc, dict) and loc.get("country")
    ))
    cities = list(set(
        loc.get("city", "") for loc in locations
        if isinstance(loc, dict) and loc.get("city")
    ))

    lead_sponsor = sponsor.get("leadSponsor", {})
    collabs      = sponsor.get("collaborators", [])
    collab_names = "; ".join(c.get("name", "") for c in collabs if isinstance(c, dict))
    design_info  = design.get("designInfo", {})
    masking_info = design_info.get("maskingInfo", {})
    phases       = design.get("phases", [])
    start             = status.get("startDateStruct", {})
    completion        = status.get("completionDateStruct", {})
    primary_completion = status.get("primaryCompletionDateStruct", {})

    return {
        "NCT Number":                 ident.get("nctId", ""),
        "Title":                      ident.get("officialTitle", ""),
        "Study Title":                ident.get("briefTitle", ""),
        "Recruitment Status":         status.get("overallStatus", ""),
        "Study Type":                 design.get("studyType", ""),
        "Phase":                      "; ".join(phases) if phases else "",
        "Enrollment":                 design.get("enrollmentInfo", {}).get("count", "") if isinstance(design.get("enrollmentInfo"), dict) else "",
        "Start Date":                 start.get("date", ""),
        "Completion Date":            completion.get("date", ""),
        "Primary Completion Date":    primary_completion.get("date", ""),
        "Last Update":                status.get("lastUpdateSubmitDate", ""),
        "Brief Summary":              desc.get("briefSummary", ""),
        "Detailed Description":       desc.get("detailedDescription", ""),
        "Conditions":                 "; ".join(c for c in conds.get("conditions", []) if isinstance(c, str)),
        "Keywords":                   "; ".join(conds.get("keywords", [])),
        "Allocation":                 design_info.get("allocation", ""),
        "Intervention Model":         design_info.get("interventionModel", ""),
        "Primary Purpose":            design_info.get("primaryPurpose", ""),
        "Masking":                    masking_info.get("masking", ""),
        "Interventions":              intervention_names,
        "Intervention Types":         intervention_types,
        "Eligibility Criteria":       elig.get("eligibilityCriteria", ""),
        "Min Age":                    elig.get("minimumAge", ""),
        "Max Age":                    elig.get("maximumAge", ""),
        "Sex":                        elig.get("sex", ""),
        "Accepts Healthy Volunteers": elig.get("healthyVolunteers", ""),
        "Location Countries":         "; ".join(countries),
        "Location Cities":            "; ".join(cities),
        "Number of Locations":        len(locations),
        "Sponsor":                    lead_sponsor.get("name", "") if isinstance(lead_sponsor, dict) else "",
        "Sponsor Class":              lead_sponsor.get("class", "") if isinstance(lead_sponsor, dict) else "",
        "Collaborators":              collab_names,
        "Study URL":                  f"https://clinicaltrials.gov/study/{ident.get('nctId', '')}",
    }


def download_raw_trials_csv(
    raw_file_path: str,
    condition_query: str,
    status: str = "RECRUITING",
    page_size: int = 1000,
    sleep_seconds: float = 0.3,
) -> int:
    """Download all trials for a condition. Single broad query with pagination."""
    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

    url    = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond":           condition_query,
        "filter.overallStatus": status,
        "pageSize":             page_size,
        "format":               "json",
    }

    all_trials      = []
    next_page_token = None
    page_num        = 0

    log.info(f"Fetching '{condition_query}' trials...")

    while True:
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed on page {page_num}: {e}")
            break

        studies  = data.get("studies", [])
        page_num += 1
        log.info(f"  Page {page_num}: {len(studies)} trials (total: {len(all_trials) + len(studies)})")

        for study in studies:
            try:
                all_trials.append(extract_study(study))
            except Exception as e:
                log.warning(f"  Skipping study: {e}")
                continue

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            log.info("  No more pages.")
            break

        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_trials)
    df.to_csv(raw_file_path, index=False)
    log.info(f"✓ Downloaded {len(df):,} trials → {raw_file_path}")
    return len(df)


def enrich_trials_csv(
    raw_file_path: str,
    enriched_file_path: str,
    disease: str,
    classifier: Callable[[str], str],
) -> pd.DataFrame:
    """Add disease and disease_type columns. Deduplicates by NCT Number."""
    os.makedirs(os.path.dirname(enriched_file_path), exist_ok=True)

    df     = pd.read_csv(raw_file_path)
    before = len(df)
    df     = df.drop_duplicates(subset=["NCT Number"], keep="first")
    log.info(f"Deduplication: {before:,} → {len(df):,} rows")

    df["disease"]      = disease
    df["disease_type"] = df["Conditions"].apply(classifier)

    df.to_csv(enriched_file_path, index=False)
    log.info(f"✓ Enriched → {enriched_file_path}")
    for dtype, count in df["disease_type"].value_counts().items():
        log.info(f"  {dtype:40} {count:,}")

    return df