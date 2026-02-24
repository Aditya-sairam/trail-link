"""
Common ingestion module for clinical trials.
Works for ANY condition via the condition registry.
"""
import os
import time
import logging
import json
import requests
import pandas as pd
from typing import Callable
from datetime import datetime, timezone
from pandas.errors import EmptyDataError

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
) -> dict:
    """
    Incrementally download trials for a condition and persist a delta + merged snapshot.

    Returns metadata with file paths and row counts.
    """
    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

    raw_dir = os.path.dirname(raw_file_path)
    raw_name, raw_ext = os.path.splitext(os.path.basename(raw_file_path))
    raw_delta_file_path = os.path.join(raw_dir, f"{raw_name}_delta{raw_ext or '.csv'}")

    disease_dir = os.path.dirname(raw_dir)
    state_dir = os.path.join(disease_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    state_file_path = os.path.join(state_dir, "fetch_state.json")

    previous_df = _read_csv_or_empty(raw_file_path) if os.path.exists(raw_file_path) else pd.DataFrame()
    previous_df = previous_df.drop_duplicates(subset=["NCT Number"], keep="last") if "NCT Number" in previous_df.columns else previous_df

    fetch_state = _load_fetch_state(state_file_path)
    since_date = fetch_state.get("last_fetch_date")
    if not since_date and not previous_df.empty and "Last Update" in previous_df.columns:
        last_update_series = pd.to_datetime(previous_df["Last Update"], errors="coerce").dropna()
        if not last_update_series.empty:
            since_date = last_update_series.max().date().isoformat()

    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.cond":           condition_query,
        "filter.overallStatus": status,
        "pageSize":             page_size,
        "format":               "json",
    }
    if since_date:
        # ClinicalTrials.gov v2 supports AREA[...]RANGE[...] in query.term.
        params["query.term"] = f"AREA[LastUpdatePostDate]RANGE[{since_date},MAX]"

    all_trials = []
    next_page_token = None
    page_num = 0

    log.info(f"Fetching '{condition_query}' trials incrementally (since={since_date or 'full'})...")

    while True:
        if next_page_token:
            params["pageToken"] = next_page_token
        elif "pageToken" in params:
            params.pop("pageToken", None)

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

    fetched_df = pd.DataFrame(all_trials)
    delta_df = _compute_delta_trials(fetched_df=fetched_df, previous_df=previous_df)

    if delta_df.empty:
        _write_csv(delta_df, raw_delta_file_path)
        log.info(f"✓ No new/updated trials found → {raw_delta_file_path}")
    else:
        _write_csv(delta_df, raw_delta_file_path)
        log.info(f"✓ Delta trials saved ({len(delta_df):,}) → {raw_delta_file_path}")

    merged_df = _merge_latest_trials(previous_df=previous_df, delta_df=delta_df)
    _write_csv(merged_df, raw_file_path)
    log.info(f"✓ Raw snapshot updated ({len(merged_df):,}) → {raw_file_path}")

    _save_fetch_state(
        state_file_path=state_file_path,
        last_fetch_date=datetime.now(timezone.utc).date().isoformat(),
        last_fetch_at=datetime.now(timezone.utc).isoformat(),
        condition_query=condition_query,
        fetched_rows=int(len(fetched_df)),
        delta_rows=int(len(delta_df)),
        snapshot_rows=int(len(merged_df)),
        since_date=since_date,
    )

    return {
        "raw_snapshot_path": raw_file_path,
        "raw_delta_path": raw_delta_file_path,
        "state_file_path": state_file_path,
        "fetched_rows": int(len(fetched_df)),
        "delta_rows": int(len(delta_df)),
        "snapshot_rows": int(len(merged_df)),
        "since_date": since_date,
    }


def enrich_trials_csv(
    raw_file_path: str,
    enriched_file_path: str,
    disease: str,
    classifier: Callable[[str], str],
    full_enriched_file_path: str | None = None,
) -> pd.DataFrame:
    """Add disease and disease_type columns to delta file and optionally merge full snapshot."""
    os.makedirs(os.path.dirname(enriched_file_path), exist_ok=True)

    df = pd.read_csv(raw_file_path)
    before = len(df)
    if "NCT Number" in df.columns:
        df = df.drop_duplicates(subset=["NCT Number"], keep="last")
    log.info(f"Deduplication: {before:,} → {len(df):,} rows")

    df["disease"]      = disease
    df["disease_type"] = df["Conditions"].apply(classifier)

    df.to_csv(enriched_file_path, index=False)
    log.info(f"✓ Enriched → {enriched_file_path}")
    for dtype, count in df["disease_type"].value_counts().items():
        log.info(f"  {dtype:40} {count:,}")

    if full_enriched_file_path:
        os.makedirs(os.path.dirname(full_enriched_file_path), exist_ok=True)
        previous_enriched = (
            _read_csv_or_empty(full_enriched_file_path)
            if os.path.exists(full_enriched_file_path)
            else pd.DataFrame(columns=df.columns)
        )
        merged_enriched = _merge_latest_trials(previous_df=previous_enriched, delta_df=df)
        _write_csv(merged_enriched, full_enriched_file_path)
        log.info(f"✓ Enriched snapshot updated ({len(merged_enriched):,}) → {full_enriched_file_path}")

    return df


def _load_fetch_state(state_file_path: str) -> dict:
    if not os.path.exists(state_file_path):
        return {}
    try:
        with open(state_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Failed to read fetch state at {state_file_path}: {e}")
        return {}


def _save_fetch_state(state_file_path: str, **state: object) -> None:
    with open(state_file_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _compute_delta_trials(fetched_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    if fetched_df.empty:
        return fetched_df

    if previous_df.empty or "NCT Number" not in previous_df.columns:
        return fetched_df.drop_duplicates(subset=["NCT Number"], keep="last") if "NCT Number" in fetched_df.columns else fetched_df

    prev_latest = _prepare_latest(previous_df)
    fetched_latest = _prepare_latest(fetched_df)

    if "NCT Number" not in fetched_latest.columns:
        return fetched_latest

    compare_cols = [c for c in ["NCT Number", "Last Update"] if c in prev_latest.columns and c in fetched_latest.columns]
    if len(compare_cols) == 2:
        merged = fetched_latest.merge(
            prev_latest[compare_cols].rename(columns={"Last Update": "_prev_last_update"}),
            on="NCT Number",
            how="left",
        )
        delta_mask = merged["_prev_last_update"].isna() | (
            merged["Last Update"].fillna("").astype(str) != merged["_prev_last_update"].fillna("").astype(str)
        )
        return merged.loc[delta_mask, fetched_latest.columns]

    prev_ids = set(prev_latest["NCT Number"].astype(str))
    return fetched_latest[fetched_latest["NCT Number"].astype(str).map(lambda nct: nct not in prev_ids)]


def _merge_latest_trials(previous_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    if previous_df.empty and delta_df.empty:
        return pd.DataFrame()
    if previous_df.empty:
        return _prepare_latest(delta_df)
    if delta_df.empty:
        return _prepare_latest(previous_df)

    combined = pd.concat([previous_df, delta_df], ignore_index=True, sort=False)
    return _prepare_latest(combined)


def _prepare_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "Last Update" in out.columns:
        out["_last_update_sort"] = pd.to_datetime(out["Last Update"], errors="coerce")
        out = out.sort_values(by=["_last_update_sort"], ascending=True, na_position="first")
    if "NCT Number" in out.columns:
        out = out.drop_duplicates(subset=["NCT Number"], keep="last")
    if "_last_update_sort" in out.columns:
        out = out.drop(columns=["_last_update_sort"])
    return out


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _read_csv_or_empty(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        log.warning(f"CSV file is empty at {path}; treating as empty dataset.")
        return pd.DataFrame()
