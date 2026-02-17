import os
import time
import logging
from typing import Optional
from src.common.logger import build_logger

import pandas as pd
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def get_logger(name: str = "breast_cancer") -> logging.Logger:
    return build_logger(name=name, pipeline_name="breast_cancer")


def sanitize_filename(text: str) -> str:
    """
    Matches your original sanitize behavior.
    """
    text = str(text).lower()
    text = text.replace(" ", "_")
    text = "".join(c for c in text if c.isalnum() or c == "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def extract_cancer_type(conditions: str) -> str:
    """
    Extract most specific breast cancer subtype from Conditions column.
    Matches your original logic.
    """
    if pd.isna(conditions):
        return "breast_cancer"

    condition_list = str(conditions).split("|")

    for condition in condition_list:
        lower = condition.lower()
        if "breast" in lower and "cancer" in lower:
            return sanitize_filename(condition)

    return "breast_cancer"


def download_raw_trials_csv(
    raw_file_path: str,
    status: str = "RECRUITING",
    page_size: int = 1000,
    condition_query: str = "breast cancer",
    sleep_seconds: float = 0.5,
    timeout_seconds: int = 120,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Downloads ALL columns in CSV format into a single raw CSV file.
    Uses JSON calls to get nextPageToken, matching your approach.
    """
    logger = logger or get_logger()

    os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)

    params = {
        "query.cond": condition_query,
        "filter.overallStatus": status,
        "pageSize": page_size,
        "format": "csv",
    }

    page_token = None
    first_page = True
    page_count = 0

    if os.path.exists(raw_file_path):
        os.remove(raw_file_path)

    with open(raw_file_path, "wb") as out:
        while True:
            request_params = dict(params)
            if page_token:
                request_params["pageToken"] = page_token

            response = requests.get(BASE_URL, params=request_params, timeout=timeout_seconds)
            response.raise_for_status()
            content = response.content

            if first_page:
                out.write(content)
                first_page = False
            else:
                lines = content.split(b"\n", 1)
                if len(lines) > 1:
                    out.write(lines[1])

            page_count += 1
            logger.info("Downloaded page %s", page_count)

            token_check_params = {
                "query.cond": condition_query,
                "filter.overallStatus": status,
                "pageSize": page_size,
            }
            if page_token:
                token_check_params["pageToken"] = page_token

            token_response = requests.get(BASE_URL, params=token_check_params, timeout=timeout_seconds)
            token_response.raise_for_status()
            token_data = token_response.json()

            page_token = token_data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(sleep_seconds)

    logger.info("Raw CSV download complete: %s", raw_file_path)
    return raw_file_path


def enrich_trials_csv(
    raw_file_path: str,
    enriched_file_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Loads raw CSV, deduplicates by NCTId, adds disease + cancer_type, writes enriched CSV.
    Matches your original behavior.
    """
    logger = logger or get_logger()

    os.makedirs(os.path.dirname(enriched_file_path), exist_ok=True)

    logger.info("Loading raw CSV into pandas: %s", raw_file_path)
    df = pd.read_csv(raw_file_path)

    logger.info("Rows before deduplication: %s", len(df))
    if "NCTId" in df.columns:
        df = df.drop_duplicates(subset="NCTId")
    logger.info("Rows after deduplication: %s", len(df))

    df["disease"] = "breast_cancer"

    if "Conditions" in df.columns:
        df["cancer_type"] = df["Conditions"].apply(extract_cancer_type)
    else:
        df["cancer_type"] = "breast_cancer"

    df.to_csv(enriched_file_path, index=False)
    logger.info("Enriched CSV saved: %s", enriched_file_path)
    logger.info("Total unique trials saved: %s", len(df))

    return enriched_file_path


def run_full_ingestion(
    raw_file_path: str,
    enriched_file_path: str,
    status: str = "RECRUITING",
    page_size: int = 1000,
) -> str:
    """
    One function to run the full pipeline locally.
    """
    logger = get_logger()
    logger.info("=" * 80)
    logger.info("FULL BREAST CANCER INGESTION STARTED")
    logger.info("=" * 80)

    download_raw_trials_csv(
        raw_file_path=raw_file_path,
        status=status,
        page_size=page_size,
        logger=logger,
    )

    output = enrich_trials_csv(
        raw_file_path=raw_file_path,
        enriched_file_path=enriched_file_path,
        logger=logger,
    )

    logger.info("=" * 80)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 80)
    return output
