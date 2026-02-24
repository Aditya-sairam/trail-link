import os
import sys
import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ingest import extract_study, download_raw_trials_csv, enrich_trials_csv
from conditions.registry import REGISTRY

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_study(**overrides):
    """Return a minimal but complete study JSON matching ClinicalTrials API v2 shape."""
    base = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "officialTitle": "Official Title",
                "briefTitle": "Brief Title",
            },
            "statusModule": {
                "overallStatus": "RECRUITING",
                "lastUpdateSubmitDate": "2024-01-01",
                "startDateStruct": {"date": "2023-01-01"},
                "completionDateStruct": {"date": "2025-01-01"},
                "primaryCompletionDateStruct": {"date": "2024-06-01"},
            },
            "descriptionModule": {
                "briefSummary": "A brief summary.",
                "detailedDescription": "A detailed description.",
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": 100},
                "designInfo": {
                    "allocation": "RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "primaryPurpose": "TREATMENT",
                    "maskingInfo": {"masking": "DOUBLE"},
                },
            },
            "eligibilityModule": {
                "eligibilityCriteria": "Inclusion: age > 18",
                "minimumAge": "18 Years",
                "maximumAge": "65 Years",
                "sex": "ALL",
                "healthyVolunteers": "No",
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "NIH", "class": "NIH"},
                "collaborators": [{"name": "ColabOrg"}],
            },
            "contactsLocationsModule": {
                "locations": [
                    {"country": "United States", "city": "Boston"},
                    {"country": "Canada", "city": "Toronto"},
                ]
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": "Drug A", "type": "DRUG"},
                    {"name": "Placebo", "type": "OTHER"},
                ]
            },
            "conditionsModule": {
                "conditions": ["Diabetes", "Obesity"],
                "keywords": ["insulin", "glucose"],
            },
        }
    }
    base.update(overrides)
    return base


# ──────────────────────────────────────────────
# extract_study — field mapping
# ──────────────────────────────────────────────

class TestExtractStudy:

    def test_all_32_fields_present(self):
        result = extract_study(make_study())
        expected_keys = [
            "NCT Number", "Title", "Study Title", "Recruitment Status",
            "Study Type", "Phase", "Enrollment", "Start Date",
            "Completion Date", "Primary Completion Date", "Last Update",
            "Brief Summary", "Detailed Description", "Conditions",
            "Keywords", "Allocation", "Intervention Model", "Primary Purpose",
            "Masking", "Interventions", "Intervention Types",
            "Eligibility Criteria", "Min Age", "Max Age", "Sex",
            "Accepts Healthy Volunteers", "Location Countries",
            "Location Cities", "Number of Locations", "Sponsor",
            "Sponsor Class", "Collaborators", "Study URL",
        ]
        for key in expected_keys:
            assert key in result, f"Missing field: {key}"

    def test_field_values_correct(self):
        result = extract_study(make_study())
        assert result["NCT Number"] == "NCT12345678"
        assert result["Recruitment Status"] == "RECRUITING"
        assert result["Phase"] == "PHASE2"
        assert result["Enrollment"] == 100
        assert result["Sponsor"] == "NIH"
        assert result["Number of Locations"] == 2
        assert "Drug A" in result["Interventions"]
        assert "Placebo" in result["Interventions"]
        assert "DRUG" in result["Intervention Types"]
        assert "Diabetes" in result["Conditions"]
        assert "insulin" in result["Keywords"]
        assert result["Study URL"] == "https://clinicaltrials.gov/study/NCT12345678"

    def test_empty_study_dict_does_not_crash(self):
        result = extract_study({})
        assert result["NCT Number"] == ""
        assert result["Number of Locations"] == 0
        assert result["Phase"] == ""
        assert result["Collaborators"] == ""

    def test_missing_protocol_section(self):
        result = extract_study({"protocolSection": {}})
        assert result["NCT Number"] == ""
        assert result["Enrollment"] == ""

    def test_empty_phases_list(self):
        study = make_study()
        study["protocolSection"]["designModule"]["phases"] = []
        result = extract_study(study)
        assert result["Phase"] == ""

    def test_empty_locations_list(self):
        study = make_study()
        study["protocolSection"]["contactsLocationsModule"]["locations"] = []
        result = extract_study(study)
        assert result["Number of Locations"] == 0
        assert result["Location Countries"] == ""
        assert result["Location Cities"] == ""

    def test_empty_collaborators(self):
        study = make_study()
        study["protocolSection"]["sponsorCollaboratorsModule"]["collaborators"] = []
        result = extract_study(study)
        assert result["Collaborators"] == ""

    def test_interventions_with_non_dict_items_skipped(self):
        study = make_study()
        study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
            {"name": "Drug A", "type": "DRUG"},
            "not_a_dict",
            None,
        ]
        result = extract_study(study)
        assert result["Interventions"] == "Drug A"
        assert result["Intervention Types"] == "DRUG"

    def test_enrollment_info_not_dict(self):
        study = make_study()
        study["protocolSection"]["designModule"]["enrollmentInfo"] = "not_a_dict"
        result = extract_study(study)
        assert result["Enrollment"] == ""

    def test_multiple_conditions_joined(self):
        result = extract_study(make_study())
        assert result["Conditions"] == "Diabetes; Obesity"

    def test_duplicate_countries_deduplicated(self):
        study = make_study()
        study["protocolSection"]["contactsLocationsModule"]["locations"] = [
            {"country": "United States", "city": "Boston"},
            {"country": "United States", "city": "NYC"},
        ]
        result = extract_study(study)
        assert result["Location Countries"].count("United States") == 1

    def test_lead_sponsor_not_dict(self):
        study = make_study()
        study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"] = "not_a_dict"
        result = extract_study(study)
        assert result["Sponsor"] == ""
        assert result["Sponsor Class"] == ""


# ──────────────────────────────────────────────
# download_raw_trials_csv — pagination + errors
# ──────────────────────────────────────────────

class TestDownloadRawTrialsCsv:

    def _mock_response(self, studies, next_token=None):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "studies": studies,
            **({"nextPageToken": next_token} if next_token else {}),
        }
        return mock

    def test_single_page_download(self, tmp_path):
        raw_path = str(tmp_path / "raw.csv")
        studies = [make_study()] * 3

        with patch("ingest.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(studies)
            count = download_raw_trials_csv(raw_path, "diabetes", sleep_seconds=0)

        assert count == 3
        df = pd.read_csv(raw_path)
        assert len(df) == 3
        assert "NCT Number" in df.columns

    def test_pagination_follows_next_token(self, tmp_path):
        raw_path = str(tmp_path / "raw.csv")
        page1 = self._mock_response([make_study()] * 2, next_token="TOKEN1")
        page2 = self._mock_response([make_study()] * 2)

        with patch("ingest.requests.get") as mock_get:
            mock_get.side_effect = [page1, page2]
            count = download_raw_trials_csv(raw_path, "diabetes", sleep_seconds=0)

        assert count == 4
        assert mock_get.call_count == 2

    def test_api_error_breaks_gracefully(self, tmp_path):
        import requests as req
        raw_path = str(tmp_path / "raw.csv")

        with patch("ingest.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.RequestException("API down")
            count = download_raw_trials_csv(raw_path, "diabetes", sleep_seconds=0)

        # When API fails on first page, 0 trials collected
        assert count == 0
        # File is written but empty — just verify it exists
        assert os.path.exists(raw_path)

    def test_bad_study_skipped_continues(self, tmp_path):
        raw_path = str(tmp_path / "raw.csv")
        # One valid study, one will cause extract_study to error (we force it)
        studies = [make_study(), make_study()]

        with patch("ingest.requests.get") as mock_get:
            mock_get.return_value = self._mock_response(studies)
            with patch("ingest.extract_study", side_effect=[{"NCT Number": "NCT00000001"}, Exception("bad")]):
                count = download_raw_trials_csv(raw_path, "diabetes", sleep_seconds=0)

        # 1 valid extracted, 1 skipped
        assert count == 1


# ──────────────────────────────────────────────
# enrich_trials_csv — kept from original + extras
# ──────────────────────────────────────────────

class TestEnrichTrialsCsv:

    @pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
    def test_dedupes_and_adds_columns(self, tmp_path, dkey):
        raw_path = tmp_path / "raw.csv"
        enriched_path = tmp_path / "enriched.csv"
        df = pd.DataFrame({
            "NCT Number": ["N1", "N1", "N2"],
            "Conditions": ["Breast Cancer", "Breast Cancer", "Diabetes"],
        })
        df.to_csv(raw_path, index=False)

        disease = REGISTRY[dkey]["disease"]
        classifier = REGISTRY[dkey]["classifier"]
        out_df = enrich_trials_csv(str(raw_path), str(enriched_path), disease, classifier)

        out = pd.read_csv(enriched_path)
        assert len(out) == 2
        assert set(out["NCT Number"].tolist()) == {"N1", "N2"}
        assert "disease" in out.columns
        assert "disease_type" in out.columns
        assert (out["disease"] == disease).all()

    def test_conditions_with_nan_does_not_crash(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        enriched_path = tmp_path / "enriched.csv"
        df = pd.DataFrame({
            "NCT Number": ["N1", "N2"],
            "Conditions": [None, "Diabetes"],
        })
        df.to_csv(raw_path, index=False)
        classifier = REGISTRY["diabetes"]["classifier"]
        # Should not raise
        enrich_trials_csv(str(raw_path), str(enriched_path), "diabetes", classifier)
        out = pd.read_csv(enriched_path)
        assert len(out) == 2

    def test_enriched_file_written_to_disk(self, tmp_path):
        raw_path = tmp_path / "raw.csv"
        enriched_path = tmp_path / "enriched.csv"
        df = pd.DataFrame({"NCT Number": ["N1"], "Conditions": ["Diabetes"]})
        df.to_csv(raw_path, index=False)
        enrich_trials_csv(str(raw_path), str(enriched_path), "diabetes", lambda x: "T2")
        assert enriched_path.exists()
