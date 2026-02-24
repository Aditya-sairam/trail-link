"""
Unit tests for the breast cancer trials pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import json
import os
import pytest
import pandas as pd
import numpy as np

from src.pipelines.breast_cancer.ingest import (
    sanitize_filename,
    extract_cancer_type,
    normalize_cancer_type,
    enrich_trials_csv,
)
from src.pipelines.breast_cancer.quality import (
    generate_stats,
    detect_anomalies,
    anomalies_found,
    run_quality_checks,
)
from src.pipelines.breast_cancer.bias import (
    generate_bias_report,
    save_bias_report,
    _value_counts_with_pct,
    _missingness_by_slice,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ingest.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSanitizeFilename:

    def test_spaces_replaced_with_underscore(self):
        assert sanitize_filename("breast cancer") == "breast_cancer"

    def test_uppercase_lowercased(self):
        assert sanitize_filename("BreastCancer") == "breastcancer"

    def test_special_characters_removed(self):
        assert sanitize_filename("breast-cancer!@#") == "breastcancer"

    def test_double_underscores_collapsed(self):
        assert sanitize_filename("breast  cancer") == "breast_cancer"

    def test_leading_trailing_underscores_stripped(self):
        assert sanitize_filename("_breast_cancer_") == "breast_cancer"

    def test_numbers_preserved(self):
        assert sanitize_filename("stage2 cancer") == "stage2_cancer"

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_only_special_chars(self):
        # all chars removed, result is empty
        assert sanitize_filename("!@#$%") == ""


# ──────────────────────────────────────────────────────────────────────────────

class TestExtractCancerType:

    def test_nan_returns_default(self):
        assert extract_cancer_type(float("nan")) == "breast_cancer"

    def test_none_returns_default(self):
        assert extract_cancer_type(None) == "breast_cancer"

    def test_single_breast_cancer_condition(self):
        result = extract_cancer_type("Breast Cancer")
        assert result == "breast_cancer"

    def test_pipe_separated_picks_breast_cancer_entry(self):
        result = extract_cancer_type("Diabetes|Breast Cancer|Hypertension")
        assert result == "breast_cancer"

    def test_her2_condition_extracted(self):
        result = extract_cancer_type("HER2-Positive Breast Cancer")
        assert "breast" in result or "her2" in result

    def test_no_breast_cancer_in_conditions_returns_default(self):
        result = extract_cancer_type("Lung Cancer|Diabetes")
        assert result == "breast_cancer"

    def test_case_insensitive_match(self):
        result = extract_cancer_type("BREAST CANCER")
        assert result == "breast_cancer"

    def test_empty_string_returns_default(self):
        assert extract_cancer_type("") == "breast_cancer"


# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeCancerType:

    def test_triple_negative(self):
        assert normalize_cancer_type("Triple-Negative Breast Cancer") == "triple_negative"

    def test_triple_negative_lowercase(self):
        assert normalize_cancer_type("triple negative breast cancer") == "triple_negative"

    def test_metastatic(self):
        assert normalize_cancer_type("Metastatic Breast Cancer") == "metastatic"

    def test_her2_positive(self):
        assert normalize_cancer_type("HER2-Positive Breast Cancer") == "her2_positive"

    def test_stage_specific(self):
        assert normalize_cancer_type("Stage IV Breast Cancer") == "stage_specific"

    def test_general_breast_cancer(self):
        assert normalize_cancer_type("Breast Cancer") == "general_breast_cancer"

    def test_nan_returns_nan(self):
        result = normalize_cancer_type(float("nan"))
        assert result is float("nan") or pd.isna(result)

    def test_none_returns_nan_like(self):
        result = normalize_cancer_type(None)
        assert pd.isna(result)

    def test_whitespace_stripped(self):
        assert normalize_cancer_type("  metastatic breast cancer  ") == "metastatic"


# ──────────────────────────────────────────────────────────────────────────────

class TestEnrichTrialsCsv:
    """
    Tests for enrich_trials_csv — we write a temp raw CSV,
    call the function, then read and assert on the output.
    """

    BASE_ROW = {
        "NCTId": "NCT001",
        "Conditions": "Breast Cancer",
        "BriefTitle": "  Trial A  ",
        "Phases": "Phase 2",
        "Start Date": "2021-01-01",
        "Enrollment": "150",
        "Results First Posted": "2022-01-01",   # should be dropped
        "Study Documents": "doc.pdf",            # should be dropped
    }

    def _write_raw_csv(self, tmp_path, rows):
        raw = tmp_path / "raw.csv"
        pd.DataFrame(rows).to_csv(raw, index=False)
        return str(raw)

    def _enrich(self, tmp_path, rows):
        raw_path = self._write_raw_csv(tmp_path, rows)
        enriched_path = str(tmp_path / "enriched.csv")
        enrich_trials_csv(raw_path, enriched_path)
        return pd.read_csv(enriched_path)

    def test_disease_column_always_added(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert "disease" in df.columns
        assert df["disease"].iloc[0] == "breast_cancer"

    def test_cancer_type_column_added(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert "cancer_type" in df.columns

    def test_dropped_columns_removed(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert "Results First Posted" not in df.columns
        assert "Study Documents" not in df.columns

    def test_text_columns_trimmed(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert df["BriefTitle"].iloc[0] == "Trial A"

    def test_phases_prefix_removed(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert df["Phases"].iloc[0] == "2"

    def test_enrollment_is_numeric(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert pd.api.types.is_numeric_dtype(df["Enrollment"])
        assert df["Enrollment"].iloc[0] == 150

    def test_enrollment_non_numeric_becomes_nan(self, tmp_path):
        row = {**self.BASE_ROW, "Enrollment": "unknown"}
        df = self._enrich(tmp_path, [row])
        assert pd.isna(df["Enrollment"].iloc[0])

    def test_start_date_parsed_as_datetime(self, tmp_path):
        df = self._enrich(tmp_path, [self.BASE_ROW])
        assert pd.api.types.is_datetime64_any_dtype(df["Start Date"])

    def test_bad_date_becomes_nat(self, tmp_path):
        row = {**self.BASE_ROW, "Start Date": "not-a-date"}
        df = self._enrich(tmp_path, [row])
        assert pd.isna(df["Start Date"].iloc[0])

    def test_deduplication_on_nct_id(self, tmp_path):
        rows = [self.BASE_ROW, self.BASE_ROW]   # duplicate
        df = self._enrich(tmp_path, rows)
        assert len(df) == 1

    def test_no_conditions_column_defaults_cancer_type(self, tmp_path):
        row = {"NCTId": "NCT002", "BriefTitle": "Some Trial"}
        df = self._enrich(tmp_path, [row])
        assert df["cancer_type"].iloc[0] == "breast_cancer"

    def test_empty_dataframe_does_not_crash(self, tmp_path):
        raw = tmp_path / "raw.csv"
        pd.DataFrame(columns=["NCTId", "Conditions"]).to_csv(raw, index=False)
        enriched_path = str(tmp_path / "enriched.csv")
        enrich_trials_csv(str(raw), enriched_path)
        df = pd.read_csv(enriched_path)
        assert len(df) == 0

    def test_returns_enriched_file_path(self, tmp_path):
        raw_path = self._write_raw_csv(tmp_path, [self.BASE_ROW])
        enriched_path = str(tmp_path / "enriched.csv")
        result = enrich_trials_csv(raw_path, enriched_path)
        assert result == enriched_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — quality.py
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def clean_df():
    """A clean, valid DataFrame with no anomalies."""
    return pd.DataFrame({
        "NCT Number": ["NCT001", "NCT002", "NCT003"],
        "BriefTitle": ["Trial A", "Trial B", "Trial C"],
        "Phases": ["2", "3", "1"],
    })

@pytest.fixture
def dirty_df():
    """DataFrame with duplicates and a mostly-missing column."""
    return pd.DataFrame({
        "NCT Number": ["NCT001", "NCT001", "NCT003"],   # duplicate
        "BriefTitle": ["Trial A", "Trial A", "Trial C"],
        "AlmostEmpty": [None, None, None],               # 100% missing
    })


class TestGenerateStats:

    def test_row_count_correct(self, clean_df):
        stats = generate_stats(clean_df)
        assert stats["rows"] == 3

    def test_col_count_correct(self, clean_df):
        stats = generate_stats(clean_df)
        assert stats["cols"] == 3

    def test_column_names_listed(self, clean_df):
        stats = generate_stats(clean_df)
        assert "NCT Number" in stats["column_names"]

    def test_unique_nct_number_counted(self, clean_df):
        stats = generate_stats(clean_df)
        assert stats["unique_nct_number"] == 3

    def test_unique_nct_number_with_duplicates(self, dirty_df):
        stats = generate_stats(dirty_df)
        assert stats["unique_nct_number"] == 2   # NCT001 appears twice

    def test_missing_by_col_tracked(self):
        df = pd.DataFrame({
            "NCT Number": ["NCT001", None],
            "BriefTitle": ["Trial A", "Trial B"],
        })
        stats = generate_stats(df)
        assert stats["missing_by_col"]["NCT Number"] == 1
        assert stats["missing_by_col"]["BriefTitle"] == 0

    def test_no_nct_column_skips_unique_count(self):
        df = pd.DataFrame({"OtherCol": [1, 2, 3]})
        stats = generate_stats(df)
        assert "unique_nct_number" not in stats

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["NCT Number", "BriefTitle"])
        stats = generate_stats(df)
        assert stats["rows"] == 0
        assert stats["unique_nct_number"] == 0


class TestDetectAnomalies:

    def test_clean_df_no_anomalies(self, clean_df):
        anomalies = detect_anomalies(clean_df)
        assert anomalies["missing_required_columns"] == []
        assert anomalies["duplicate_nct_number_found"] is False
        assert anomalies["high_missing_columns"] == []

    def test_missing_nct_column_flagged(self):
        df = pd.DataFrame({"BriefTitle": ["Trial A"]})
        anomalies = detect_anomalies(df)
        assert "NCT Number" in anomalies["missing_required_columns"]

    def test_duplicate_nct_flagged(self, dirty_df):
        anomalies = detect_anomalies(dirty_df)
        assert anomalies["duplicate_nct_number_found"] is True

    def test_no_duplicate_nct_not_flagged(self, clean_df):
        anomalies = detect_anomalies(clean_df)
        assert anomalies["duplicate_nct_number_found"] is False

    def test_high_missing_column_flagged(self, dirty_df):
        anomalies = detect_anomalies(dirty_df)
        flagged_cols = [item["col"] for item in anomalies["high_missing_columns"]]
        assert "AlmostEmpty" in flagged_cols

    def test_high_missing_fraction_value_correct(self, dirty_df):
        anomalies = detect_anomalies(dirty_df)
        flagged = {item["col"]: item["missing_frac"] for item in anomalies["high_missing_columns"]}
        assert flagged["AlmostEmpty"] == 1.0

    def test_column_below_70pct_missing_not_flagged(self):
        df = pd.DataFrame({
            "NCT Number": ["NCT001", "NCT002"],
            "SomeCol": [None, "value"],   # 50% missing — under threshold
        })
        anomalies = detect_anomalies(df)
        flagged_cols = [item["col"] for item in anomalies["high_missing_columns"]]
        assert "SomeCol" not in flagged_cols

    def test_empty_dataframe_flags_missing_required_col(self):
        df = pd.DataFrame()
        anomalies = detect_anomalies(df)
        assert "NCT Number" in anomalies["missing_required_columns"]


class TestAnomaliesFound:

    def test_returns_false_when_clean(self):
        anomalies = {
            "missing_required_columns": [],
            "duplicate_nct_number_found": False,
            "high_missing_columns": [],
        }
        assert anomalies_found(anomalies) is False

    def test_returns_true_for_missing_col(self):
        anomalies = {
            "missing_required_columns": ["NCT Number"],
            "duplicate_nct_number_found": False,
            "high_missing_columns": [],
        }
        assert anomalies_found(anomalies) is True

    def test_returns_true_for_duplicate(self):
        anomalies = {
            "missing_required_columns": [],
            "duplicate_nct_number_found": True,
            "high_missing_columns": [],
        }
        assert anomalies_found(anomalies) is True

    def test_returns_true_for_high_missing(self):
        anomalies = {
            "missing_required_columns": [],
            "duplicate_nct_number_found": False,
            "high_missing_columns": [{"col": "X", "missing_frac": 0.9}],
        }
        assert anomalies_found(anomalies) is True


class TestRunQualityChecks:

    def test_creates_stats_and_anomaly_json_files(self, tmp_path, clean_df):
        csv_path = str(tmp_path / "enriched.csv")
        stats_path = str(tmp_path / "stats.json")
        anomalies_path = str(tmp_path / "anomalies.json")
        clean_df.to_csv(csv_path, index=False)

        run_quality_checks(csv_path, stats_path, anomalies_path)

        assert os.path.exists(stats_path)
        assert os.path.exists(anomalies_path)

    def test_stats_json_is_valid(self, tmp_path, clean_df):
        csv_path = str(tmp_path / "enriched.csv")
        stats_path = str(tmp_path / "stats.json")
        anomalies_path = str(tmp_path / "anomalies.json")
        clean_df.to_csv(csv_path, index=False)

        run_quality_checks(csv_path, stats_path, anomalies_path)

        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["rows"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — bias.py
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def bias_df():
    return pd.DataFrame({
        "Sex": ["Female", "Female", "Male", None],
        "Phase": ["2", "3", "2", "3"],
        "StudyType": ["Interventional"] * 4,
        "OverallStatus": ["RECRUITING"] * 4,
        "NCTId": ["NCT001", "NCT002", "NCT003", "NCT004"],
        "MinimumAge": [18, None, 21, 30],
    })


class TestValueCountsWithPct:

    def test_counts_sum_to_total(self):
        s = pd.Series(["A", "B", "A", "C"])
        result = _value_counts_with_pct(s)
        assert sum(result["counts"].values()) == 4

    def test_percentages_sum_to_one(self):
        s = pd.Series(["A", "B", "A", "C"])
        result = _value_counts_with_pct(s)
        assert abs(sum(result["pct"].values()) - 1.0) < 1e-6

    def test_nan_becomes_missing_key(self):
        s = pd.Series(["A", None, "A"])
        result = _value_counts_with_pct(s)
        assert "MISSING" in result["counts"]

    def test_all_nan_series(self):
        s = pd.Series([None, None, None])
        result = _value_counts_with_pct(s)
        assert result["counts"]["MISSING"] == 3

    def test_single_value_series(self):
        s = pd.Series(["Female"] * 5)
        result = _value_counts_with_pct(s)
        assert result["pct"]["Female"] == pytest.approx(1.0)


class TestMissingnessBySlice:

    def test_returns_dict_per_sampled_col(self, bias_df):
        result = _missingness_by_slice(bias_df, "Sex", ["MinimumAge"])
        assert "MinimumAge" in result

    def test_missing_fraction_correct(self, bias_df):
        # "Female" group: rows 0,1 — MinimumAge = [18, None] → 0.5 missing
        result = _missingness_by_slice(bias_df, "Sex", ["MinimumAge"])
        assert result["MinimumAge"]["Female"] == pytest.approx(0.5)

    def test_nan_slice_grouped_as_missing(self, bias_df):
        result = _missingness_by_slice(bias_df, "Sex", ["MinimumAge"])
        assert "MISSING" in result["MinimumAge"]

    def test_nonexistent_sample_col_skipped(self, bias_df):
        result = _missingness_by_slice(bias_df, "Sex", ["DoesNotExist"])
        assert "DoesNotExist" not in result

    def test_empty_sample_cols_returns_empty(self, bias_df):
        result = _missingness_by_slice(bias_df, "Sex", [])
        assert result == {}


class TestGenerateBiasReport:

    def test_report_has_required_keys(self, bias_df):
        report = generate_bias_report(bias_df, slice_columns=["Sex"])
        for key in ["dataset_rows", "slice_columns_requested", "slice_columns_found", "slices", "notes"]:
            assert key in report

    def test_dataset_rows_matches_input(self, bias_df):
        report = generate_bias_report(bias_df, slice_columns=["Sex"])
        assert report["dataset_rows"] == len(bias_df)

    def test_found_columns_only_includes_existing(self, bias_df):
        report = generate_bias_report(bias_df, slice_columns=["Sex", "NonExistentCol"])
        assert "Sex" in report["slice_columns_found"]
        assert "NonExistentCol" not in report["slice_columns_found"]

    def test_missing_column_gets_error_entry(self, bias_df):
        report = generate_bias_report(bias_df, slice_columns=["NonExistentCol"])
        assert report["slices"]["NonExistentCol"]["error"] == "column_not_found"

    def test_existing_slice_has_representation_and_missingness(self, bias_df):
        report = generate_bias_report(bias_df, slice_columns=["Sex"])
        assert "representation" in report["slices"]["Sex"]
        assert "missingness_by_slice" in report["slices"]["Sex"]

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame(columns=["Sex", "Phase"])
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert report["dataset_rows"] == 0

    def test_all_nan_slice_column(self):
        df = pd.DataFrame({"Sex": [None, None, None], "Phase": ["2", "3", "2"]})
        report = generate_bias_report(df, slice_columns=["Sex"])
        counts = report["slices"]["Sex"]["representation"]["counts"]
        assert "MISSING" in counts


class TestSaveBiasReport:

    def test_file_created(self, tmp_path):
        report = {"dataset_rows": 10, "slices": {}}
        out_path = str(tmp_path / "reports" / "bias_report.json")
        save_bias_report(report, out_path)
        assert os.path.exists(out_path)

    def test_file_content_is_valid_json(self, tmp_path):
        report = {"dataset_rows": 10, "slices": {}}
        out_path = str(tmp_path / "bias_report.json")
        save_bias_report(report, out_path)
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["dataset_rows"] == 10

    def test_returns_output_path(self, tmp_path):
        out_path = str(tmp_path / "bias_report.json")
        result = save_bias_report({}, out_path)
        assert result == out_path