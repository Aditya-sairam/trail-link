import os
import sys
import json
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schema import (
    generate_schema,
    validate_against_schema,
    run_schema_checkpoint,
    RAW_REQUIRED_DEFAULT,
    PROCESSED_REQUIRED_DEFAULT,
    _dtype_family,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_valid_df():
    return pd.DataFrame({
        "NCT Number": ["NCT12345678", "NCT00000001"],
        "Study Title": ["Trial A", "Trial B"],
        "Recruitment Status": ["RECRUITING", "COMPLETED"],
        "Conditions": ["Diabetes", "Cancer"],
        "Interventions": ["Drug", "Placebo"],
        "Sponsor": ["NIH", "NIH"],
        "Enrollment": [100, 200],
        "Sex": ["ALL", "FEMALE"],
        "Location Countries": ["United States", "Canada"],
        "Location Cities": ["Boston", "Toronto"],
    })


# ──────────────────────────────────────────────
# _dtype_family
# ──────────────────────────────────────────────

class TestDtypeFamily:
    def test_string_column(self):
        s = pd.Series(["a", "b"])
        assert _dtype_family(s) == "string"

    def test_numeric_column(self):
        s = pd.Series([1, 2, 3])
        assert _dtype_family(s) == "number"

    def test_bool_column(self):
        s = pd.Series([True, False])
        assert _dtype_family(s) == "bool"

    def test_datetime_column(self):
        s = pd.to_datetime(pd.Series(["2024-01-01", "2024-06-01"]))
        assert _dtype_family(s) == "datetime"


# ──────────────────────────────────────────────
# generate_schema
# ──────────────────────────────────────────────

class TestGenerateSchema:
    def test_output_keys_present(self):
        df = make_valid_df()
        schema = generate_schema(df, required_columns=RAW_REQUIRED_DEFAULT)
        assert "generated_at" in schema
        assert "required_columns" in schema
        assert "rules" in schema
        assert "columns" in schema

    def test_all_columns_described(self):
        df = make_valid_df()
        schema = generate_schema(df, required_columns=RAW_REQUIRED_DEFAULT)
        for col in df.columns:
            assert col in schema["columns"]

    def test_column_metadata_structure(self):
        df = make_valid_df()
        schema = generate_schema(df, required_columns=RAW_REQUIRED_DEFAULT)
        meta = schema["columns"]["Enrollment"]
        assert "type" in meta
        assert "null_pct" in meta
        assert "n_unique" in meta

    def test_null_pct_correct(self):
        df = pd.DataFrame({"NCT Number": ["N1", None, None, None]})
        schema = generate_schema(df, required_columns=[])
        assert schema["columns"]["NCT Number"]["null_pct"] == 75.0

    def test_required_columns_stored(self):
        df = make_valid_df()
        schema = generate_schema(df, required_columns=RAW_REQUIRED_DEFAULT)
        assert schema["required_columns"] == RAW_REQUIRED_DEFAULT


# ──────────────────────────────────────────────
# validate_against_schema
# ──────────────────────────────────────────────

class TestValidateAgainstSchema:
    def _make_baseline(self, df):
        return generate_schema(df, required_columns=RAW_REQUIRED_DEFAULT)

    def test_valid_df_passes(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        result = validate_against_schema(df, baseline)
        assert result["passed"] is True
        assert result["violations"] == []

    def test_missing_required_column_fails(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        df2 = df.drop(columns=["Study Title"])
        result = validate_against_schema(df2, baseline)
        assert result["passed"] is False
        assert any("Study Title" in v for v in result["violations"])

    def test_type_drift_detected(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        # Change Enrollment to string to cause type drift
        df2 = df.copy()
        df2["Enrollment"] = df2["Enrollment"].astype(str)
        result = validate_against_schema(df2, baseline)
        assert result["passed"] is False
        assert any("Enrollment" in v and "drift" in v for v in result["violations"])

    def test_null_pct_exceeds_threshold_fails(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        # Make NCT Number 100% null — required column
        df2 = df.copy()
        df2["NCT Number"] = None
        result = validate_against_schema(df2, baseline, required_columns=["NCT Number"])
        assert result["passed"] is False
        assert any("NCT Number" in v and "null_pct" in v for v in result["violations"])

    def test_invalid_sex_category_fails(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        df2 = df.copy()
        df2["Sex"] = ["UNKNOWN", "UNKNOWN"]
        result = validate_against_schema(df2, baseline)
        assert result["passed"] is False
        assert any("Sex" in v for v in result["violations"])

    def test_new_columns_flagged_when_not_allowed(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        df2 = df.copy()
        df2["extra_column"] = "value"
        result = validate_against_schema(df2, baseline, allow_new_columns=False)
        assert result["passed"] is False
        assert any("extra_column" in v for v in result["violations"])

    def test_new_columns_allowed_by_default(self):
        df = make_valid_df()
        baseline = self._make_baseline(df)
        df2 = df.copy()
        df2["extra_column"] = "value"
        result = validate_against_schema(df2, baseline, allow_new_columns=True)
        assert result["passed"] is True

    def test_numeric_range_min_violation(self):
        df = make_valid_df()
        baseline = generate_schema(
            df,
            required_columns=RAW_REQUIRED_DEFAULT,
            numeric_ranges={"Enrollment": {"min": 0, "max": 50}},
        )
        result = validate_against_schema(df, baseline)
        assert result["passed"] is False
        assert any("Enrollment" in v and "> 50" in v for v in result["violations"])


# ──────────────────────────────────────────────
# run_schema_checkpoint
# ──────────────────────────────────────────────

class TestRunSchemaCheckpoint:
    def test_creates_baseline_and_passes(self, tmp_path):
        df = make_valid_df()
        csv_path = str(tmp_path / "raw.csv")
        baseline_path = str(tmp_path / "schema.json")
        report_path = str(tmp_path / "report.json")
        df.to_csv(csv_path, index=False)
        report = run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_path,
            report_path=report_path,
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="enforce",
        )
        assert report["passed"] is True
        assert os.path.exists(baseline_path)
        assert os.path.exists(report_path)

    def test_enforce_mode_raises_on_violation(self, tmp_path):
        df = make_valid_df()
        csv_path = str(tmp_path / "raw.csv")
        baseline_path = str(tmp_path / "schema.json")
        report_path = str(tmp_path / "report.json")
        df.to_csv(csv_path, index=False)
        # Create baseline first
        run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_path,
            report_path=report_path,
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="enforce",
        )
        # Now drop a required column and re-run in enforce mode
        df2 = df.drop(columns=["Study Title"])
        df2.to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            run_schema_checkpoint(
                csv_path=csv_path,
                baseline_schema_path=baseline_path,
                report_path=report_path,
                required_columns=RAW_REQUIRED_DEFAULT,
                mode="enforce",
            )

    def test_warn_mode_does_not_raise_on_violation(self, tmp_path):
        df = make_valid_df()
        csv_path = str(tmp_path / "raw.csv")
        baseline_path = str(tmp_path / "schema.json")
        report_path = str(tmp_path / "report.json")
        df.to_csv(csv_path, index=False)
        run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_path,
            report_path=report_path,
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="warn",
        )
        # Drop column and re-run in warn mode — should NOT raise
        df2 = df.drop(columns=["Study Title"])
        df2.to_csv(csv_path, index=False)
        report = run_schema_checkpoint(
            csv_path=csv_path,
            baseline_schema_path=baseline_path,
            report_path=report_path,
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="warn",
        )
        assert report["passed"] is False  # violation exists
        # But no exception was raised

    def test_processed_required_includes_disease_columns(self):
        assert "disease" in PROCESSED_REQUIRED_DEFAULT
        assert "disease_type" in PROCESSED_REQUIRED_DEFAULT