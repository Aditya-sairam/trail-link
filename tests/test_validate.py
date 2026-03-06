import os
import sys
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validate import run_validation, REQUIRED_COLUMNS


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_valid_df(n=20):
    """Return a fully valid enriched DataFrame."""
    return pd.DataFrame({
        "NCT Number":          [f"NCT{str(i).zfill(8)}" for i in range(n)],
        "Study Title":         [f"Trial {i}" for i in range(n)],
        "Recruitment Status":  ["RECRUITING"] * n,
        "Conditions":          ["Diabetes"] * n,
        "Interventions":       ["Drug"] * n,
        "Sponsor":             ["NIH"] * n,
        "Enrollment":          [100] * n,
        "Sex":                 ["ALL"] * n,
        "Location Countries":  ["United States"] * n,
        "Location Cities":     ["Boston"] * n,
        "disease":             ["diabetes"] * n,
        "disease_type":        ["Type 2 Diabetes"] * n,
    })


def write_csv(df, tmp_path, name="enriched.csv"):
    path = str(tmp_path / name)
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────
# File existence
# ──────────────────────────────────────────────

class TestFileExistence:
    def test_file_not_found_returns_false(self):
        assert run_validation("/nonexistent/path/file.csv") is False


# ──────────────────────────────────────────────
# Row count checks
# ──────────────────────────────────────────────

class TestRowCount:
    def test_empty_csv_fails(self, tmp_path):
        df = pd.DataFrame(columns=make_valid_df().columns)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is False

    def test_fewer_than_10_rows_fails(self, tmp_path):
        df = make_valid_df(n=5)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is False

    def test_exactly_10_rows_passes(self, tmp_path):
        df = make_valid_df(n=10)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True

    def test_20_rows_passes(self, tmp_path):
        df = make_valid_df(n=20)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True


# ──────────────────────────────────────────────
# Schema checks — required columns
# ──────────────────────────────────────────────

class TestRequiredColumns:
    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_missing_any_required_column_fails(self, tmp_path, col):
        df = make_valid_df()
        df = df.drop(columns=[col])
        path = write_csv(df, tmp_path, name=f"missing_{col.replace(' ', '_')}.csv")
        assert run_validation(path) is False


# ──────────────────────────────────────────────
# NCT Number format
# ──────────────────────────────────────────────

class TestNctFormat:
    def test_valid_nct_format_passes(self, tmp_path):
        df = make_valid_df(n=20)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True

    def test_over_10_pct_invalid_nct_fails(self, tmp_path):
        df = make_valid_df(n=20)
        # Make 4 out of 20 invalid (20%) → exceeds 10% threshold
        df.loc[:3, "NCT Number"] = ["BADFORMAT"] * 4
        path = write_csv(df, tmp_path)
        assert run_validation(path) is False

    def test_under_10_pct_invalid_nct_passes(self, tmp_path):
        df = make_valid_df(n=20)
        # Make 1 out of 20 invalid (5%) → under threshold
        df.loc[0, "NCT Number"] = "BADFORMAT"
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True


# ──────────────────────────────────────────────
# disease column
# ──────────────────────────────────────────────

class TestDiseaseColumn:
    def test_disease_nulls_fail(self, tmp_path):
        df = make_valid_df(n=20)
        df.loc[0, "disease"] = None
        path = write_csv(df, tmp_path)
        assert run_validation(path) is False

    def test_disease_no_nulls_passes(self, tmp_path):
        df = make_valid_df(n=20)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True


# ──────────────────────────────────────────────
# disease_type column
# ──────────────────────────────────────────────

class TestDiseaseTypeColumn:
    def test_over_50_pct_unknown_fails(self, tmp_path):
        df = make_valid_df(n=20)
        # 11 out of 20 unknown → 55%
        df.loc[:10, "disease_type"] = "Unknown"
        path = write_csv(df, tmp_path)
        assert run_validation(path) is False

    def test_under_50_pct_unknown_passes(self, tmp_path):
        df = make_valid_df(n=20)
        # 5 out of 20 unknown → 25%
        df.loc[:4, "disease_type"] = "Unknown"
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True


# ──────────────────────────────────────────────
# Critical column nulls
# ──────────────────────────────────────────────

class TestCriticalNulls:
    @pytest.mark.parametrize("col", ["NCT Number", "Study Title", "Conditions"])
    def test_over_80_pct_null_critical_column_fails(self, tmp_path, col):
        df = make_valid_df(n=20)
        # Set 17 out of 20 to None → 85% null
        df.loc[:16, col] = None
        path = write_csv(df, tmp_path, name=f"null_{col.replace(' ', '_')}.csv")
        assert run_validation(path) is False

    def test_valid_critical_columns_pass(self, tmp_path):
        df = make_valid_df(n=20)
        path = write_csv(df, tmp_path)
        assert run_validation(path) is True
