import os
import sys
import json
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from stats import compute_stats, _top_countries


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_df(n=10):
    return pd.DataFrame({
        "NCT Number":         [f"NCT{str(i).zfill(8)}" for i in range(n)],
        "disease":            ["diabetes"] * n,
        "disease_type":       (["Type 1"] * (n // 2)) + (["Type 2"] * (n - n // 2)),
        "Recruitment Status": ["RECRUITING"] * n,
        "Sex":                ["ALL"] * n,
        "Enrollment":         list(range(10, 10 + n)),
        "Sponsor":            ["NIH"] * (n // 2) + ["Pfizer"] * (n - n // 2),
        "Location Countries": ["United States"] * (n - 1) + ["Canada"],
    })


def write_csv(df, tmp_path, name="enriched.csv"):
    path = str(tmp_path / name)
    df.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────
# compute_stats — basic structure
# ──────────────────────────────────────────────

class TestComputeStatsStructure:
    def test_basic_keys_present(self, tmp_path):
        df = make_df()
        path = write_csv(df, tmp_path)
        stats_path = str(tmp_path / "stats.json")
        stats = compute_stats(path, stats_path)
        assert "total_trials" in stats
        assert "total_columns" in stats
        assert "columns" in stats

    def test_total_trials_correct(self, tmp_path):
        df = make_df(n=15)
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["total_trials"] == 15

    def test_total_columns_correct(self, tmp_path):
        df = make_df()
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["total_columns"] == len(df.columns)

    def test_stats_json_written_to_disk(self, tmp_path):
        df = make_df()
        path = write_csv(df, tmp_path)
        stats_path = str(tmp_path / "stats.json")
        compute_stats(path, stats_path)
        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            loaded = json.load(f)
        assert loaded["total_trials"] == len(df)


# ──────────────────────────────────────────────
# compute_stats — disease and disease_type
# ──────────────────────────────────────────────

class TestComputeStatsDiseaseFields:
    def test_disease_captured(self, tmp_path):
        df = make_df()
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["disease"] == "diabetes"

    def test_disease_type_distribution(self, tmp_path):
        df = make_df(n=10)
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert "Type 1" in stats["disease_type_distribution"]
        assert "Type 2" in stats["disease_type_distribution"]


# ──────────────────────────────────────────────
# compute_stats — enrollment
# ──────────────────────────────────────────────

class TestComputeStatsEnrollment:
    def test_enrollment_stats_correct(self, tmp_path):
        df = pd.DataFrame({
            "Enrollment": [10, 20, 30],
            "Location Countries": ["United States"] * 3,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["enrollment"]["mean"] == 20.0
        assert stats["enrollment"]["median"] == 20.0
        assert stats["enrollment"]["min"] == 10
        assert stats["enrollment"]["max"] == 30
        assert stats["enrollment"]["total"] == 60

    def test_enrollment_with_nans_handled(self, tmp_path):
        df = pd.DataFrame({
            "Enrollment": [10, None, 30],
            "Location Countries": ["United States"] * 3,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["enrollment"]["mean"] == 20.0
        assert stats["enrollment"]["total"] == 40


# ──────────────────────────────────────────────
# compute_stats — sponsors
# ──────────────────────────────────────────────

class TestComputeStatsSponsors:
    def test_top_sponsors_returned(self, tmp_path):
        df = make_df(n=10)
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert "top_sponsors" in stats
        assert "NIH" in stats["top_sponsors"]

    def test_top_sponsors_max_10(self, tmp_path):
        sponsors = [f"Sponsor{i}" for i in range(20)]
        df = pd.DataFrame({
            "Sponsor": sponsors,
            "Location Countries": ["United States"] * 20,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert len(stats["top_sponsors"]) <= 10


# ──────────────────────────────────────────────
# compute_stats — geography
# ──────────────────────────────────────────────

class TestComputeStatsGeography:
    def test_us_trials_count_correct(self, tmp_path):
        df = pd.DataFrame({
            "Location Countries": ["United States"] * 7 + ["Canada"] * 3,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["geography"]["us_trials"] == 7
        assert stats["geography"]["international_trials"] == 3
        assert stats["geography"]["us_percentage"] == 70.0

    def test_top_countries_included(self, tmp_path):
        df = make_df(n=10)
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert "United States" in stats["geography"]["top_countries"]


# ──────────────────────────────────────────────
# compute_stats — missing values
# ──────────────────────────────────────────────

class TestComputeStatsMissingValues:
    def test_missing_values_only_includes_null_columns(self, tmp_path):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [None, None, 1],
            "Location Countries": ["United States"] * 3,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert "B" in stats["missing_values"]
        assert "A" not in stats["missing_values"]

    def test_missing_values_count_correct(self, tmp_path):
        df = pd.DataFrame({
            "A": [None, None, 1],
            "Location Countries": ["United States"] * 3,
        })
        path = write_csv(df, tmp_path)
        stats = compute_stats(path, str(tmp_path / "s.json"))
        assert stats["missing_values"]["A"] == 2


# ──────────────────────────────────────────────
# _top_countries
# ──────────────────────────────────────────────

class TestTopCountries:
    def test_semicolon_separated_countries_counted_individually(self):
        df = pd.DataFrame({
            "Location Countries": [
                "United States; Canada",
                "United States",
                "Canada; Germany",
            ]
        })
        result = _top_countries(df)
        assert result["United States"] == 2
        assert result["Canada"] == 2
        assert result["Germany"] == 1

    def test_empty_cells_skipped(self):
        df = pd.DataFrame({
            "Location Countries": ["United States", None, "", "Canada"]
        })
        result = _top_countries(df)
        assert "United States" in result
        assert "Canada" in result
        assert "" not in result

    def test_returns_top_n_sorted_desc(self):
        df = pd.DataFrame({
            "Location Countries": ["United States"] * 10 + ["Canada"] * 3 + ["Germany"] * 1
        })
        result = _top_countries(df, n=2)
        keys = list(result.keys())
        assert keys[0] == "United States"
        assert keys[1] == "Canada"
        assert len(result) == 2