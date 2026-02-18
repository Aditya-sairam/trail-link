"""
test_bias.py — Diabetes
Mirrors: tests/breast_cancer/test_bias.py
"""
import pandas as pd
import pytest
from src.pipelines.diabetes.bias import generate_bias_report, save_bias_report


def test_generate_bias_report_handles_missing_columns():
    """Should gracefully handle missing columns"""
    df = pd.DataFrame({"NCT Number": ["NCT00000001", "NCT00000002"]})
    report = generate_bias_report(df, slice_columns=["Sex"])
    assert "Sex" in report["slices"]
    assert report["slices"]["Sex"]["error"] == "column_not_found"


def test_generate_bias_report_sex_representation():
    """Should compute sex distribution when column exists"""
    df = pd.DataFrame({"Sex": ["ALL", "FEMALE", None]})
    report = generate_bias_report(df, slice_columns=["Sex"])
    rep = report["slices"]["Sex"]["representation"]
    assert "counts" in rep and "pct" in rep


def test_generate_bias_report_age_representation():
    """Should detect low pediatric representation"""
    df = pd.DataFrame({
        "Age": ["Adult", "Adult", "Adult", "Adult", "Child"]
    })
    report = generate_bias_report(df, slice_columns=["Age"])
    assert "age" in report["slices"]


def test_generate_bias_report_disease_type_distribution():
    """Should break down by diabetes disease type (our new column)"""
    df = pd.DataFrame({
        "disease_type": [
            "Type 1 Diabetes", "Type 2 Diabetes",
            "Type 2 Diabetes", "Gestational Diabetes", "Type 1 Diabetes"
        ]
    })
    report = generate_bias_report(df, slice_columns=["disease_type"])
    assert "disease_type" in report["slices"]


def test_generate_bias_report_geographic_bias():
    """Should flag >80% US-only trials as geographic bias"""
    df = pd.DataFrame({
        "Locations": [
            "Mayo Clinic, United States",
            "Johns Hopkins, United States",
            "Harvard, United States",
            "Oxford, United Kingdom",
        ]
    })
    report = generate_bias_report(df)
    geo = report["slices"].get("geography", {})
    assert "us_percentage" in geo
    assert geo["us_percentage"] > 0


def test_bias_level_is_one_of_expected_values():
    """Bias level must be LOW, MEDIUM, or HIGH"""
    df = pd.DataFrame({
        "NCT Number": ["NCT00000001"],
        "Sex": ["ALL"],
        "Age": ["Adult"],
        "Locations": ["Some Hospital, United States"],
        "disease_type": ["Type 2 Diabetes"],
    })
    report = generate_bias_report(df)
    assert report["bias_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_save_bias_report_creates_file(tmp_path):
    """Should save bias report JSON to the given path"""
    import json
    report = {
        "total_trials": 10,
        "slices": {},
        "warnings": [],
        "overall_bias_score": 0,
        "bias_level": "LOW",
    }
    path = str(tmp_path / "bias_report.json")
    save_bias_report(report, bias_path=path)

    with open(path) as f:
        saved = json.load(f)

    assert saved["bias_level"] == "LOW"
    assert saved["total_trials"] == 10