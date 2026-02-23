# tests/test_schema.py

import os
import sys
import json
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from schema import run_schema_checkpoint, RAW_REQUIRED_DEFAULT, PROCESSED_REQUIRED_DEFAULT


def test_schema_checkpoint_creates_baseline_and_passes_for_good_raw(tmp_path):
    df = pd.DataFrame(
        {
            "NCT Number": ["N1", "N2"],
            "Study Title": ["T1", "T2"],
            "Recruitment Status": ["RECRUITING", "COMPLETED"],
            "Conditions": ["Diabetes", "Diabetes"],
            "Interventions": ["Drug", "Drug"],
            "Sponsor": ["NIH", "NIH"],
            "Enrollment": [10, 25],
            "Sex": ["ALL", "FEMALE"],
            "Location Countries": ["US", "US"],
            "Location Cities": ["Boston", "NYC"],
        }
    )

    csv_path = tmp_path / "raw.csv"
    baseline_path = tmp_path / "raw_schema.json"
    report_path = tmp_path / "raw_report.json"

    df.to_csv(csv_path, index=False)

    report = run_schema_checkpoint(
        csv_path=str(csv_path),
        baseline_schema_path=str(baseline_path),
        report_path=str(report_path),
        required_columns=RAW_REQUIRED_DEFAULT,
        mode="enforce",
        allow_new_columns=True,
    )

    assert report["passed"] is True
    assert baseline_path.exists()
    assert report_path.exists()

    baseline = json.loads(baseline_path.read_text())
    assert "columns" in baseline
    assert "required_columns" in baseline


def test_schema_checkpoint_fails_when_required_missing(tmp_path):
    df = pd.DataFrame(
        {
            "NCT Number": ["N1"],
            # Study Title missing
            "Recruitment Status": ["RECRUITING"],
            "Conditions": ["Diabetes"],
            "Interventions": ["Drug"],
            "Sponsor": ["NIH"],
            "Enrollment": [10],
            "Sex": ["ALL"],
            "Location Countries": ["US"],
            "Location Cities": ["Boston"],
        }
    )

    csv_path = tmp_path / "bad.csv"
    baseline_path = tmp_path / "raw_schema.json"
    report_path = tmp_path / "bad_report.json"

    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        run_schema_checkpoint(
            csv_path=str(csv_path),
            baseline_schema_path=str(baseline_path),
            report_path=str(report_path),
            required_columns=RAW_REQUIRED_DEFAULT,
            mode="enforce",
            allow_new_columns=True,
        )


def test_processed_required_includes_disease_columns():
    assert "disease" in PROCESSED_REQUIRED_DEFAULT
    assert "disease_type" in PROCESSED_REQUIRED_DEFAULT