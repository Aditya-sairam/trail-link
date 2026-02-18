"""
test_ingest.py — Diabetes
Mirrors: tests/breast_cancer/test_ingest.py
"""
import pandas as pd
import pytest
from src.pipelines.diabetes.ingest import (
    enrich_trials_csv,
    classify_disease_type,
)


def test_classify_disease_type_type1():
    """Should classify Type 1 diabetes variations correctly"""
    assert classify_disease_type("Type 1 Diabetes Mellitus") == "Type 1 Diabetes"
    assert classify_disease_type("T1DM patients")            == "Type 1 Diabetes"
    assert classify_disease_type("juvenile diabetes")        == "Type 1 Diabetes"
    assert classify_disease_type("IDDM")                     == "Type 1 Diabetes"


def test_classify_disease_type_type2():
    """Should classify Type 2 diabetes variations correctly"""
    assert classify_disease_type("Type 2 Diabetes")         == "Type 2 Diabetes"
    assert classify_disease_type("t2dm management")         == "Type 2 Diabetes"
    assert classify_disease_type("Non-insulin dependent")   == "Type 2 Diabetes"


def test_classify_disease_type_gestational():
    """Should classify gestational diabetes correctly"""
    assert classify_disease_type("Gestational Diabetes Mellitus") == "Gestational Diabetes"
    assert classify_disease_type("GDM screening")                 == "Gestational Diabetes"


def test_classify_disease_type_prediabetes():
    """Should classify prediabetes correctly"""
    assert classify_disease_type("prediabetes prevention") == "Pre-Diabetes"
    assert classify_disease_type("pre-diabetes study")     == "Pre-Diabetes"


def test_classify_disease_type_unknown():
    """Should return general bucket for unrecognised conditions"""
    assert classify_disease_type("some other condition") == "Diabetes (General)"
    assert classify_disease_type(None)                   == "Unknown"


def test_enrich_trials_csv_dedupes_and_adds_columns(tmp_path):
    """Should deduplicate by NCT Number and add disease_type + data_source"""
    raw_path      = tmp_path / "raw.csv"
    enriched_path = tmp_path / "enriched.csv"

    df = pd.DataFrame({
        "NCT Number": ["NCT00000001", "NCT00000001", "NCT00000002"],
        "Conditions": [
            "Type 1 Diabetes Mellitus",
            "Type 1 Diabetes Mellitus",
            "Type 2 Diabetes",
        ],
    })
    df.to_csv(raw_path, index=False)

    enrich_trials_csv(str(raw_path), str(enriched_path))

    out = pd.read_csv(enriched_path)

    # Deduped: NCT00000001 + NCT00000002
    assert len(out) == 2
    assert set(out["NCT Number"].tolist()) == {"NCT00000001", "NCT00000002"}

    # New columns added
    assert "disease_type"  in out.columns
    assert "data_source"   in out.columns

    # disease_type correctly classified
    t1_row = out[out["NCT Number"] == "NCT00000001"].iloc[0]
    assert t1_row["disease_type"] == "Type 1 Diabetes"

    t2_row = out[out["NCT Number"] == "NCT00000002"].iloc[0]
    assert t2_row["disease_type"] == "Type 2 Diabetes"

    # data_source always ClinicalTrials.gov
    assert (out["data_source"] == "ClinicalTrials.gov").all()


def test_enrich_trials_csv_handles_empty_file(tmp_path):
    """Should not crash on an empty CSV"""
    raw_path      = tmp_path / "raw.csv"
    enriched_path = tmp_path / "enriched.csv"

    pd.DataFrame(columns=["NCT Number", "Conditions"]).to_csv(raw_path, index=False)
    enrich_trials_csv(str(raw_path), str(enriched_path))

    out = pd.read_csv(enriched_path)
    assert len(out) == 0