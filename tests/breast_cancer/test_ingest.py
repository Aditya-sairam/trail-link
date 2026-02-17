import pandas as pd

from src.pipelines.breast_cancer.ingest import (
    sanitize_filename,
    extract_cancer_type,
    enrich_trials_csv,
)


def test_sanitize_filename_basic():
    assert sanitize_filename("Triple Negative Breast Cancer") == "triple_negative_breast_cancer"


def test_extract_cancer_type_picks_breast_cancer_subtype():
    conditions = "Lung Cancer|Triple Negative Breast Cancer|Diabetes"
    out = extract_cancer_type(conditions)
    assert out == "triple_negative_breast_cancer"


def test_enrich_trials_csv_dedupes_and_adds_columns(tmp_path):
    raw_path = tmp_path / "raw.csv"
    enriched_path = tmp_path / "enriched.csv"

    df = pd.DataFrame(
        {
            "NCTId": ["N1", "N1", "N2"],
            "Conditions": ["Breast Cancer", "Breast Cancer", "Breast Cancer"],
        }
    )
    df.to_csv(raw_path, index=False)

    enrich_trials_csv(str(raw_path), str(enriched_path))

    out = pd.read_csv(enriched_path)

    # Deduped by NCTId => N1 + N2
    assert len(out) == 2
    assert set(out["NCTId"].tolist()) == {"N1", "N2"}

    # Added columns
    assert "disease" in out.columns
    assert "cancer_type" in out.columns
    assert (out["disease"] == "breast_cancer").all()
