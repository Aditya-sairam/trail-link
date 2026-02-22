import os
import sys
import pandas as pd
import pytest

# Make pipeline src importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ingest import enrich_trials_csv
from conditions.registry import REGISTRY


@pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
def test_enrich_trials_csv_dedupes_and_adds_columns(tmp_path, dkey):
    raw_path = tmp_path / "raw.csv"
    enriched_path = tmp_path / "enriched.csv"

    # Create a small raw DataFrame with duplicates and Conditions
    df = pd.DataFrame(
        {
            "NCT Number": ["N1", "N1", "N2"],
            "Conditions": ["Breast Cancer", "Breast Cancer", "Diabetes"],
        }
    )
    df.to_csv(raw_path, index=False)

    disease = REGISTRY[dkey]["disease"]
    classifier = REGISTRY[dkey]["classifier"]

    # Run enrichment which should dedupe and add disease / disease_type
    out_df = enrich_trials_csv(str(raw_path), str(enriched_path), disease, classifier)

    out = pd.read_csv(enriched_path)

    # Deduped by NCT Number => N1 + N2
    assert len(out) == 2
    assert set(out["NCT Number"].tolist()) == {"N1", "N2"}

    # Added columns
    assert "disease" in out.columns
    assert "disease_type" in out.columns
    assert (out["disease"] == disease).all()
