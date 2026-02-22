import os
import sys
import pandas as pd
import pytest

# Ensure the pipeline src is importable from tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bias import generate_bias_report
from conditions.registry import REGISTRY


@pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
def test_generate_bias_report_handles_missing_columns(dkey):
    # Create a minimal dataframe without the Sex column
    df = pd.DataFrame({"NCT Number": ["N1", "N2"], "disease": [REGISTRY[dkey]["disease"]] * 2})
    report = generate_bias_report(df, slice_columns=["Sex"])

    # When Sex column is missing, the implementation does not add a Sex slice
    assert "Sex" not in report["slices"]


@pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
def test_generate_bias_report_representation_present_when_col_exists(dkey):
    # Sex column present should produce counts and pct
    df = pd.DataFrame({"Sex": ["All", "Female", None], "disease": [REGISTRY[dkey]["disease"]] * 3})
    report = generate_bias_report(df, slice_columns=["Sex"]) 

    assert "Sex" in report["slices"]
    rep = report["slices"]["Sex"]
    assert "counts" in rep and "pct" in rep
