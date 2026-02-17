import pandas as pd
from src.pipelines.breast_cancer.bias import generate_bias_report

def test_generate_bias_report_handles_missing_columns():
    df = pd.DataFrame({"NCTId": ["N1", "N2"]})
    report = generate_bias_report(df, slice_columns=["Sex"])
    assert "Sex" in report["slices"]
    assert report["slices"]["Sex"]["error"] == "column_not_found"


def test_generate_bias_report_representation_present_when_col_exists():
    df = pd.DataFrame({"Sex": ["All", "Female", None]})
    report = generate_bias_report(df, slice_columns=["Sex"])
    rep = report["slices"]["Sex"]["representation"]
    assert "counts" in rep and "pct" in rep
