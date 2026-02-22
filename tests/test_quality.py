import os
import sys
import pandas as pd
import pytest

# Make pipeline src importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quality import generate_stats, detect_anomalies, anomalies_found


def test_generate_stats_has_expected_keys():
    df = pd.DataFrame({"NCT Number": ["N1", "N2"], "A": [1, None]})
    stats = generate_stats(df)
    assert stats["rows"] == 2
    assert "missing_by_col" in stats
    assert stats["unique_nct_numbers"] == 2


def test_detect_anomalies_flags_duplicates():
    df = pd.DataFrame({"NCT Number": ["N1", "N1"]})
    anom = detect_anomalies(df)
    assert anom["duplicate_nct_found"] is True
    assert anomalies_found(anom) is True
