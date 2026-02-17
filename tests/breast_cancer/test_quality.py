import pandas as pd

from src.pipelines.breast_cancer.quality import generate_stats, detect_anomalies, anomalies_found


def test_generate_stats_has_expected_keys():
    df = pd.DataFrame({"NCTId": ["N1", "N2"], "A": [1, None]})
    stats = generate_stats(df)
    assert stats["rows"] == 2
    assert "missing_by_col" in stats
    assert stats["unique_nctid"] == 2


def test_detect_anomalies_flags_duplicates():
    df = pd.DataFrame({"NCTId": ["N1", "N1"]})
    anom = detect_anomalies(df)
    assert anom["duplicate_nctid_found"] is True
    assert anomalies_found(anom) is True
