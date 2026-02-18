"""
test_quality.py — Diabetes
Mirrors: tests/breast_cancer/test_quality.py
"""
import pandas as pd
import pytest
from src.pipelines.diabetes.quality import (
    clean_whitespace,
    remove_html_entities,
    remove_invalid_characters,
    normalize_medical_terminology,
    remove_duplicate_words,
    apply_all_text_cleaning,
    remove_duplicates,
    run_quality_checks,
    anomalies_found,
)


# ── Text cleaning unit tests ──────────────────────────────────────────────────

def test_clean_whitespace_strips_and_collapses():
    assert clean_whitespace("  diabetes   mellitus  ") == "diabetes mellitus"
    assert clean_whitespace("diabetes\t\tmellitus")    == "diabetes mellitus"


def test_clean_whitespace_returns_none_for_empty():
    assert clean_whitespace(None) is None
    assert clean_whitespace("   ")  is None


def test_remove_html_entities():
    assert remove_html_entities("diabetes &amp; obesity") == "diabetes & obesity"
    assert remove_html_entities("&lt;10%")                == "<10%"
    assert remove_html_entities("patient&#39;s trial")    == "patient's trial"


def test_remove_invalid_characters_strips_nullbytes():
    assert remove_invalid_characters("diabetes\x00 trial") == "diabetes trial"
    assert remove_invalid_characters("test\ufffdreplaced") == "testreplaced"


def test_normalize_medical_terminology_type1():
    assert normalize_medical_terminology("T1D patients")      == "Type 1 Diabetes Mellitus patients"
    assert normalize_medical_terminology("T1DM study")        == "Type 1 Diabetes Mellitus study"
    assert normalize_medical_terminology("juvenile diabetes")  == "Type 1 Diabetes Mellitus"
    assert normalize_medical_terminology("IDDM management")   == "Type 1 Diabetes Mellitus management"


def test_normalize_medical_terminology_type2():
    assert normalize_medical_terminology("T2D cohort")  == "Type 2 Diabetes Mellitus cohort"
    assert normalize_medical_terminology("T2DM trial")  == "Type 2 Diabetes Mellitus trial"
    assert normalize_medical_terminology("NIDDM study") == "Type 2 Diabetes Mellitus study"


def test_normalize_medical_terminology_gestational():
    assert normalize_medical_terminology("GDM screening") == "Gestational Diabetes Mellitus screening"


def test_normalize_medical_terminology_case_insensitive():
    assert normalize_medical_terminology("t1dm") == "Type 1 Diabetes Mellitus"
    assert normalize_medical_terminology("T2DM") == "Type 2 Diabetes Mellitus"


def test_remove_duplicate_words():
    assert remove_duplicate_words("diabetes diabetes mellitus") == "diabetes mellitus"
    assert remove_duplicate_words("type type 2 diabetes")       == "type 2 diabetes"


def test_remove_duplicate_words_no_change_when_clean():
    assert remove_duplicate_words("Type 2 Diabetes Mellitus") == "Type 2 Diabetes Mellitus"


def test_apply_all_text_cleaning_full_pipeline():
    """Full cleaning pipeline on a messy string"""
    messy = "  T1DM &amp; obesity  diabetes diabetes  "
    result = apply_all_text_cleaning(messy)
    assert result == "Type 1 Diabetes Mellitus & obesity diabetes"


def test_apply_all_text_cleaning_returns_none_for_null():
    assert apply_all_text_cleaning(None) is None
    assert apply_all_text_cleaning("")   is None


# ── Deduplication tests ───────────────────────────────────────────────────────

def test_remove_duplicates_keeps_most_complete():
    """Should keep the row with more non-null values"""
    df = pd.DataFrame({
        "NCT Number":  ["NCT00000001", "NCT00000001"],
        "Study Title": ["Diabetes Study", "Diabetes Study"],
        "Sponsor":     [None, "Harvard Medical School"],  # second row more complete
    })
    result = remove_duplicates(df)
    assert len(result) == 1
    assert result.iloc[0]["Sponsor"] == "Harvard Medical School"


def test_remove_duplicates_unique_rows_unchanged():
    df = pd.DataFrame({
        "NCT Number": ["NCT00000001", "NCT00000002"],
        "Study Title": ["Study A", "Study B"],
    })
    result = remove_duplicates(df)
    assert len(result) == 2


# ── Stats + anomaly tests ─────────────────────────────────────────────────────

def test_run_quality_checks_has_expected_keys(tmp_path):
    """run_quality_checks should return dict with required keys"""
    enriched = tmp_path / "enriched.csv"
    df = pd.DataFrame({
        "NCT Number":  ["NCT00000001", "NCT00000002"],
        "Study Title": ["Diabetes Study A", "Diabetes Study B"],
        "Conditions":  ["Type 1 Diabetes", "Type 2 Diabetes"],
    })
    df.to_csv(enriched, index=False)

    stats = run_quality_checks(enriched_file_path=str(enriched))

    assert "total_rows"          in stats
    assert "issues"              in stats
    assert "rows_after_cleaning" in stats
    assert stats["total_rows"]   == 2


def test_anomalies_found_returns_true_when_quality_ok():
    """Should return True (continue pipeline) when nulls are acceptable"""
    stats = {"total_rows": 100, "issues": {"null_values": 5}}
    assert anomalies_found(stats) is True


def test_anomalies_found_returns_false_when_too_many_nulls():
    """Should return False (short-circuit) when >80% values are null"""
    stats = {"total_rows": 100, "issues": {"null_values": 85}}
    assert anomalies_found(stats) is False


def test_anomalies_found_handles_zero_rows():
    """Should not crash on empty dataset"""
    stats = {"total_rows": 0, "issues": {"null_values": 0}}
    result = anomalies_found(stats)
    assert isinstance(result, bool)