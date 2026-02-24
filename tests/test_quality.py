import os
import sys
import json
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quality import (
    clean_whitespace,
    remove_html_entities,
    remove_invalid_characters,
    normalize_medical_terminology,
    remove_duplicate_words,
    apply_all_text_cleaning,
    generate_stats,
    detect_anomalies,
    anomalies_found,
    run_quality_checks,
)


# ──────────────────────────────────────────────
# clean_whitespace
# ──────────────────────────────────────────────

class TestCleanWhitespace:
    def test_strips_leading_trailing(self):
        assert clean_whitespace("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        assert clean_whitespace("hello   world") == "hello world"

    def test_collapses_tabs_and_newlines(self):
        assert clean_whitespace("hello\t\nworld") == "hello world"

    def test_none_returns_none(self):
        assert clean_whitespace(None) is None

    def test_nan_returns_none(self):
        assert clean_whitespace(float("nan")) is None

    def test_empty_string_returns_none(self):
        assert clean_whitespace("   ") is None


# ──────────────────────────────────────────────
# remove_html_entities
# ──────────────────────────────────────────────

class TestRemoveHtmlEntities:
    def test_amp(self):
        assert remove_html_entities("cats &amp; dogs") == "cats & dogs"

    def test_lt_gt(self):
        assert remove_html_entities("a &lt; b &gt; c") == "a < b > c"

    def test_quote_entities(self):
        assert remove_html_entities("say &quot;hi&quot;") == 'say "hi"'
        assert remove_html_entities("it&#39;s") == "it's"

    def test_nbsp(self):
        assert remove_html_entities("hello&nbsp;world") == "hello world"

    def test_none_returns_none(self):
        assert remove_html_entities(None) is None

    def test_no_entities_unchanged(self):
        assert remove_html_entities("plain text") == "plain text"


# ──────────────────────────────────────────────
# remove_invalid_characters
# ──────────────────────────────────────────────

class TestRemoveInvalidCharacters:
    def test_removes_null_byte(self):
        assert "\x00" not in remove_invalid_characters("hel\x00lo")

    def test_removes_replacement_char(self):
        assert "\ufffd" not in remove_invalid_characters("hel\ufffdlo")

    def test_removes_control_characters(self):
        result = remove_invalid_characters("hello\x01\x1fworld")
        assert "\x01" not in result
        assert "\x1f" not in result

    def test_none_returns_none(self):
        assert remove_invalid_characters(None) is None

    def test_clean_string_unchanged(self):
        assert remove_invalid_characters("normal text") == "normal text"


# ──────────────────────────────────────────────
# normalize_medical_terminology
# ──────────────────────────────────────────────

class TestNormalizeMedicalTerminology:
    @pytest.mark.parametrize("abbr,expected", [
        ("T1DM", "Type 1 Diabetes Mellitus"),
        ("t1dm", "Type 1 Diabetes Mellitus"),
        ("T1D", "Type 1 Diabetes Mellitus"),
        ("IDDM", "Type 1 Diabetes Mellitus"),
        ("T2DM", "Type 2 Diabetes Mellitus"),
        ("T2D", "Type 2 Diabetes Mellitus"),
        ("NIDDM", "Type 2 Diabetes Mellitus"),
        ("GDM", "Gestational Diabetes Mellitus"),
    ])
    def test_abbreviation_expanded(self, abbr, expected):
        result = normalize_medical_terminology(f"Patient with {abbr} enrolled")
        assert expected in result

    def test_none_returns_none(self):
        assert normalize_medical_terminology(None) is None

    def test_unrelated_text_unchanged(self):
        text = "No abbreviations here"
        assert normalize_medical_terminology(text) == text


# ──────────────────────────────────────────────
# remove_duplicate_words
# ──────────────────────────────────────────────

class TestRemoveDuplicateWords:
    def test_removes_consecutive_duplicates(self):
        assert remove_duplicate_words("the the cat sat") == "the cat sat"

    def test_case_insensitive_dedup(self):
        assert remove_duplicate_words("The the cat") == "The cat"

    def test_non_consecutive_duplicates_kept(self):
        result = remove_duplicate_words("cat sat cat")
        assert result == "cat sat cat"

    def test_single_word_unchanged(self):
        assert remove_duplicate_words("hello") == "hello"

    def test_none_returns_none(self):
        assert remove_duplicate_words(None) is None


# ──────────────────────────────────────────────
# apply_all_text_cleaning
# ──────────────────────────────────────────────

class TestApplyAllTextCleaning:
    def test_none_returns_none(self):
        assert apply_all_text_cleaning(None) is None

    def test_chains_all_cleaning(self):
        dirty = "  T1DM &amp; the the patient\x00  "
        result = apply_all_text_cleaning(dirty)
        assert "Type 1 Diabetes Mellitus" in result
        assert "&amp;" not in result
        assert "\x00" not in result
        assert "the the" not in result
        assert not result.startswith(" ")

    def test_empty_like_string_returns_none(self):
        assert apply_all_text_cleaning("   ") is None


# ──────────────────────────────────────────────
# generate_stats
# ──────────────────────────────────────────────

class TestGenerateStats:
    def test_basic_keys_present(self):
        df = pd.DataFrame({"NCT Number": ["N1", "N2"], "A": [1, None]})
        stats = generate_stats(df)
        assert stats["rows"] == 2
        assert stats["cols"] == 2
        assert "missing_by_col" in stats
        assert stats["unique_nct_numbers"] == 2

    def test_missing_by_col_only_includes_nulls(self):
        df = pd.DataFrame({"NCT Number": ["N1", "N2"], "B": [1, 2]})
        stats = generate_stats(df)
        assert "B" not in stats["missing_by_col"]

    def test_enrollment_stats_computed(self):
        df = pd.DataFrame({
            "NCT Number": ["N1", "N2", "N3"],
            "Enrollment": [10, 20, 30],
        })
        stats = generate_stats(df)
        assert stats["enrollment"]["mean"] == 20.0
        assert stats["enrollment"]["median"] == 20.0
        assert stats["enrollment"]["min"] == 10
        assert stats["enrollment"]["max"] == 30

    def test_disease_type_distribution(self):
        df = pd.DataFrame({
            "disease": ["diabetes", "diabetes"],
            "disease_type": ["Type 1", "Type 2"],
        })
        stats = generate_stats(df)
        assert stats["disease"] == "diabetes"
        assert "Type 1" in stats["disease_type_distribution"]

    def test_sex_distribution(self):
        df = pd.DataFrame({"Sex": ["ALL", "FEMALE", "ALL"]})
        stats = generate_stats(df)
        assert stats["sex_distribution"]["ALL"] == 2
        assert stats["sex_distribution"]["FEMALE"] == 1

    def test_recruitment_status_distribution(self):
        df = pd.DataFrame({"Recruitment Status": ["RECRUITING", "RECRUITING", "COMPLETED"]})
        stats = generate_stats(df)
        assert stats["recruitment_status"]["RECRUITING"] == 2


# ──────────────────────────────────────────────
# detect_anomalies
# ──────────────────────────────────────────────

class TestDetectAnomalies:
    def test_flags_duplicate_nct(self):
        df = pd.DataFrame({"NCT Number": ["N1", "N1"]})
        anom = detect_anomalies(df)
        assert anom["duplicate_nct_found"] is True

    def test_no_duplicates_not_flagged(self):
        df = pd.DataFrame({"NCT Number": ["N1", "N2"]})
        anom = detect_anomalies(df)
        assert anom["duplicate_nct_found"] is False

    def test_missing_required_column_flagged(self):
        df = pd.DataFrame({"NCT Number": ["N1"]})  # missing disease, disease_type etc.
        anom = detect_anomalies(df)
        assert "disease" in anom["missing_required_columns"]
        assert "disease_type" in anom["missing_required_columns"]

    def test_high_missing_column_flagged(self):
        df = pd.DataFrame({
            "NCT Number": ["N1"] * 10,
            "disease": ["diabetes"] * 10,
            "disease_type": ["T1"] * 10,
            "Study Title": [None] * 10,      # 100% missing
            "Conditions": ["Diabetes"] * 10,
        })
        anom = detect_anomalies(df)
        flagged_cols = [x["col"] for x in anom["high_missing_columns"]]
        assert "Study Title" in flagged_cols

    def test_phase_column_ignored_even_if_high_missing(self):
        df = pd.DataFrame({
            "NCT Number": ["N1"] * 10,
            "disease": ["diabetes"] * 10,
            "disease_type": ["T1"] * 10,
            "Study Title": ["T"] * 10,
            "Conditions": ["D"] * 10,
            "Phase": [None] * 10,   # 100% missing but in IGNORE set
        })
        anom = detect_anomalies(df)
        flagged_cols = [x["col"] for x in anom["high_missing_columns"]]
        assert "Phase" not in flagged_cols

    def test_invalid_nct_format_counted(self):
        df = pd.DataFrame({"NCT Number": ["NCT12345678", "ABC123", "BADFORMAT"]})
        anom = detect_anomalies(df)
        assert anom["invalid_nct_format"] == 2

    def test_valid_nct_format_not_counted(self):
        df = pd.DataFrame({"NCT Number": ["NCT12345678", "NCT00000001"]})
        anom = detect_anomalies(df)
        assert anom["invalid_nct_format"] == 0

    def test_enrollment_outliers_detected(self):
        # Normal values + extreme outlier
        enrollment = [100, 110, 105, 95, 100, 98, 102, 99, 101, 100000]
        df = pd.DataFrame({"Enrollment": enrollment})
        anom = detect_anomalies(df)
        assert len(anom["enrollment_outliers"]) > 0
        assert 100000 in anom["enrollment_outliers"]

    def test_no_enrollment_outliers_when_uniform(self):
        df = pd.DataFrame({"Enrollment": [100, 100, 100, 100, 100]})
        anom = detect_anomalies(df)
        assert len(anom["enrollment_outliers"]) == 0


# ──────────────────────────────────────────────
# anomalies_found
# ──────────────────────────────────────────────

class TestAnomaliesFound:
    def test_returns_false_when_clean(self):
        anom = {
            "missing_required_columns": [],
            "duplicate_nct_found": False,
            "high_missing_columns": [],
            "enrollment_outliers": [],
            "invalid_nct_format": 0,
        }
        assert anomalies_found(anom) is False

    def test_returns_true_when_duplicates(self):
        anom = {
            "missing_required_columns": [],
            "duplicate_nct_found": True,
            "high_missing_columns": [],
        }
        assert anomalies_found(anom) is True

    def test_returns_true_when_missing_columns(self):
        anom = {
            "missing_required_columns": ["disease"],
            "duplicate_nct_found": False,
            "high_missing_columns": [],
        }
        assert anomalies_found(anom) is True

    def test_returns_true_when_high_missing(self):
        anom = {
            "missing_required_columns": [],
            "duplicate_nct_found": False,
            "high_missing_columns": [{"col": "Study Title", "missing_pct": 80.0}],
        }
        assert anomalies_found(anom) is True


# ──────────────────────────────────────────────
# run_quality_checks (integration)
# ──────────────────────────────────────────────

class TestRunQualityChecks:
    def _make_valid_df(self):
        return pd.DataFrame({
            "NCT Number": ["NCT12345678", "NCT00000001"],
            "Study Title": ["Trial A", "Trial B"],
            "Brief Summary": ["Summary A &amp; more", "Summary B"],
            "Conditions": ["T1DM", "T2DM"],
            "Interventions": ["Drug", "Placebo"],
            "Sponsor": ["NIH", "NIH"],
            "disease": ["diabetes", "diabetes"],
            "disease_type": ["Type 1", "Type 2"],
        })

    def test_stats_json_created(self, tmp_path):
        df = self._make_valid_df()
        csv_path = str(tmp_path / "enriched.csv")
        stats_path = str(tmp_path / "stats.json")
        anomalies_path = str(tmp_path / "anomalies.json")
        df.to_csv(csv_path, index=False)
        run_quality_checks(csv_path, stats_path, anomalies_path)
        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["rows"] == 2

    def test_anomalies_json_created(self, tmp_path):
        df = self._make_valid_df()
        csv_path = str(tmp_path / "enriched.csv")
        stats_path = str(tmp_path / "stats.json")
        anomalies_path = str(tmp_path / "anomalies.json")
        df.to_csv(csv_path, index=False)
        run_quality_checks(csv_path, stats_path, anomalies_path)
        assert os.path.exists(anomalies_path)

    def test_text_cleaning_applied_to_csv(self, tmp_path):
        df = self._make_valid_df()
        csv_path = str(tmp_path / "enriched.csv")
        df.to_csv(csv_path, index=False)
        run_quality_checks(csv_path, str(tmp_path / "s.json"), str(tmp_path / "a.json"))
        cleaned = pd.read_csv(csv_path)
        # &amp; should be replaced with &
        assert "&amp;" not in cleaned["Brief Summary"].fillna("").to_string()
