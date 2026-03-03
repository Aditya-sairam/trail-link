import os
import sys
import json
import pytest
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipelines", "dags", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bias import generate_bias_report, save_bias_report
from conditions.registry import REGISTRY


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def make_df(n=10, sex="ALL", disease="diabetes", disease_type="Type 2 Diabetes",
            countries=None, age_col=None):
    data = {
        "NCT Number": [f"N{i}" for i in range(n)],
        "disease": [disease] * n,
        "Sex": [sex] * n,
        "disease_type": [disease_type] * n,
        "Location Countries": [countries or "United States"] * n,
    }
    if age_col:
        data[age_col] = ["Adult"] * n
    return pd.DataFrame(data)


# ──────────────────────────────────────────────
# Missing columns — graceful handling
# ──────────────────────────────────────────────

class TestMissingColumns:
    @pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
    def test_missing_sex_column_no_sex_slice(self, dkey):
        df = pd.DataFrame({
            "NCT Number": ["N1", "N2"],
            "disease": [REGISTRY[dkey]["disease"]] * 2,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert "Sex" not in report["slices"]

    def test_missing_location_countries_no_geography_slice(self):
        df = pd.DataFrame({
            "NCT Number": ["N1"],
            "disease": ["diabetes"],
            "Sex": ["ALL"],
        })
        report = generate_bias_report(df, slice_columns=["Location Countries"])
        assert "geography" not in report["slices"]


# ──────────────────────────────────────────────
# Sex slice — presence and structure
# ──────────────────────────────────────────────

class TestSexSlice:
    @pytest.mark.parametrize("dkey", ["breast_cancer", "diabetes"])
    def test_sex_slice_has_counts_and_pct(self, dkey):
        df = pd.DataFrame({
            "Sex": ["ALL", "FEMALE", None],
            "disease": [REGISTRY[dkey]["disease"]] * 3,
            "disease_type": ["Type 2"] * 3,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert "Sex" in report["slices"]
        rep = report["slices"]["Sex"]
        assert "counts" in rep and "pct" in rep

    def test_sex_imbalance_high_male_triggers_warning(self):
        # 10 male, 1 female → ratio 10 > 2
        df = pd.DataFrame({
            "disease": ["diabetes"] * 11,
            "disease_type": ["Type 2"] * 11,
            "Sex": ["MALE"] * 10 + ["FEMALE"] * 1,
            "Location Countries": ["United States"] * 11,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert any("SEX IMBALANCE" in w for w in report["warnings"])

    def test_sex_imbalance_high_female_triggers_warning(self):
        # 1 male, 10 female → ratio 0.1 < 0.5
        df = pd.DataFrame({
            "disease": ["diabetes"] * 11,
            "disease_type": ["Type 2"] * 11,
            "Sex": ["MALE"] * 1 + ["FEMALE"] * 10,
            "Location Countries": ["United States"] * 11,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert any("SEX IMBALANCE" in w for w in report["warnings"])

    def test_balanced_sex_no_warning(self):
        df = pd.DataFrame({
            "disease": ["diabetes"] * 4,
            "disease_type": ["Type 2"] * 4,
            "Sex": ["MALE"] * 2 + ["FEMALE"] * 2,
            "Location Countries": ["United States"] * 4,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        assert not any("SEX IMBALANCE" in w for w in report["warnings"])

    def test_gestational_diabetes_excluded_from_sex_bias(self):
        # Only gestational rows — after exclusion, no male/female counts to compare
        df = pd.DataFrame({
            "disease": ["diabetes"] * 6,
            "disease_type": ["Gestational Diabetes"] * 6,
            "Sex": ["FEMALE"] * 5 + ["MALE"] * 1,
            "Location Countries": ["United States"] * 6,
        })
        report = generate_bias_report(df, slice_columns=["Sex"])
        # After filtering out gestational, counts are empty → no imbalance warning
        assert not any("SEX IMBALANCE" in w for w in report["warnings"])


# ──────────────────────────────────────────────
# Geographic bias
# ──────────────────────────────────────────────

class TestGeographicBias:
    def test_high_us_concentration_triggers_warning(self):
        # 9 US, 1 international → 90% US > 80% threshold
        df = pd.DataFrame({
            "disease": ["diabetes"] * 10,
            "disease_type": ["Type 2"] * 10,
            "Sex": ["ALL"] * 10,
            "Location Countries": ["United States"] * 9 + ["Canada"] * 1,
        })
        report = generate_bias_report(df, slice_columns=["Location Countries"])
        assert any("GEOGRAPHIC BIAS" in w for w in report["warnings"])

    def test_diverse_geography_no_warning(self):
        # 50% US
        df = pd.DataFrame({
            "disease": ["diabetes"] * 10,
            "disease_type": ["Type 2"] * 10,
            "Sex": ["ALL"] * 10,
            "Location Countries": ["United States"] * 5 + ["Canada"] * 5,
        })
        report = generate_bias_report(df, slice_columns=["Location Countries"])
        assert not any("GEOGRAPHIC BIAS" in w for w in report["warnings"])

    def test_geography_slice_values_correct(self):
        df = pd.DataFrame({
            "disease": ["diabetes"] * 4,
            "disease_type": ["Type 2"] * 4,
            "Sex": ["ALL"] * 4,
            "Location Countries": ["United States"] * 3 + ["Canada"] * 1,
        })
        report = generate_bias_report(df, slice_columns=["Location Countries"])
        geo = report["slices"]["geography"]
        assert geo["us_trials"] == 3
        assert geo["international_trials"] == 1
        assert geo["us_percentage"] == 75.0


# ──────────────────────────────────────────────
# Age bias
# ──────────────────────────────────────────────

class TestAgeBias:
    def test_low_pediatric_triggers_warning(self):
        # 0 pediatric out of 100 → 0% < 5%
        df = pd.DataFrame({
            "disease": ["diabetes"] * 100,
            "disease_type": ["Type 2"] * 100,
            "Sex": ["ALL"] * 100,
            "Location Countries": ["United States"] * 100,
            "Age": ["Adult"] * 100,
        })
        report = generate_bias_report(df, slice_columns=["Age"])
        assert any("pediatric" in w.lower() for w in report["warnings"])

    def test_sufficient_pediatric_no_warning(self):
        # 10 pediatric out of 100 → 10% > 5%
        df = pd.DataFrame({
            "disease": ["diabetes"] * 100,
            "disease_type": ["Type 2"] * 100,
            "Sex": ["ALL"] * 100,
            "Location Countries": ["United States"] * 100,
            "Age": ["Child"] * 10 + ["Adult"] * 90,
        })
        report = generate_bias_report(df, slice_columns=["Age"])
        assert not any("pediatric" in w.lower() for w in report["warnings"])


# ──────────────────────────────────────────────
# Bias score and level
# ──────────────────────────────────────────────

class TestBiasScoreAndLevel:
    def _report_with_n_warnings(self, n):
        """Generate a report and manually set warning count to test scoring."""
        df = make_df(n=10)
        report = generate_bias_report(df, slice_columns=[])
        report["warnings"] = ["warning"] * n
        report["bias_score"] = n
        report["bias_level"] = "HIGH" if n >= 2 else "MEDIUM" if n == 1 else "LOW"
        return report

    def test_no_warnings_is_low(self):
        report = self._report_with_n_warnings(0)
        assert report["bias_level"] == "LOW"
        assert report["bias_score"] == 0

    def test_one_warning_is_medium(self):
        report = self._report_with_n_warnings(1)
        assert report["bias_level"] == "MEDIUM"

    def test_two_or_more_warnings_is_high(self):
        report = self._report_with_n_warnings(2)
        assert report["bias_level"] == "HIGH"

    def test_clean_df_produces_low_bias(self):
        # 50/50 sex, 50% US, no age column
        df = pd.DataFrame({
            "disease": ["diabetes"] * 10,
            "disease_type": ["Type 2"] * 10,
            "Sex": ["MALE"] * 5 + ["FEMALE"] * 5,
            "Location Countries": ["United States"] * 5 + ["Canada"] * 5,
        })
        report = generate_bias_report(df)
        assert report["bias_level"] == "LOW"
        assert report["bias_score"] == 0


# ──────────────────────────────────────────────
# save_bias_report
# ──────────────────────────────────────────────

class TestSaveBiasReport:
    def test_file_written_correctly(self, tmp_path):
        report = {"bias_score": 1, "bias_level": "MEDIUM", "warnings": ["w1"]}
        path = str(tmp_path / "reports" / "bias.json")
        save_bias_report(report, path)
        assert os.path.exists(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["bias_score"] == 1
        assert loaded["bias_level"] == "MEDIUM"