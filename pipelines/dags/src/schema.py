# pipelines/dags/src/schema.py

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


RAW_REQUIRED_DEFAULT = [
    "NCT Number",
    "Study Title",
    "Recruitment Status",
    "Conditions",
    "Interventions",
    "Sponsor",
    "Enrollment",
    "Sex",
    "Location Countries",
    "Location Cities",
]

PROCESSED_REQUIRED_DEFAULT = RAW_REQUIRED_DEFAULT + [
    "disease",
    "disease_type",
]

DEFAULT_CATEGORICAL_LIMITS = {
    "Sex": {"allowed": ["ALL", "FEMALE", "MALE"]},
}

DEFAULT_NUMERIC_RANGES = {}


def _dtype_family(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "string"


def generate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    categorical_limits: Optional[Dict[str, Any]] = None,
    numeric_ranges: Optional[Dict[str, Any]] = None,
    max_null_pct_required: float = 80.0,
) -> Dict[str, Any]:
    categorical_limits = categorical_limits or DEFAULT_CATEGORICAL_LIMITS
    numeric_ranges = numeric_ranges or DEFAULT_NUMERIC_RANGES

    schema: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "required_columns": required_columns,
        "rules": {
            "max_null_pct_required": float(max_null_pct_required),
            "categorical_limits": categorical_limits,
            "numeric_ranges": numeric_ranges,
        },
        "columns": {},
    }

    for col in df.columns:
        s = df[col]
        schema["columns"][col] = {
            "type": _dtype_family(s),
            "null_pct": round(float(s.isna().mean()) * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
        }

    return schema


def validate_against_schema(
    df: pd.DataFrame,
    baseline_schema: Dict[str, Any],
    required_columns: Optional[List[str]] = None,
    allow_new_columns: bool = True,
) -> Dict[str, Any]:
    violations: List[str] = []

    baseline_required = baseline_schema.get("required_columns", [])
    required = required_columns or baseline_required

    rules = baseline_schema.get("rules", {})
    max_null = float(rules.get("max_null_pct_required", 80.0))

    baseline_cols = set(baseline_schema.get("columns", {}).keys())
    actual_cols = set(df.columns)

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        violations.append(f"Missing required columns: {missing_required}")

    removed_cols = sorted(list(baseline_cols - actual_cols))
    if removed_cols:
        violations.append(f"Columns missing vs baseline schema: {removed_cols}")

    new_cols = sorted(list(actual_cols - baseline_cols))
    if new_cols and not allow_new_columns:
        violations.append(f"New columns appeared (not allowed): {new_cols}")

    for col, meta in baseline_schema.get("columns", {}).items():
        if col not in df.columns:
            continue

        s = df[col]
        actual_type = _dtype_family(s)
        expected_type = meta.get("type")

        if expected_type and actual_type != expected_type:
            violations.append(f"Type drift for '{col}': expected {expected_type}, got {actual_type}")

        if col in required:
            null_pct = float(s.isna().mean()) * 100
            if null_pct > max_null:
                violations.append(
                    f"Required '{col}' null_pct too high: {null_pct:.1f}% > {max_null:.1f}%"
                )

    cat_limits = rules.get("categorical_limits", {})
    for col, cfg in cat_limits.items():
        if col not in df.columns:
            continue
        allowed = set(str(x).upper() for x in cfg.get("allowed", []))
        if allowed:
            observed = set(str(x).upper() for x in df[col].dropna().astype(str).unique().tolist())
            bad = sorted(list(observed - allowed))
            if bad:
                violations.append(f"Invalid category values in '{col}': {bad}")

    num_ranges = rules.get("numeric_ranges", {})
    for col, rng in num_ranges.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(x) == 0:
            continue
        mn = rng.get("min", None)
        mx = rng.get("max", None)
        if mn is not None and (x < float(mn)).any():
            violations.append(f"'{col}' has values < {mn}")
        if mx is not None and (x > float(mx)).any():
            violations.append(f"'{col}' has values > {mx}")

    return {"passed": len(violations) == 0, "violations": violations}


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_schema_checkpoint(
    csv_path: str,
    baseline_schema_path: str,
    report_path: str,
    required_columns: List[str],
    mode: str = "enforce",
    allow_new_columns: bool = True,
    categorical_limits: Optional[Dict[str, Any]] = None,
    numeric_ranges: Optional[Dict[str, Any]] = None,
    max_null_pct_required: float = 80.0,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    if not os.path.exists(baseline_schema_path):
        baseline = generate_schema(
            df=df,
            required_columns=required_columns,
            categorical_limits=categorical_limits,
            numeric_ranges=numeric_ranges,
            max_null_pct_required=max_null_pct_required,
        )
        _write_json(baseline, baseline_schema_path)

    baseline = _read_json(baseline_schema_path)

    result = validate_against_schema(
        df=df,
        baseline_schema=baseline,
        required_columns=required_columns,
        allow_new_columns=allow_new_columns,
    )

    report = {
        "csv_path": csv_path,
        "baseline_schema_path": baseline_schema_path,
        "checked_at": datetime.utcnow().isoformat(),
        "mode": mode,
        "passed": bool(result["passed"]),
        "violations": result["violations"],
    }

    _write_json(report, report_path)

    if mode == "enforce" and not report["passed"]:
        raise ValueError(f"Schema checkpoint failed for {csv_path}: {report['violations']}")

    return report
