# pipelines/dags/src/conditions/registry.py

import pandas as pd


def classify_diabetes(conditions: str) -> str:
    if conditions is None or pd.isna(conditions):
        return "Unknown"
    c = str(conditions).lower()
    if "type 1" in c or "t1d" in c or "t1dm" in c or "juvenile" in c or "iddm" in c or "lada" in c:
        return "Type 1 Diabetes"
    elif "type 2" in c or "t2d" in c or "t2dm" in c or "niddm" in c or "non-insulin dependent" in c:
        return "Type 2 Diabetes"
    elif "gestational" in c or "gdm" in c:
        return "Gestational Diabetes"
    elif "prediabetes" in c or "pre-diabetes" in c or "impaired glucose" in c:
        return "Pre-Diabetes"
    elif "insipidus" in c:
        return "Diabetes Insipidus"
    elif "neonatal" in c or "monogenic" in c or "mody" in c:
        return "Rare Diabetes"
    else:
        return "Diabetes (General)"


def classify_breast_cancer(conditions: str) -> str:
    if conditions is None or pd.isna(conditions):
        return "Unknown"
    c = str(conditions).lower()
    if "triple negative" in c or "tnbc" in c:
        return "Triple Negative Breast Cancer"
    elif "her2" in c or "her-2" in c or "erbb2" in c:
        return "HER2 Positive Breast Cancer"
    elif "hormone receptor" in c or "hr+" in c or "er+" in c or "estrogen receptor" in c:
        return "Hormone Receptor Positive Breast Cancer"
    elif "inflammatory" in c:
        return "Inflammatory Breast Cancer"
    elif "metastatic" in c:
        return "Metastatic Breast Cancer"
    elif "ductal" in c or "dcis" in c:
        return "Ductal Carcinoma"
    elif "lobular" in c:
        return "Lobular Carcinoma"
    elif "early stage" in c or "early-stage" in c:
        return "Early Stage Breast Cancer"
    else:
        return "Breast Cancer (General)"


REGISTRY = {
    "diabetes": {
        "query": "diabetes",
        "disease": "diabetes",
        "raw_path": "data/diabetes/raw/diabetes_trials_raw.csv",
        "enriched_path": "data/diabetes/processed/diabetes_trials_enriched.csv",
        "reports_dir": "data/diabetes/reports",
        "schema_dir": "data/diabetes/schema",
        "raw_schema_path": "data/diabetes/schema/raw_schema.json",
        "processed_schema_path": "data/diabetes/schema/processed_schema.json",
        "classifier": classify_diabetes,
    },
    "breast_cancer": {
        "query": "breast cancer",
        "disease": "breast_cancer",
        "raw_path": "data/breast_cancer/raw/breast_cancer_trials_raw.csv",
        "enriched_path": "data/breast_cancer/processed/breast_cancer_trials_enriched.csv",
        "reports_dir": "data/breast_cancer/reports",
        "schema_dir": "data/breast_cancer/schema",
        "raw_schema_path": "data/breast_cancer/schema/raw_schema.json",
        "processed_schema_path": "data/breast_cancer/schema/processed_schema.json",
        "classifier": classify_breast_cancer,
    },
}