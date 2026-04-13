"""
TrialLink RAG Pipeline — Evaluation Runner
==========================================
Fetches ALL patients from Firestore (patient-db-dev),
runs the full RAG pipeline on each, and saves results.

Usage:
    cd /path/to/trail-link
    source .venv/bin/activate

    # Run all patients
    python tests/evaluation_1/run_eval.py

    # Dry run — list patients without running pipeline
    python tests/evaluation_1/run_eval.py --dry_run

    # Run only specific condition
    python tests/evaluation_1/run_eval.py --condition breast_cancer
    python tests/evaluation_1/run_eval.py --condition diabetes

    # Resume a previous run (skip already-completed patients)
    python tests/evaluation_1/run_eval.py --resume --run_id 20260411_014618

    # Limit patients (for quick testing)
    python tests/evaluation_1/run_eval.py --limit 5

Output:
    tests/evaluation_1/results/<run_id>/
        <patient_id>.json       <- one per patient
        summary.json            <- aggregated stats

Then generate report:
    python tests/evaluation_1/generate_eval_report.py --run_id <run_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# ── GCP env vars (must be set before importing rag_service) ──────────────────
os.environ.setdefault("GCP_PROJECT_ID",           "triallink-eval-001")
os.environ.setdefault("MODEL_PROJECT_ID",          "mlops-triallink")
os.environ.setdefault("MODEL_PROJECT_NUMBER",      "153563619775")
os.environ.setdefault("GCP_REGION",                "us-central1")
os.environ.setdefault("MEDGEMMA_ENDPOINT_ID",      "mg-endpoint-a55baaeb-7cc5-4ff4-b3c9-97e6c20bfaad")
os.environ.setdefault("FIRESTORE_DATABASE",        "clinical-trials-db")
os.environ.setdefault("PATIENT_DB",                "patient-db-dev")
os.environ.setdefault("VECTOR_SEARCH_ENDPOINT_ID",
    "projects/408416535077/locations/us-central1/indexEndpoints/4231811348100546560")
os.environ.setdefault("DEPLOYED_INDEX_ID",         "clinical_trials_dev")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parent.parent.parent
EVAL_DIR     = Path(__file__).parent
RESULTS_ROOT = EVAL_DIR / "results"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sdk" / "patient_package"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONDITION INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

BREAST_CANCER_CODES = {
    "254837009", "372137005", "413448000", "188161004", "276796008",
    "408643008", "722524005", "396752002", "427685000", "353431000",
}
BREAST_CANCER_KEYWORDS = [
    "breast cancer", "breast carcinoma", "breast neoplasm",
    "malignant neoplasm of breast", "her2", "estrogen receptor",
    "triple negative", "dcis", "ductal carcinoma",
]

DIABETES_CODES = {
    "44054006", "73211009", "359642000", "314771006",
    "9859006",  "190331003", "408540003", "420868002",
}
DIABETES_KEYWORDS = [
    "diabetes mellitus type 2", "type 2 diabetes", "t2dm",
    "diabetes mellitus", "prediabetes", "prediabetic",
    "hyperglycemia", "insulin resistance", "metabolic syndrome",
]


def infer_condition(patient_data: dict) -> str:
    """
    Infer patient condition from their conditions list.
    Returns: 'breast_cancer' | 'diabetes' | 'other'
    Priority: breast_cancer > diabetes > other
    """
    conditions = patient_data.get("conditions", [])

    for cond in conditions:
        code    = str(cond.get("code", "")).strip()
        display = str(cond.get("display_name", "")).lower()
        if code in BREAST_CANCER_CODES:
            return "breast_cancer"
        if any(kw in display for kw in BREAST_CANCER_KEYWORDS):
            return "breast_cancer"

    for cond in conditions:
        code    = str(cond.get("code", "")).strip()
        display = str(cond.get("display_name", "")).lower()
        if code in DIABETES_CODES:
            return "diabetes"
        if any(kw in display for kw in DIABETES_KEYWORDS):
            return "diabetes"

    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# FIRESTORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_all_patients(project_id: str, patient_db: str) -> list[dict]:
    """
    Fetch all patient documents from Firestore.
    Returns list of {patient_id, condition, raw_data}.
    """
    from google.cloud import firestore

    logger.info(f"Connecting to Firestore: project={project_id}, db={patient_db}")
    db   = firestore.Client(project=project_id, database=patient_db)
    docs = list(db.collection("patients").stream())
    logger.info(f"Found {len(docs)} patient documents")

    patients = []
    for doc in docs:
        data      = doc.to_dict() or {}
        condition = infer_condition(data)
        patients.append({
            "patient_id": doc.id,
            "condition":  condition,
        })

    return patients


# ══════════════════════════════════════════════════════════════════════════════
# RESULT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_overall_verdict(result: dict) -> str:
    """Extract the best overall verdict from a pipeline result."""
    guard_status = result.get("guardrail", {}).get("status", "unknown")
    if guard_status == "blocked":
        return "BLOCKED"

    rec      = result.get("recommendation", "")
    verdicts = re.findall(
        r"VERDICT\s*:\s*(ELIGIBLE|INELIGIBLE|BORDERLINE)", rec, re.IGNORECASE
    )

    if any(v.upper() == "ELIGIBLE"   for v in verdicts): return "ELIGIBLE"
    if any(v.upper() == "BORDERLINE" for v in verdicts): return "BORDERLINE"
    if verdicts:                                          return "INELIGIBLE"
    return "BLOCKED"


def save_result(result: dict, run_dir: Path) -> None:
    pid      = result["patient_id"]
    out_path = run_dir / f"{pid}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def save_summary(run_dir: Path, run_id: str, stats: dict, records: list[dict]) -> None:
    summary = {
        "run_id":           run_id,
        "generated_at":     datetime.utcnow().isoformat(),
        **stats,
        "patients":         records,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved → {run_dir / 'summary.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

W = 72

def print_header(run_id: str, total: int) -> None:
    print("\n" + "=" * W)
    print(f"  TrialLink Evaluation Run")
    print(f"  Run ID   : {run_id}")
    print(f"  Patients : {total}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W)


def print_row(idx: int, total: int, pid: str, condition: str,
              overall: str, guard: str, elapsed: float) -> None:
    icon = {"PASSED": "✓", "BLOCKED": "✗", "FLAGGED": "⚠"}.get(guard.upper(), "?")
    v_icon = {"ELIGIBLE": "✅", "BORDERLINE": "🔶", "INELIGIBLE": "❌", "BLOCKED": "🚫", "FAILED": "💥"}.get(overall, "?")
    print(f"  [{idx:>3}/{total}] {pid[:8]}  {condition:<16} {icon} {guard:<8}  {v_icon} {overall:<12} {elapsed:.1f}s")


def print_summary(stats: dict, run_id: str, run_dir: Path) -> None:
    print("\n" + "=" * W)
    print("  EVALUATION COMPLETE")
    print("=" * W)
    print(f"  run_id       : {run_id}")
    print(f"  total        : {stats['total']}")
    print(f"  passed       : {stats['passed']}")
    print(f"  blocked      : {stats['blocked']}  (guardrail blocked)")
    print(f"  flagged      : {stats['flagged']}  (guardrail flagged)")
    print(f"  failed       : {stats['failed']}  (pipeline errors)")
    print(f"  skipped      : {stats['skipped']}  (resume mode)")
    print(f"  duration     : {stats['duration_seconds']:.0f}s")
    print()
    print("  By condition:")
    for cond, counts in sorted(stats["by_condition"].items()):
        print(f"    {cond:<18}: {counts}")
    print()
    print("  By verdict:")
    for verdict, count in sorted(stats["by_verdict"].items()):
        print(f"    {verdict:<14}: {count}")
    print(f"\n  results dir  : {run_dir}")
    print("=" * W)
    print(f"\n  Next — generate report:")
    print(f"  python tests/evaluation_1/generate_eval_report.py --run_id {run_id}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TrialLink Evaluation Runner")
    parser.add_argument("--dry_run",   action="store_true",
                        help="List patients without running pipeline")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["breast_cancer", "diabetes", "other"],
                        help="Only evaluate patients of this condition")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip patients already saved in --run_id dir")
    parser.add_argument("--run_id",    type=str, default=None,
                        help="Existing run_id to resume (required with --resume)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max patients to process (for quick testing)")
    args = parser.parse_args()

    project_id = os.environ["GCP_PROJECT_ID"]
    patient_db = os.environ["PATIENT_DB"]

    # ── Run directory ─────────────────────────────────────────────────────────
    if args.resume and args.run_id:
        run_id  = args.run_id
        run_dir = RESULTS_ROOT / run_id
        if not run_dir.exists():
            print(f"Error: {run_dir} does not exist. Cannot resume.")
            sys.exit(1)
        print(f"Resuming run: {run_id}")
    else:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch all patients ────────────────────────────────────────────────────
    print(f"\nFetching patients from Firestore ({patient_db})...")
    all_patients = fetch_all_patients(project_id, patient_db)

    if not all_patients:
        print("No patients found. Upload patients via the TrialLink frontend first.")
        sys.exit(1)

    # Condition filter
    if args.condition:
        all_patients = [p for p in all_patients if p["condition"] == args.condition]
        print(f"Filtered to {len(all_patients)} '{args.condition}' patients")

    # Limit
    if args.limit:
        all_patients = all_patients[:args.limit]
        print(f"Limited to first {args.limit} patients")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'─' * W}")
        print(f"DRY RUN — {len(all_patients)} patients found")
        print(f"{'─' * W}")
        cond_counts = Counter(p["condition"] for p in all_patients)
        for cond, count in sorted(cond_counts.items()):
            print(f"  {cond:<18}: {count} patients")
        print(f"{'─' * W}")
        for p in all_patients:
            print(f"  {p['patient_id'][:8]}...  [{p['condition']}]")
        print(f"\nRun without --dry_run to execute.\n")
        return

    # ── Import pipeline (after env vars set) ──────────────────────────────────
    from models.rag_service import rag_pipeline_for_patient

    # ── Stats ─────────────────────────────────────────────────────────────────
    stats = {
        "total":            len(all_patients),
        "passed":           0,
        "blocked":          0,
        "flagged":          0,
        "failed":           0,
        "skipped":          0,
        "duration_seconds": 0.0,
        "by_condition":     {},
        "by_verdict":       {
            "ELIGIBLE": 0, "BORDERLINE": 0,
            "INELIGIBLE": 0, "BLOCKED": 0, "FAILED": 0,
        },
    }
    records    = []
    start_time = time.time()

    print_header(run_id, len(all_patients))

    # ── Pipeline loop ─────────────────────────────────────────────────────────
    for idx, p in enumerate(all_patients, 1):
        pid       = p["patient_id"]
        condition = p["condition"]
        out_file  = run_dir / f"{pid}.json"

        # Resume: skip already completed
        if args.resume and out_file.exists():
            print(f"  [{idx:>3}/{len(all_patients)}] {pid[:8]}  SKIPPED")
            stats["skipped"] += 1
            continue

        t0 = time.time()
        try:
            result  = rag_pipeline_for_patient(pid)
            elapsed = time.time() - t0

            # Tag
            result["patient_id"]     = pid
            result["condition"]      = condition
            result["evaluated_at"]   = datetime.utcnow().isoformat()
            result["elapsed_seconds"] = round(elapsed, 2)

            overall      = extract_overall_verdict(result)
            guard_status = result.get("guardrail", {}).get("status", "unknown")

            result["overall_verdict"] = overall

            # Stats
            if guard_status == "passed":   stats["passed"]  += 1
            elif guard_status == "blocked": stats["blocked"] += 1
            else:                           stats["flagged"] += 1

            stats["by_verdict"][overall] = stats["by_verdict"].get(overall, 0) + 1

            if condition not in stats["by_condition"]:
                stats["by_condition"][condition] = {
                    "total": 0, "eligible": 0, "borderline": 0,
                    "ineligible": 0, "blocked": 0, "failed": 0,
                }
            stats["by_condition"][condition]["total"] += 1
            stats["by_condition"][condition][overall.lower()] += 1

            # Save
            save_result(result, run_dir)
            records.append({
                "patient_id":       pid,
                "condition":        condition,
                "overall_verdict":  overall,
                "guardrail_status": guard_status,
                "n_retrieved":      len(result.get("retrieved_trials", [])),
                "elapsed_s":        round(elapsed, 2),
            })

            print_row(idx, len(all_patients), pid, condition, overall, guard_status, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"Pipeline failed for {pid}: {e}")
            stats["failed"] += 1

            error_result = {
                "patient_id":     pid,
                "condition":      condition,
                "error":          str(e),
                "overall_verdict": "FAILED",
                "guardrail":      {"status": "error"},
                "evaluated_at":   datetime.utcnow().isoformat(),
                "elapsed_seconds": round(elapsed, 2),
            }
            save_result(error_result, run_dir)
            stats["by_condition"].setdefault(condition, {
                "total": 0, "eligible": 0, "borderline": 0,
                "ineligible": 0, "blocked": 0, "failed": 0,
            })
            stats["by_condition"][condition]["total"]  += 1
            stats["by_condition"][condition]["failed"] += 1
            stats["by_verdict"]["FAILED"] = stats["by_verdict"].get("FAILED", 0) + 1

            records.append({
                "patient_id":       pid,
                "condition":        condition,
                "overall_verdict":  "FAILED",
                "guardrail_status": "error",
                "n_retrieved":      0,
                "elapsed_s":        round(elapsed, 2),
            })
            print(f"  [{idx:>3}/{len(all_patients)}] {pid[:8]}  FAILED: {e}")

    # ── Finalize ──────────────────────────────────────────────────────────────
    stats["duration_seconds"] = round(time.time() - start_time, 2)
    save_summary(run_dir, run_id, stats, records)
    print_summary(stats, run_id, run_dir)


if __name__ == "__main__":
    main()