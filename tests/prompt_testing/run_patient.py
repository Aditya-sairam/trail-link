"""
Run RAG pipeline locally for a patient uploaded via the TrialLink frontend.

Usage:
    cd /Users/vaishnavisarmalkar/Documents/trail-link
    source venv/bin/activate
    python tests/prompt_testing/run_patient.py

Output is printed to the terminal AND saved to:
    tests/prompt_testing/results/<patient_id>_<timestamp>.txt   (formatted)
    tests/prompt_testing/results/<patient_id>_<timestamp>.json  (raw pipeline result)
"""

import sys
import os
import re
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sdk/patient_package")))

from models.rag_service import rag_pipeline_for_patient

# Patient uploaded via frontend — swap this ID to test different patients
PATIENT_ID = "d1693b81-62c4-4149-af0e-ee6d9a99d3a1"

# ── Output setup ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.getcwd(), "tests", "prompt_testing", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_path  = os.path.join(RESULTS_DIR, f"{PATIENT_ID}_{timestamp}.txt")
json_path = os.path.join(RESULTS_DIR, f"{PATIENT_ID}_{timestamp}.json")

_txt_file = open(txt_path, "w", encoding="utf-8")

def out(*args, **kwargs):
    """Print to both terminal and the result file."""
    print(*args, **kwargs)
    kwargs.pop("file", None)
    print(*args, **kwargs, file=_txt_file)

# ── Helpers ───────────────────────────────────────────────────────────────────
W = 72

def divider(char="="):
    out(char * W)

def section(title, char="="):
    divider(char)
    out(title)
    divider(char)

VERDICT_LABEL = {
    "ELIGIBLE":   "✓  ELIGIBLE",
    "INELIGIBLE": "✗  INELIGIBLE",
    "BORDERLINE": "~  BORDERLINE (requires clinician review)",
}

# ── Run pipeline ──────────────────────────────────────────────────────────────
out(f"\nRunning RAG pipeline for patient: {PATIENT_ID}\n")

result = rag_pipeline_for_patient(PATIENT_ID)

# Save raw JSON immediately so it's available even if display crashes
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(result, jf, indent=2, default=str)

# ── Patient Summary ───────────────────────────────────────────────────────────
section("PATIENT SUMMARY")
out(result["patient_summary"])

# ── Guardrail Status ──────────────────────────────────────────────────────────
guardrail = result["guardrail"]
status    = guardrail["status"].upper()
icon      = {"PASSED": "✓", "BLOCKED": "✗", "FLAGGED": "⚠"}.get(status, "?")

out()
section(f"GUARDRAIL  {icon}  {status}")
if guardrail.get("reason"):
    out(f"  Reason : {guardrail['reason']}")
if guardrail.get("pii_hits"):
    hits = {k: v for k, v in guardrail["pii_hits"].items() if v > 0}
    if hits:
        out(f"  PII    : {hits}")
if guardrail.get("flag_reasons"):
    for fr in guardrail["flag_reasons"]:
        out(f"  Flag   : {fr}")
if guardrail.get("llm_input_judgment"):
    j = guardrail["llm_input_judgment"]
    out(f"  LLM input judge  : category={j.get('category')}  valid={j.get('is_valid')}")
    out(f"    reason: {j.get('reason')}")
if guardrail.get("llm_output_judgment"):
    j = guardrail["llm_output_judgment"]
    out(f"  LLM output judge : category={j.get('category')}  safe={j.get('is_safe')}  grounded={j.get('is_grounded')}")
    out(f"    reason: {j.get('reason')}")

# Blocked — show reason and stop
if status == "BLOCKED":
    out()
    out(result.get("recommendation", ""))
    _txt_file.close()
    out(f"\nResults saved to:\n  {txt_path}\n  {json_path}")
    sys.exit(0)

# ── Retrieved Trials Overview ─────────────────────────────────────────────────
trials = result.get("retrieved_trials", [])
out()
section(
    f"TOP {len(trials)} RETRIEVED TRIALS  "
    f"(from {len(result.get('candidates_before_rerank', []))} candidates before rerank)"
)
for i, t in enumerate(trials, 1):
    nct       = t.get("nct_number") or "?"
    title     = t.get("study_title") or t.get("title") or "N/A"
    phase     = str(t.get("phase") or "N/A")
    rec_st    = str(t.get("recruitment_status") or "N/A")
    condition = str(t.get("conditions") or t.get("disease") or "N/A")
    sex       = str(t.get("sex") or "N/A")
    min_age   = str(t.get("min_age") or "?")
    max_age   = str(t.get("max_age") or "?")
    url       = str(t.get("study_url") or "")
    keywords  = str(t.get("keywords") or "")
    summary   = str(t.get("brief_summary") or "")

    out(f"\n  [{i}] {nct}")
    out(f"      Title       : {title}")
    out(f"      Condition   : {condition}")
    out(f"      Phase       : {phase:<14}  Recruitment : {rec_st}")
    out(f"      Age range   : {min_age} – {max_age}   Sex : {sex}")
    if keywords:
        out(f"      Keywords    : {str(keywords)[:100]}")
    if summary:
        out(f"      Summary     : {str(summary)[:200]}")
    if url:
        out(f"      URL         : {url}")

# ── Per-Trial Analysis from MedGemma ─────────────────────────────────────────
# If guardrail replaced the recommendation, fall back to the raw MedGemma output
recommendation = result.get("recommendation", "")
raw_output      = result.get("raw_medgemma_output", "")
guardrail_replaced = recommendation != raw_output and raw_output

if guardrail_replaced:
    out()
    out("  NOTE: Output guardrail fired — showing raw MedGemma output below.")
    out(f"  Guardrail reason: {guardrail.get('reason', '')}")
    recommendation = raw_output

out()
divider()
out("TRIAL-BY-TRIAL ANALYSIS  (MedGemma)")
divider()

# MedGemma omits newlines between sections — insert them before known markers
# so the rest of the parsing can work line-by-line.
SECTION_MARKERS = [
    "VERDICT:",
    "Inclusion Criteria Check:",
    "Exclusion Criteria Check:",
    "Medication/Allergy Conflicts:",
    "Comorbidity Flags:",
    "Intervention Summary:",
    "Clinical Rationale:",
    "---",
]

def normalise(text: str) -> str:
    """Insert newlines before every known section marker and trial header."""
    for marker in SECTION_MARKERS:
        text = text.replace(marker, f"\n{marker}")
    # Also ensure **Trial N** always starts on its own line
    text = re.sub(r"(\*\*Trial\s+\d+)", r"\n\1", text)
    return text

normalised = normalise(recommendation.strip())

# Split on **Trial N ...** headers
raw_blocks   = re.split(r"(?=\*\*Trial\s+\d+)", normalised)
trial_blocks = [b.strip() for b in raw_blocks if b.strip()]

if trial_blocks:
    for block in trial_blocks:
        header_match = re.match(r"\*\*(Trial\s+\d+[^*]*)\*\*", block)
        trial_header = header_match.group(1).strip() if header_match else block.splitlines()[0]

        verdict_match = re.search(r"VERDICT\s*:\s*(\w+)", block, re.IGNORECASE)
        verdict_key   = verdict_match.group(1).upper() if verdict_match else "UNKNOWN"
        verdict_label = VERDICT_LABEL.get(verdict_key, verdict_key)

        out(f"\n{'─' * W}")
        out(f"  {trial_header}")
        out(f"  VERDICT : {verdict_label}")
        out(f"{'─' * W}")

        # Print body — skip the **Trial N** header and VERDICT lines
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or re.match(r"\*\*Trial\s+\d+", stripped):
                continue
            if re.match(r"VERDICT\s*:", stripped, re.IGNORECASE):
                continue
            if stripped == "---":
                continue
            # Section sub-headings get their own line with spacing
            if re.match(
                r"(Inclusion Criteria|Exclusion Criteria|Medication/Allergy|"
                r"Comorbidity Flags|Intervention Summary|Clinical Rationale)",
                stripped, re.IGNORECASE
            ):
                out(f"\n  {stripped}")
            elif stripped.startswith("-"):
                out(f"    {stripped}")
            else:
                out(f"    {stripped}")

    out(f"\n{'─' * W}")
else:
    # Fallback: raw text when MedGemma didn't follow the expected format
    out()
    out(recommendation or "(no recommendation — patient was blocked/flagged)")

# ── Disclaimer ────────────────────────────────────────────────────────────────
disclaimer_match = re.search(r"(Disclaimer:.*)", recommendation, re.IGNORECASE | re.DOTALL)
if disclaimer_match:
    out()
    divider("-")
    out(disclaimer_match.group(1).strip())
    divider("-")

# ── Footer ────────────────────────────────────────────────────────────────────
_txt_file.close()
print(f"\nFull results saved to:")
print(f"  Formatted : {txt_path}")
print(f"  Raw JSON  : {json_path}")
