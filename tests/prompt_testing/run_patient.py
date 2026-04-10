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
PATIENT_ID = "94a45350-5770-48ff-8002-75beaa9f99eb"

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
recommendation = result.get("recommendation", "")
raw_output     = result.get("raw_medgemma_output", "")

# Use raw MedGemma output when guardrail replaced the recommendation
analysis_text = raw_output if (raw_output and recommendation != raw_output) else recommendation

# Strip any leading "Output:" or "Prompt:..." prefix MedGemma echoes back
analysis_text = re.sub(r"^(Output|Prompt)\s*:\s*\n?", "", analysis_text.strip(), flags=re.IGNORECASE)

# ── Normalise: insert newlines before section markers (MedGemma omits them) ──
_MARKERS = [
    "VERDICT:", "Inclusion Criteria Check:", "Exclusion Criteria Check:",
    "Medication/Allergy Conflicts:", "Comorbidity Flags:",
    "Intervention Summary:", "Clinical Rationale:",
]
for _m in _MARKERS:
    analysis_text = analysis_text.replace(_m, f"\n{_m}")
analysis_text = re.sub(r"(\*\*Trial\s+\d+)", r"\n\1", analysis_text)

# ── Parse each criterion bullet into (status, criterion, reason) ──────────────
_NO_INFO_PHRASES = ("no information", "not mentioned", "not specified", "not provided",
                    "not stated", "not available", "not described")

def parse_criteria_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split criterion bullets into (detailed, no_info_names).
    detailed  = lines that have real patient evidence (shown in full).
    no_info   = short criterion names where only 'no info' was noted (collapsed).
    """
    detailed, no_info = [], []
    for line in lines:
        stripped = line.strip().lstrip("- ").strip()
        if not stripped:
            continue
        reason_lower = stripped.lower()
        if any(p in reason_lower for p in _NO_INFO_PHRASES):
            # Extract just the criterion name (before the ✓/✗)
            name = re.split(r"[✓✗]", stripped)[0].strip().rstrip(":").strip()
            # Keep it short
            no_info.append(name[:80] + ("…" if len(name) > 80 else ""))
        else:
            detailed.append(stripped)
    return detailed, no_info

# ── Split and render per-trial blocks ─────────────────────────────────────────
raw_blocks   = re.split(r"(?=\*\*Trial\s+\d+)", analysis_text)
trial_blocks = [b.strip() for b in raw_blocks if re.search(r"\*\*Trial\s+\d+", b)]

out()
divider()
out("TRIAL-BY-TRIAL ANALYSIS  (MedGemma)")
divider()

if trial_blocks:
    for block in trial_blocks:
        # Header
        hm = re.match(r"\*\*(Trial\s+\d+[^*]*)\*\*", block)
        trial_header = hm.group(1).strip() if hm else block.splitlines()[0]

        # Verdict
        vm = re.search(r"VERDICT\s*:\s*(\w+)", block, re.IGNORECASE)
        verdict_key   = vm.group(1).upper() if vm else "UNKNOWN"
        verdict_label = VERDICT_LABEL.get(verdict_key, verdict_key)

        out(f"\n{'─' * W}")
        out(f"  {trial_header}")
        out(f"  VERDICT : {verdict_label}")
        out(f"{'─' * W}")

        # Split block into named sections
        sections: dict[str, list[str]] = {}
        current = None
        for line in block.splitlines():
            s = line.strip()
            if re.match(r"\*\*Trial\s+\d+", s) or re.match(r"VERDICT\s*:", s, re.IGNORECASE):
                continue
            matched_section = None
            for marker in _MARKERS[1:]:   # skip VERDICT
                if s.startswith(marker):
                    matched_section = marker.rstrip(":")
                    sections.setdefault(matched_section, [])
                    remainder = s[len(marker):].strip()
                    if remainder:
                        sections[matched_section].append(remainder)
                    current = matched_section
                    break
            if matched_section is None and current and s and s != "---":
                sections[current].append(s)

        # Inclusion Criteria
        inc_lines = sections.get("Inclusion Criteria Check", [])
        if inc_lines:
            out(f"\n  INCLUSION CRITERIA")
            detailed, no_info = parse_criteria_lines(inc_lines)
            for d in detailed:
                out(f"    • {d}")
            if no_info:
                out(f"    • Not enough patient data to assess ({len(no_info)} criteria):")
                for n in no_info:
                    out(f"        – {n}")

        # Exclusion Criteria
        exc_lines = sections.get("Exclusion Criteria Check", [])
        if exc_lines:
            out(f"\n  EXCLUSION CRITERIA")
            detailed, no_info = parse_criteria_lines(exc_lines)
            for d in detailed:
                out(f"    • {d}")
            if no_info:
                out(f"    • Not enough patient data to assess ({len(no_info)} criteria):")
                for n in no_info:
                    out(f"        – {n}")

        # Medication/Allergy
        med = " ".join(sections.get("Medication/Allergy Conflicts", [])).strip()
        if med and med.lower() not in ("none", "n/a", "[none]"):
            out(f"\n  MEDICATION / ALLERGY CONFLICTS")
            out(f"    {med}")

        # Comorbidity Flags
        comor = " ".join(sections.get("Comorbidity Flags", [])).strip()
        if comor and comor.lower() not in ("none", "n/a", "[none]"):
            out(f"\n  COMORBIDITY FLAGS")
            out(f"    {comor}")

        # Intervention Summary
        intv = " ".join(sections.get("Intervention Summary", [])).strip()
        if intv:
            out(f"\n  INTERVENTION")
            out(f"    {intv[:300]}{'…' if len(intv) > 300 else ''}")

        # Clinical Rationale  ← the key "why" section
        rat = " ".join(sections.get("Clinical Rationale", [])).strip()
        if rat:
            out(f"\n  CLINICAL RATIONALE")
            # Word-wrap at ~66 chars
            words, line_buf = rat.split(), []
            for w in words:
                if sum(len(x) + 1 for x in line_buf) + len(w) > 66:
                    out(f"    {'  '.join(line_buf)}")
                    line_buf = [w]
                else:
                    line_buf.append(w)
            if line_buf:
                out(f"    {'  '.join(line_buf)}")

    out(f"\n{'─' * W}")
else:
    out()
    out(analysis_text or "(no analysis available)")

# ── Disclaimer ────────────────────────────────────────────────────────────────
dm = re.search(r"(Disclaimer:.*)", analysis_text, re.IGNORECASE | re.DOTALL)
if dm:
    out()
    divider("-")
    out(dm.group(1).strip())
    divider("-")

# ── Footer ────────────────────────────────────────────────────────────────────
_txt_file.close()
print(f"\nFull results saved to:")
print(f"  Formatted : {txt_path}")
print(f"  Raw JSON  : {json_path}")
