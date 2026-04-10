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
PATIENT_ID = "b889f1a4-10b7-495e-a10e-1a79a597b3c1"

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
    "Matched Criteria:", "Concerns:",
    "Intervention Summary:", "Clinical Rationale:",
]
for _m in _MARKERS:
    analysis_text = analysis_text.replace(_m, f"\n{_m}")
analysis_text = re.sub(r"(\*\*Trial\s+\d+)", r"\n\1", analysis_text)

# ── helpers ───────────────────────────────────────────────────────────────────
def render_bullet_lines(lines: list[str]):
    seen = set()
    for line in lines:
        s = line.strip().lstrip("- •*").strip()
        if s and s.lower() != "none" and s not in seen:
            seen.add(s)
            out(f"    • {s}")

def word_wrap(text: str, width: int = 66):
    words, buf = text.split(), []
    for w in words:
        if sum(len(x) + 1 for x in buf) + len(w) > width:
            out(f"    {' '.join(buf)}")
            buf = [w]
        else:
            buf.append(w)
    if buf:
        out(f"    {' '.join(buf)}")

# ── Split and render per-trial blocks ─────────────────────────────────────────
raw_blocks   = re.split(r"(?=\*\*Trial\s+\d+)", analysis_text)
trial_blocks = [b.strip() for b in raw_blocks if re.search(r"\*\*Trial\s+\d+", b)]

out()
divider()
out("TRIAL-BY-TRIAL ANALYSIS  (Gemini 2.5 Flash)")
divider()

if trial_blocks:
    for block in trial_blocks:
        hm = re.match(r"\*\*(Trial\s+\d+[^*]*)\*\*", block)
        trial_header = hm.group(1).strip() if hm else block.splitlines()[0]

        vm = re.search(r"VERDICT\s*:\s*(\w+)", block, re.IGNORECASE)
        verdict_key   = vm.group(1).upper() if vm else "UNKNOWN"
        verdict_label = VERDICT_LABEL.get(verdict_key, verdict_key)

        out(f"\n{'─' * W}")
        out(f"  {trial_header}")
        out(f"  VERDICT : {verdict_label}")
        out(f"{'─' * W}")

        # Parse sections
        sections: dict[str, list[str]] = {}
        current = None
        for line in block.splitlines():
            s = line.strip()
            if re.match(r"\*\*Trial\s+\d+", s) or re.match(r"VERDICT\s*:", s, re.IGNORECASE):
                continue
            hit = None
            for marker in _MARKERS[1:]:
                if s.startswith(marker):
                    hit = marker.rstrip(":")
                    sections.setdefault(hit, [])
                    rest = s[len(marker):].strip()
                    if rest:
                        sections[hit].append(rest)
                    current = hit
                    break
            if hit is None and current and s and s != "---":
                sections[current].append(s)

        matched = sections.get("Matched Criteria", [])
        if matched:
            out(f"\n  WHY THIS TRIAL MATCHES")
            render_bullet_lines(matched)

        concerns = sections.get("Concerns", [])
        non_empty = [c for c in concerns if c.strip().lstrip("-• ").lower() not in ("", "none")]
        if non_empty:
            out(f"\n  CONCERNS / OPEN QUESTIONS")
            render_bullet_lines(non_empty)

        intv = " ".join(sections.get("Intervention Summary", [])).strip()
        if intv:
            out(f"\n  WHAT THE PATIENT WOULD DO")
            out(f"    {intv[:300]}{'…' if len(intv) > 300 else ''}")

        rat = " ".join(sections.get("Clinical Rationale", [])).strip()
        if rat:
            out(f"\n  CLINICAL RATIONALE")
            word_wrap(rat)

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

# ── LLM as Judge ──────────────────────────────────────────────────────────────
out()
section("LLM AS JUDGE  (MedGemma — independent second opinion)")

# Build a map of Gemini's verdicts by trial number for comparison
gemini_verdicts: dict[int, str] = {}
for block in trial_blocks:
    nm = re.search(r"\*\*Trial\s+(\d+)", block)
    vm = re.search(r"VERDICT\s*:\s*(\w+)", block, re.IGNORECASE)
    if nm and vm:
        gemini_verdicts[int(nm.group(1))] = vm.group(1).upper()

VERDICT_ICON = {
    "ELIGIBLE":   "✓  ELIGIBLE",
    "INELIGIBLE": "✗  INELIGIBLE",
    "BORDERLINE": "~  BORDERLINE",
}
MATCH_LABEL = {True: "✓ matches Gemini", False: "⚠ differs from Gemini"}

medgemma_judgment = result.get("medgemma_judgment") or ""
if not medgemma_judgment:
    out("  (Judge output unavailable)")
elif medgemma_judgment.startswith("(Judge unavailable") or medgemma_judgment.startswith("(MedGemma"):
    out(f"  {medgemma_judgment}")
else:
    current_ref = None
    current_num = None
    current_verdict = None
    current_lines = []

    def flush_judge_block():
        if current_ref and current_verdict:
            gemini_v = gemini_verdicts.get(current_num, "")
            match = (current_verdict == gemini_v)
            icon  = VERDICT_ICON.get(current_verdict, current_verdict)
            out(f"\n{'─' * W}")
            out(f"  {current_ref}  →  MedGemma: {icon}")
            if gemini_v:
                out(f"             Gemini:   {VERDICT_ICON.get(gemini_v, gemini_v)}   [{MATCH_LABEL[match]}]")
            out(f"{'─' * W}")
            for cl in current_lines:
                word_wrap(cl, width=66)

    expanded_lines = []
    for raw_line in medgemma_judgment.splitlines():
        parts = re.split(r"(?=\bTrial\s+\d+\s*[:\*])", raw_line)
        expanded_lines.extend(parts)

    for raw_line in expanded_lines:
        line = re.sub(r"\*+", "", raw_line).strip()
        if not line:
            continue
        m = re.match(
            r"(Trial\s+(\d+))\s*:?\s*(ELIGIBLE|INELIGIBLE|BORDERLINE)\s*[—\-]+\s*(.*)",
            line, re.IGNORECASE
        )
        if m:
            flush_judge_block()
            current_ref     = m.group(1)
            current_num     = int(m.group(2))
            current_verdict = m.group(3).upper()
            current_lines   = [m.group(4).strip()] if m.group(4).strip() else []
        elif current_ref:
            current_lines.append(line)

    flush_judge_block()
    if not current_ref:
        out()
        out("  (Could not parse judge output — raw response below:)")
        for raw_line in medgemma_judgment.splitlines():
            if raw_line.strip():
                out(f"  {raw_line.strip()}")
    out()

# ── Footer ────────────────────────────────────────────────────────────────────
_txt_file.close()
print(f"\nFull results saved to:")
print(f"  Formatted : {txt_path}")
print(f"  Raw JSON  : {json_path}")
