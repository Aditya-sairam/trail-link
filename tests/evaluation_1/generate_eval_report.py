"""
TrialLink RAG Pipeline — Automated Evaluation Report Generator
==============================================================
Generates a full PDF evaluation report from results JSONs.

Usage:
    python generate_eval_report.py --run_id 20260411_014618
    python generate_eval_report.py --results_dir tests/evaluation_1/results

Output:
    tests/evaluation_1/reports/eval_report_<run_id>.pdf
    tests/evaluation_1/reports/eval_report_<run_id>.html  (interactive)

Install:
    pip install pandas plotly kaleido reportlab --break-system-packages
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════
TEAL       = "#1ABC9C"
DARK_TEAL  = "#148F77"
ORANGE     = "#E67E22"
RED        = "#E74C3C"
BLUE       = "#2980B9"
LIGHT_GRAY = "#F2F3F4"
MID_GRAY   = "#BDC3C7"
DARK_GRAY  = "#2C3E50"

CONDITION_COLORS = {
    "breast_cancer": "#E91E8C",
    "diabetes":      "#2196F3",
    "other":         "#9E9E9E",
}

VERDICT_COLORS = {
    "ELIGIBLE":   "#1ABC9C",
    "BORDERLINE": "#F39C12",
    "INELIGIBLE": "#E74C3C",
    "BLOCKED":    "#95A5A6",
    "FAILED":     "#8E44AD",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PARSE
# ══════════════════════════════════════════════════════════════════════════════

def load_results(run_dir: Path) -> list[dict]:
    results = []
    for f in sorted(run_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"  Warning: could not parse {f.name}: {e}")
    return results


def parse_verdicts_from_recommendation(recommendation: str) -> list[str]:
    return re.findall(
        r"VERDICT\s*:\s*(ELIGIBLE|INELIGIBLE|BORDERLINE)",
        recommendation,
        re.IGNORECASE,
    )


def build_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        pid      = str(r.get("patient_id", "unknown"))
        condition = str(r.get("condition", "unknown")).lower()
        rec      = str(r.get("recommendation", ""))
        guard    = r.get("guardrail", {})
        trials   = r.get("retrieved_trials", [])
        mg_text  = str(r.get("medgemma_judgment", ""))

        verdicts    = parse_verdicts_from_recommendation(rec)
        mg_verdicts = parse_verdicts_from_recommendation(mg_text)

        # Overall verdict
        if "ELIGIBLE" in [v.upper() for v in verdicts]:
            patient_verdict = "ELIGIBLE"
        elif "BORDERLINE" in [v.upper() for v in verdicts]:
            patient_verdict = "BORDERLINE"
        elif guard.get("status") == "blocked":
            patient_verdict = "BLOCKED"
        elif not verdicts:
            patient_verdict = "BLOCKED"
        else:
            patient_verdict = "INELIGIBLE"

        flag_reasons     = guard.get("flag_reasons", [])
        guardrail_status = guard.get("status", "unknown")
        mg_available     = bool(mg_text) and not mg_text.startswith("(MedGemma judge unavailable")

        consensus_count = 0
        disagree_count  = 0
        if mg_available and verdicts and mg_verdicts:
            for gv, mv in zip(verdicts, mg_verdicts):
                if gv.upper() == mv.upper():
                    consensus_count += 1
                else:
                    disagree_count += 1

        rows.append({
            "patient_id":        pid[:8],
            "condition":         condition,
            "guardrail_status":  guardrail_status,
            "patient_verdict":   patient_verdict,
            "n_retrieved":       len(trials),
            "n_eligible":        sum(1 for v in verdicts if v.upper() == "ELIGIBLE"),
            "n_borderline":      sum(1 for v in verdicts if v.upper() == "BORDERLINE"),
            "n_ineligible":      sum(1 for v in verdicts if v.upper() == "INELIGIBLE"),
            "n_verdicts_total":  len(verdicts),
            "flag_count":        len(flag_reasons),
            "flag_reasons":      "; ".join(flag_reasons[:2]),
            "mg_available":      mg_available,
            "consensus_count":   consensus_count,
            "disagree_count":    disagree_count,
            "input_llm_error":   any("input_llm_guardrail_error" in f for f in flag_reasons),
            "output_llm_error":  any("output_llm_guardrail_error" in f for f in flag_reasons),
            "policy_flagged":    any("Dosage" in f for f in flag_reasons),
            "grounding_flagged": any("grounding" in f.lower() for f in flag_reasons),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary_metrics(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {}

    eligible   = (df["patient_verdict"] == "ELIGIBLE").sum()
    borderline = (df["patient_verdict"] == "BORDERLINE").sum()
    ineligible = (df["patient_verdict"] == "INELIGIBLE").sum()
    blocked    = (df["patient_verdict"] == "BLOCKED").sum()
    flagged    = (df["guardrail_status"] == "flagged").sum()
    passed     = (df["guardrail_status"] == "passed").sum()

    mg_rows = df[df["mg_available"]]
    consensus_rate = (
        mg_rows["consensus_count"].sum() /
        max(mg_rows["consensus_count"].sum() + mg_rows["disagree_count"].sum(), 1)
    ) * 100

    return {
        "total_patients":     total,
        "eligible_count":     int(eligible),
        "borderline_count":   int(borderline),
        "ineligible_count":   int(ineligible),
        "blocked_count":      int(blocked),
        "flagged_count":      int(flagged),
        "passed_count":       int(passed),
        "eligible_rate":      round(eligible / total * 100, 1),
        "borderline_rate":    round(borderline / total * 100, 1),
        "block_rate":         round(blocked / total * 100, 1),
        "flag_rate":          round(flagged / total * 100, 1),
        "avg_retrieved":      round(df["n_retrieved"].mean(), 1),
        "consensus_rate":     round(consensus_rate, 1),
        "mg_available_count": int(mg_rows.shape[0]),
        "input_llm_errors":   int(df["input_llm_error"].sum()),
        "output_llm_errors":  int(df["output_llm_error"].sum()),
        "policy_flags":       int(df["policy_flagged"].sum()),
    }


def compute_condition_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in ["breast_cancer", "diabetes", "other"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        n = len(sub)
        rows.append({
            "Condition":     cond.replace("_", " ").title(),
            "Patients":      n,
            "Eligible":      int((sub["patient_verdict"] == "ELIGIBLE").sum()),
            "Borderline":    int((sub["patient_verdict"] == "BORDERLINE").sum()),
            "Ineligible":    int((sub["patient_verdict"] == "INELIGIBLE").sum()),
            "Blocked":       int((sub["patient_verdict"] == "BLOCKED").sum()),
            "Eligible Rate": f"{(sub['patient_verdict'] == 'ELIGIBLE').sum() / n * 100:.0f}%",
            "Avg Retrieved": round(sub["n_retrieved"].mean(), 1),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bar_layout(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent layout and fix label clipping."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        margin=dict(t=60, b=40, l=40, r=40),
        font=dict(size=12),
    )
    # Add 20% headroom above tallest bar so labels are never clipped
    fig.update_yaxes(automargin=True)
    return fig


def _save(fig: go.Figure, path: str) -> str:
    fig.write_image(path, scale=2)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def chart_verdict_distribution(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    counts = df["patient_verdict"].value_counts().reset_index()
    counts.columns = ["Verdict", "Count"]
    order = ["ELIGIBLE", "BORDERLINE", "INELIGIBLE", "BLOCKED", "FAILED"]
    counts["Verdict"] = pd.Categorical(counts["Verdict"], categories=order, ordered=True)
    counts = counts.sort_values("Verdict")

    fig = px.bar(
        counts, x="Verdict", y="Count",
        color="Verdict",
        color_discrete_map=VERDICT_COLORS,
        title="Overall Verdict Distribution",
        text="Count",
    )
    fig.update_traces(textposition="auto", textfont_size=13)
    fig.update_layout(showlegend=False)
    _bar_layout(fig, height=400)

    # Expand y-axis range to prevent label clipping
    max_count = counts["Count"].max()
    fig.update_yaxes(range=[0, max_count * 1.25])

    path = str(out_dir / "chart_verdict_dist.png")
    _save(fig, path)
    return path, fig


def chart_condition_breakdown(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    cond_map = {"breast_cancer": "Breast Cancer", "diabetes": "Diabetes", "other": "Other"}
    df2 = df.copy()
    df2["Condition"] = df2["condition"].map(cond_map).fillna("Other")

    grouped = (
        df2.groupby(["Condition", "patient_verdict"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        grouped, x="Condition", y="Count",
        color="patient_verdict",
        color_discrete_map=VERDICT_COLORS,
        title="Verdict by Condition",
        barmode="stack",
        text="Count",
    )
    fig.update_traces(textposition="auto", textfont_size=12)
    _bar_layout(fig, height=400)

    path = str(out_dir / "chart_condition_breakdown.png")
    _save(fig, path)
    return path, fig


def chart_guardrail_analysis(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    guard_counts = df["guardrail_status"].value_counts().reset_index()
    guard_counts.columns = ["Status", "Count"]

    color_map = {
        "passed":  "#1ABC9C",
        "flagged": "#F39C12",
        "blocked": "#E74C3C",
        "error":   "#8E44AD",
        "unknown": "#95A5A6",
    }
    fig = px.pie(
        guard_counts, names="Status", values="Count",
        title="Guardrail Status Distribution",
        color="Status",
        color_discrete_map=color_map,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=380, margin=dict(t=60, b=40, l=40, r=40))

    path = str(out_dir / "chart_guardrail.png")
    _save(fig, path)
    return path, fig


def chart_flag_breakdown(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    flag_data = {
        "Input LLM Error":   int(df["input_llm_error"].sum()),
        "Output LLM Error":  int(df["output_llm_error"].sum()),
        "Policy Flagged":    int(df["policy_flagged"].sum()),
        "Grounding Flagged": int(df["grounding_flagged"].sum()),
    }
    flag_df = pd.DataFrame(list(flag_data.items()), columns=["Flag Type", "Count"])

    fig = px.bar(
        flag_df, x="Flag Type", y="Count",
        title="Guardrail Flag Breakdown",
        color_discrete_sequence=[ORANGE],
        text="Count",
    )
    fig.update_traces(textposition="auto", textfont_size=13)
    fig.update_layout(showlegend=False)
    _bar_layout(fig, height=380)

    max_count = flag_df["Count"].max()
    fig.update_yaxes(range=[0, max(max_count * 1.3, 1)])

    path = str(out_dir / "chart_flags.png")
    _save(fig, path)
    return path, fig


def chart_retrieval_stats(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    cond_map = {"breast_cancer": "Breast Cancer", "diabetes": "Diabetes", "other": "Other"}
    df2 = df.copy()
    df2["Condition"] = df2["condition"].map(cond_map).fillna("Other")

    avg_ret = df2.groupby("Condition")["n_retrieved"].mean().reset_index()
    avg_ret.columns = ["Condition", "Avg Trials Retrieved"]
    avg_ret["Avg Trials Retrieved"] = avg_ret["Avg Trials Retrieved"].round(1)

    fig = px.bar(
        avg_ret, x="Condition", y="Avg Trials Retrieved",
        title="Average Trials Retrieved per Patient by Condition",
        color="Condition",
        color_discrete_map={
            "Breast Cancer": CONDITION_COLORS["breast_cancer"],
            "Diabetes":      CONDITION_COLORS["diabetes"],
            "Other":         CONDITION_COLORS["other"],
        },
        text="Avg Trials Retrieved",
    )
    fig.update_traces(textposition="auto", textfont_size=13)
    fig.update_layout(showlegend=False)
    _bar_layout(fig, height=380)

    max_val = avg_ret["Avg Trials Retrieved"].max()
    fig.update_yaxes(range=[0, max_val * 1.3])

    path = str(out_dir / "chart_retrieval.png")
    _save(fig, path)
    return path, fig


def chart_eligible_rate_per_condition(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    rows = []
    for cond in ["breast_cancer", "diabetes", "other"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        n = len(sub)
        rows.append({
            "Condition":    cond.replace("_", " ").title(),
            "Eligible %":   round((sub["patient_verdict"] == "ELIGIBLE").sum() / n * 100, 1),
            "Borderline %": round((sub["patient_verdict"] == "BORDERLINE").sum() / n * 100, 1),
            "Blocked %":    round((sub["patient_verdict"] == "BLOCKED").sum() / n * 100, 1),
        })

    rate_df = pd.DataFrame(rows).melt(
        id_vars="Condition", var_name="Metric", value_name="Rate (%)"
    )
    color_map = {
        "Eligible %":   VERDICT_COLORS["ELIGIBLE"],
        "Borderline %": VERDICT_COLORS["BORDERLINE"],
        "Blocked %":    VERDICT_COLORS["BLOCKED"],
    }

    fig = px.bar(
        rate_df, x="Condition", y="Rate (%)", color="Metric",
        barmode="group",
        title="Eligibility Rates by Condition (%)",
        color_discrete_map=color_map,
        text="Rate (%)",
    )
    fig.update_traces(textposition="auto", textfont_size=11)
    _bar_layout(fig, height=400)
    fig.update_yaxes(range=[0, 120])

    path = str(out_dir / "chart_eligible_rate.png")
    _save(fig, path)
    return path, fig


def chart_other_condition_scope(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    """Validate that 'other' patients get 0 eligible suggestions."""
    other_df = df[df["condition"] == "other"]

    if other_df.empty:
        # No other-condition patients in this run — show a note chart
        fig = go.Figure()
        fig.add_annotation(
            text="No 'other' condition patients in this run.<br>All patients are breast cancer or diabetes.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=DARK_GRAY),
            align="center",
        )
        fig.update_layout(
            title="Other Condition Patients — Scope Validation",
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=60, b=40, l=40, r=40),
        )
        path = str(out_dir / "chart_other_scope.png")
        _save(fig, path)
        return path, fig

    statuses = other_df["patient_verdict"].value_counts().reset_index()
    statuses.columns = ["Verdict", "Count"]

    fig = px.bar(
        statuses, x="Verdict", y="Count",
        color="Verdict",
        color_discrete_map=VERDICT_COLORS,
        title="Other Condition Patients — Verdict Distribution (Should all be BLOCKED)",
        text="Count",
    )
    fig.update_traces(textposition="auto", textfont_size=13)
    fig.update_layout(showlegend=False)
    _bar_layout(fig, height=360)

    max_count = statuses["Count"].max()
    fig.update_yaxes(range=[0, max_count * 1.3])

    path = str(out_dir / "chart_other_scope.png")
    _save(fig, path)
    return path, fig


def chart_subtype_filter_effectiveness(df: pd.DataFrame, out_dir: Path) -> tuple[str, go.Figure]:
    """Show how many trials were eligible vs ineligible per condition."""
    rows = []
    for cond in ["breast_cancer", "diabetes"]:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        rows.append({
            "Condition": cond.replace("_", " ").title(),
            "Avg Eligible Trials":   round(sub["n_eligible"].mean(), 1),
            "Avg Ineligible Trials": round(sub["n_ineligible"].mean(), 1),
        })

    if not rows:
        fig = go.Figure()
        path = str(out_dir / "chart_subtype.png")
        _save(fig, path)
        return path, fig

    sub_df = pd.DataFrame(rows).melt(
        id_vars="Condition", var_name="Type", value_name="Avg Trials"
    )

    fig = px.bar(
        sub_df, x="Condition", y="Avg Trials", color="Type",
        barmode="group",
        title="Avg Eligible vs Ineligible Trials per Patient by Condition",
        color_discrete_sequence=[VERDICT_COLORS["ELIGIBLE"], VERDICT_COLORS["INELIGIBLE"]],
        text="Avg Trials",
    )
    fig.update_traces(textposition="auto", textfont_size=12)
    _bar_layout(fig, height=380)

    path = str(out_dir / "chart_subtype.png")
    _save(fig, path)
    return path, fig


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(
    df: pd.DataFrame,
    metrics: dict,
    run_id: str,
    charts: dict,
    out_path: Path,
) -> None:
    figs_html = "\n".join(
        f'<div class="chart-block">{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>'
        for fig in charts.values()
    )
    cond_table = compute_condition_metrics(df).to_html(index=False, classes="metrics-table")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TrialLink Evaluation Report — {run_id}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; color: #2c3e50; }}
  h1 {{ color: #1ABC9C; }}
  h2 {{ color: #148F77; border-bottom: 2px solid #1ABC9C; padding-bottom: 4px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .kpi {{ background: white; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .kpi .val {{ font-size: 2em; font-weight: bold; color: #1ABC9C; }}
  .kpi .lbl {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
  .chart-block {{ background: white; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .metrics-table {{ width: 100%; border-collapse: collapse; }}
  .metrics-table th {{ background: #1ABC9C; color: white; padding: 8px; }}
  .metrics-table td {{ padding: 8px; border-bottom: 1px solid #eee; }}
  .metrics-table tr:nth-child(even) {{ background: #f9f9f9; }}
  .footer {{ text-align: center; color: #999; margin-top: 40px; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🧬 TrialLink — RAG Pipeline Evaluation Report</h1>
<p><strong>Run ID:</strong> {run_id} &nbsp;|&nbsp; <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Key Metrics</h2>
<div class="kpi-grid">
  <div class="kpi"><div class="val">{metrics['total_patients']}</div><div class="lbl">Total Patients</div></div>
  <div class="kpi"><div class="val">{metrics['eligible_count']}</div><div class="lbl">Eligible</div></div>
  <div class="kpi"><div class="val">{metrics['borderline_count']}</div><div class="lbl">Borderline</div></div>
  <div class="kpi"><div class="val">{metrics['eligible_rate']}%</div><div class="lbl">Eligible Rate</div></div>
  <div class="kpi"><div class="val">{metrics['block_rate']}%</div><div class="lbl">Block Rate</div></div>
  <div class="kpi"><div class="val">{metrics['flag_rate']}%</div><div class="lbl">Guardrail Flag Rate</div></div>
  <div class="kpi"><div class="val">{metrics['avg_retrieved']}</div><div class="lbl">Avg Trials Retrieved</div></div>
  <div class="kpi"><div class="val">{metrics['consensus_rate']}%</div><div class="lbl">Gemini-MedGemma Consensus</div></div>
</div>
<h2>Condition Breakdown</h2>
{cond_table}
<h2>Charts</h2>
{figs_html}
<div class="footer">TrialLink — MLOps Course IE-7374 | Northeastern University | {datetime.now().year}</div>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"  HTML report saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "Title", parent=base["Title"],
            fontSize=22, textColor=colors.HexColor(TEAL),
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"],
            fontSize=11, textColor=colors.HexColor(DARK_GRAY),
            spaceAfter=4, alignment=TA_CENTER,
        ),
        "H1": ParagraphStyle(
            "H1", parent=base["Heading1"],
            fontSize=14, textColor=colors.HexColor(DARK_TEAL),
            spaceBefore=14, spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=9.5, leading=14, spaceAfter=4,
        ),
        "Bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9.5, leading=14,
            leftIndent=16, spaceAfter=3,
        ),
        "Caption": ParagraphStyle(
            "Caption", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor(MID_GRAY),
            alignment=TA_CENTER, spaceAfter=6,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader", parent=base["Normal"],
            fontSize=9, textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def _kpi_table(metrics: dict) -> Table:
    kpis = [
        ("Total Patients",       str(metrics["total_patients"])),
        ("Eligible",             str(metrics["eligible_count"])),
        ("Borderline",           str(metrics["borderline_count"])),
        ("Ineligible",           str(metrics["ineligible_count"])),
        ("Blocked",              str(metrics["blocked_count"])),
        ("Eligible Rate",        f"{metrics['eligible_rate']}%"),
        ("Flag Rate",            f"{metrics['flag_rate']}%"),
        ("Avg Trials Retrieved", str(metrics["avg_retrieved"])),
        ("Consensus Rate",       f"{metrics['consensus_rate']}%"),
        ("Input LLM Errors",     str(metrics["input_llm_errors"])),
        ("Output LLM Errors",    str(metrics["output_llm_errors"])),
        ("Policy Flags",         str(metrics["policy_flags"])),
    ]
    rows = []
    for i in range(0, len(kpis), 3):
        chunk = kpis[i:i+3]
        while len(chunk) < 3:
            chunk.append(("", ""))
        rows.append([
            Paragraph(f"<b>{c[0]}</b>", ParagraphStyle("kl", fontSize=8, textColor=colors.HexColor(DARK_GRAY), alignment=TA_CENTER))
            for c in chunk
        ])
        rows.append([
            Paragraph(f"<font size=16 color={TEAL}><b>{c[1]}</b></font>", ParagraphStyle("kv", fontSize=16, alignment=TA_CENTER))
            for c in chunk
        ])

    t = Table(rows, colWidths=[5.8*cm, 5.8*cm, 5.8*cm])
    t.setStyle(TableStyle([
        ("BOX",          (0,0), (-1,-1), 0.5, colors.HexColor(MID_GRAY)),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, colors.HexColor(MID_GRAY)),
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor(LIGHT_GRAY)),
        ("ROWBACKGROUND",(0,0), (-1,-1), [colors.white, colors.HexColor(LIGHT_GRAY)]),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    return t


def _df_to_table(df: pd.DataFrame, s: dict, col_widths=None) -> Table:
    header = [Paragraph(f"<b>{c}</b>", s["TableHeader"]) for c in df.columns]
    data   = [header]
    for _, row in df.iterrows():
        data.append([
            Paragraph(str(v), ParagraphStyle("tc", fontSize=8.5, alignment=TA_CENTER))
            for v in row
        ])
    if col_widths is None:
        w = 17.0 / len(df.columns)
        col_widths = [w * cm] * len(df.columns)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor(TEAL)),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("ROWBACKGROUND", (0,1), (-1,-1), [colors.white, colors.HexColor(LIGHT_GRAY)]),
        ("BOX",           (0,0), (-1,-1), 0.5, colors.HexColor(MID_GRAY)),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, colors.HexColor(MID_GRAY)),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return t


def _img(path: str, width: float = 16.5) -> Image:
    return Image(path, width=width*cm, height=9*cm, kind="proportional")


def generate_pdf_report(
    df: pd.DataFrame,
    metrics: dict,
    run_id: str,
    chart_paths: dict,
    out_path: Path,
) -> None:
    s   = _styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )
    story = []

    # Cover
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("🧬 TrialLink", s["Title"]),
        Paragraph("RAG Pipeline — Evaluation Report", s["Subtitle"]),
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor(TEAL)),
        Spacer(1, 0.3*cm),
        Paragraph(f"Run ID: <b>{run_id}</b>", s["Subtitle"]),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", s["Subtitle"]),
        Paragraph("MLOps Course IE-7374 | Northeastern University | Khoury College of Computer Sciences", s["Caption"]),
        Spacer(1, 0.8*cm),
    ]

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", s["H1"]))
    story.append(Paragraph(
        f"TrialLink is a clinical trial matching platform that uses a Retrieval-Augmented Generation (RAG) "
        f"pipeline to match patients to relevant clinical trials. This report evaluates the performance of "
        f"the RAG pipeline on <b>{metrics['total_patients']} synthetic patients</b> generated using Synthea, "
        f"covering breast cancer and diabetes conditions. "
        f"The pipeline achieved an <b>eligible rate of {metrics['eligible_rate']}%</b> and a "
        f"<b>borderline rate of {metrics['borderline_rate']}%</b>, with a guardrail flag rate of "
        f"{metrics['flag_rate']}%.",
        s["Body"]
    ))
    story.append(Spacer(1, 0.4*cm))

    # 2. Key Metrics
    story.append(Paragraph("2. Key Performance Metrics", s["H1"]))
    story.append(_kpi_table(metrics))
    story.append(Spacer(1, 0.4*cm))

    # 3. Dataset
    story.append(Paragraph("3. Dataset", s["H1"]))
    story.append(Paragraph(
        "Synthetic patients were generated using the Synthea patient generator and parsed using "
        "the TrialLink FHIRParser. Patients are categorized by their primary condition:",
        s["Body"]
    ))
    cond_df = compute_condition_metrics(df)
    story.append(_df_to_table(cond_df, s))
    story.append(Paragraph(
        "Note: 'Other' condition patients should receive zero trial suggestions — "
        "TrialLink only supports breast cancer and diabetes.",
        s["Caption"]
    ))
    story.append(Spacer(1, 0.4*cm))

    # 4. Verdict Distribution
    story.append(Paragraph("4. Verdict Distribution", s["H1"]))
    story.append(Paragraph(
        "Each patient is assigned an overall verdict: ELIGIBLE (at least one trial fully matched), "
        "BORDERLINE (partial match requiring clinician review), or BLOCKED (guardrail intervention).",
        s["Body"]
    ))
    if "verdict_dist" in chart_paths:
        story.append(_img(chart_paths["verdict_dist"]))
        story.append(Paragraph("Figure 1: Overall verdict distribution across all patients.", s["Caption"]))
    if "condition_breakdown" in chart_paths:
        story.append(_img(chart_paths["condition_breakdown"]))
        story.append(Paragraph("Figure 2: Verdict distribution broken down by patient condition.", s["Caption"]))
    story.append(PageBreak())

    # 5. Condition Segregation
    story.append(Paragraph("5. Condition Segregation Validation", s["H1"]))
    story.append(Paragraph(
        "TrialLink enforces that patients with unsupported conditions receive no trial suggestions. "
        "The validate_trials_scope() guardrail and condition-aware Firestore collection filtering "
        "enforce this at both retrieval and output stages.",
        s["Body"]
    ))
    if "other_scope" in chart_paths:
        story.append(_img(chart_paths["other_scope"]))
        story.append(Paragraph("Figure 3: Scope validation for non-supported condition patients.", s["Caption"]))
    story.append(Spacer(1, 0.4*cm))

    # 6. Guardrail Analysis
    story.append(Paragraph("6. Guardrail Analysis", s["H1"]))
    story.append(Paragraph(
        "The pipeline implements a multi-layer guardrail system: PII redaction, structural "
        "validation, LLM input/output judges, policy checks, and grounding verification.",
        s["Body"]
    ))
    if "guardrail" in chart_paths:
        story.append(_img(chart_paths["guardrail"]))
        story.append(Paragraph("Figure 4: Guardrail status distribution (passed / flagged / blocked).", s["Caption"]))
    if "flags" in chart_paths:
        story.append(_img(chart_paths["flags"]))
        story.append(Paragraph("Figure 5: Guardrail flag breakdown by type.", s["Caption"]))

    for note in [
        f"<b>Input LLM Guardrail Errors:</b> {metrics['input_llm_errors']} — caused by JSON truncation in guardrail response.",
        f"<b>Output LLM Guardrail Errors:</b> {metrics['output_llm_errors']} — same root cause.",
        f"<b>Policy Flags:</b> {metrics['policy_flags']} — triggered by medication name patterns.",
    ]:
        story.append(Paragraph(f"• {note}", s["Bullet"]))
    story.append(PageBreak())

    # 7. Retrieval Quality
    story.append(Paragraph("7. Retrieval Quality", s["H1"]))
    story.append(Paragraph(
        "The hybrid retrieval pipeline uses LLM-based clinical context enrichment to build targeted "
        "search queries, Vertex AI Vector Search (text-embedding-005) for dense retrieval, "
        "condition-aware Firestore collection filtering using the engineered 'disease' field, "
        "and Vertex AI Ranking API for reranking. This eliminates cross-condition contamination "
        "(breast cancer trials appearing for diabetes patients).",
        s["Body"]
    ))
    if "retrieval" in chart_paths:
        story.append(_img(chart_paths["retrieval"]))
        story.append(Paragraph("Figure 6: Average number of trials retrieved per patient by condition.", s["Caption"]))
    if "eligible_rate" in chart_paths:
        story.append(_img(chart_paths["eligible_rate"]))
        story.append(Paragraph("Figure 7: Eligibility rates (%) broken down by condition.", s["Caption"]))
    if "subtype" in chart_paths:
        story.append(_img(chart_paths["subtype"]))
        story.append(Paragraph("Figure 8: Average eligible vs ineligible trials per patient by condition.", s["Caption"]))
    story.append(Spacer(1, 0.4*cm))

    # 8. Dual LLM Judge
    story.append(Paragraph("8. Dual LLM Judge — Gemini vs MedGemma", s["H1"]))
    story.append(Paragraph(
        "TrialLink uses a dual-judge architecture. Gemini 2.5 Flash generates the primary eligibility "
        "recommendation. MedGemma (a medical-domain specialist LLM deployed on Vertex AI) acts as an "
        "independent second-opinion judge. A trial is only shown to the patient if BOTH judges agree "
        "it is ELIGIBLE or BORDERLINE.",
        s["Body"]
    ))
    mg_available = metrics.get("mg_available_count", 0)
    story.append(Paragraph(
        f"• MedGemma was available for <b>{mg_available}</b> of {metrics['total_patients']} patients in this run.",
        s["Bullet"]
    ))
    story.append(Paragraph(
        f"• Gemini-MedGemma consensus rate: <b>{metrics['consensus_rate']}%</b>",
        s["Bullet"]
    ))
    story.append(Paragraph(
        "• MedGemma dedicated endpoints are only accessible within the GCP VPC. "
        "Local evaluation uses Gemini-only fallback. MedGemma is validated separately via Cloud Shell.",
        s["Bullet"]
    ))
    story.append(Spacer(1, 0.4*cm))

    # 9. Per-Patient Results
    story.append(Paragraph("9. Per-Patient Results", s["H1"]))
    display_cols = ["patient_id", "condition", "patient_verdict", "guardrail_status",
                    "n_retrieved", "n_eligible", "n_borderline"]
    table_df = df[display_cols].copy()
    table_df.columns = ["Patient ID", "Condition", "Verdict", "Guardrail",
                        "Retrieved", "Eligible", "Borderline"]
    story.append(_df_to_table(
        table_df, s,
        col_widths=[2.8*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.0*cm, 2.0*cm]
    ))
    story.append(PageBreak())



    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(MID_GRAY)))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "TrialLink — MLOps Course IE-7374 | Northeastern University | Khoury College of Computer Sciences",
        s["Caption"]
    ))
    story.append(Paragraph(
        f"Report generated automatically by generate_eval_report.py | Run ID: {run_id}",
        s["Caption"]
    ))

    doc.build(story)
    print(f"  PDF report saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TrialLink Evaluation Report Generator")
    parser.add_argument("--run_id",      type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="tests/evaluation_1/results")
    args = parser.parse_args()

    results_root = Path(args.results_dir)

    if args.run_id:
        run_dir = results_root / args.run_id
    else:
        run_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
        if not run_dirs:
            print("No run directories found.")
            sys.exit(1)
        run_dir = run_dirs[-1]

    run_id = run_dir.name
    print(f"\nGenerating evaluation report for run: {run_id}")
    print(f"  Loading results from: {run_dir}")

    report_dir = results_root.parent / "reports"
    chart_dir  = report_dir / "charts" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(run_dir)
    print(f"  Loaded {len(results)} patient results")

    if not results:
        print("No results found. Exiting.")
        sys.exit(1)

    df      = build_dataframe(results)
    metrics = compute_summary_metrics(df)

    print(f"\n  Summary:")
    print(f"    Total    : {metrics['total_patients']}")
    print(f"    Eligible : {metrics['eligible_count']} ({metrics['eligible_rate']}%)")
    print(f"    Borderline: {metrics['borderline_count']}")
    print(f"    Blocked  : {metrics['blocked_count']}")
    print(f"    Flag rate: {metrics['flag_rate']}%")

    print("\n  Generating charts...")
    chart_paths = {}
    chart_figs  = {}

    generators = {
        "verdict_dist":        chart_verdict_distribution,
        "condition_breakdown": chart_condition_breakdown,
        "guardrail":           chart_guardrail_analysis,
        "flags":               chart_flag_breakdown,
        "retrieval":           chart_retrieval_stats,
        "eligible_rate":       chart_eligible_rate_per_condition,
        "other_scope":         chart_other_condition_scope,
        "subtype":             chart_subtype_filter_effectiveness,
    }

    for key, fn in generators.items():
        try:
            path, fig = fn(df, chart_dir)
            chart_paths[key] = path
            chart_figs[key]  = fig
            print(f"    ✓ {key}")
        except Exception as e:
            print(f"    ✗ {key}: {e}")

    print("\n  Generating HTML report...")
    html_path = report_dir / f"eval_report_{run_id}.html"
    generate_html_report(df, metrics, run_id, chart_figs, html_path)

    print("\n  Generating PDF report...")
    pdf_path = report_dir / f"eval_report_{run_id}.pdf"
    generate_pdf_report(df, metrics, run_id, chart_paths, pdf_path)

    print(f"\n{'='*60}")
    print(f"  PDF  : {pdf_path}")
    print(f"  HTML : {html_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()