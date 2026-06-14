"""streamlit_app.py — entry point for Streamlit Community Cloud.

Runs the full LangGraph pipeline inline (no FastAPI needed). For local
development against the API instead, use ``src/dashboard/app.py``.
"""
import os
import sys
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# On Streamlit Cloud, lift the LLM key out of st.secrets into the environment.
for _k in ("GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    try:
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
    except Exception:
        pass

import plotly.graph_objects as go

from src.graph.workflow import app_graph
from src.demo_presets import PRESETS

st.set_page_config(page_title="CredAgent — Credit Decisioning", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def _seed_once():
    """Seed historical decisions so the monitoring view (drift + fair-lending)
    has realistic data on a fresh Streamlit Cloud deploy. Runs once per process."""
    try:
        from src.seed_history import ensure_seeded
        return ensure_seeded()
    except Exception:
        return 0


_seed_once()

st.markdown("""
<style>
 .decision-approve{background:#d4edda;color:#155724;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #28a745;}
 .decision-decline{background:#f8d7da;color:#721c24;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #dc3545;}
 .decision-refer{background:#fff3cd;color:#856404;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #ffc107;}
 .adverse-notice{background:#f8f9fa;padding:16px;border-radius:8px;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;border:1px solid #dee2e6;}
 .threshold-line{font-size:1.02rem;margin:14px 0 6px;color:#1f2d3d;}
 .factor-card{background:#fff;border:1px solid #e6e9ef;border-left:4px solid #dc3545;border-radius:8px;padding:11px 14px;margin-bottom:9px;}
 .factor-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;}
 .factor-name{font-weight:700;color:#2a2f37;font-size:0.96rem;}
 .factor-val{font-weight:700;color:#dc3545;font-size:0.96rem;white-space:nowrap;}
 .factor-bar{height:5px;background:#f0e3e4;border-radius:3px;margin:8px 0 5px;overflow:hidden;}
 .factor-fill{height:100%;background:#dc3545;border-radius:3px;}
 .factor-impact{font-size:0.74rem;color:#8a929c;}
 .policy-card{background:#f8f9fb;border:1px solid #e6e9ef;border-radius:8px;padding:12px 16px;margin-bottom:10px;}
 .policy-src{font-size:0.72rem;font-weight:700;letter-spacing:.4px;text-transform:uppercase;color:#5b6b80;margin-bottom:6px;}
 .policy-body{font-size:0.9rem;line-height:1.55;color:#2f3742;}
 .policy-body p{margin:0 0 8px;}
 .rec-header{font-size:1.05rem;font-weight:700;margin:18px 0 6px;color:#1f6f54;}
 .rec-card{background:#f3fbf7;border:1px solid #d7efe4;border-left:4px solid #1f9d6b;border-radius:8px;padding:12px 15px;margin-bottom:9px;}
 .rec-card.rec-positive{background:#eef7ff;border-left-color:#2f7fe0;}
 .rec-title{font-weight:700;color:#1f6f54;font-size:0.96rem;margin-bottom:3px;}
 .rec-card.rec-positive .rec-title{color:#1f5fae;}
 .rec-body{font-size:0.9rem;line-height:1.55;color:#2f3742;}
 .audit-footer{margin-top:14px;font-size:0.72rem;color:#aeb6c2;letter-spacing:.2px;}
</style>""", unsafe_allow_html=True)


def _empty_state(applicant_id, raw):
    return {
        "applicant_id": applicant_id, "raw_application": raw,
        "request_timestamp": datetime.now().isoformat(),
        "cleaned_features": None, "derived_features": None,
        "ingestion_errors": [], "ingestion_complete": False,
        "risk_probability": None, "risk_tier": None, "shap_values": None,
        "top_risk_factors": None, "model_version": None,
        "compliance_flags": [], "retrieved_policy_excerpts": [], "policy_check_complete": False,
        "requires_human_review": False, "human_decision": None, "human_notes": None,
        "final_decision": None, "credit_limit": None, "decision_reasoning": None,
        "decision_confidence": None, "adverse_action_notice": None,
        "audit_trail": [], "processing_time_ms": None, "final_response_packaged": False,
        "llm_calls": [],
    }


def run_pipeline(payload: dict, human_decision: Optional[str] = None, human_notes: str = "") -> dict:
    config = {"configurable": {"thread_id": payload["applicant_id"]}}
    start = time.time()
    if human_decision is None:
        result = app_graph.invoke(_empty_state(payload["applicant_id"], payload), config=config)
    else:
        result = app_graph.invoke(
            {"human_decision": human_decision, "human_notes": human_notes,
             "requires_human_review": False},
            config=config,
        )
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    # Persist so the live demo's drift + fair-lending monitoring reflect real
    # decisions (the inline pipeline, unlike the API, must log explicitly).
    try:
        from src import store
        store.log_decision(result, source="demo")
        # Count only NEW decisions this session (not the seeded baseline), so the
        # monitoring panel gives immediate, legible feedback on a fresh submit.
        if human_decision is None:
            st.session_state["session_decisions"] = st.session_state.get("session_decisions", 0) + 1
    except Exception:  # pragma: no cover - persistence is non-fatal to the demo
        pass
    return result


SHAP_DISPLAY = {
    "EXT_SOURCE_1": "Ext Credit Score 1", "EXT_SOURCE_2": "Ext Credit Score 2",
    "EXT_SOURCE_3": "Ext Credit Score 3", "debt_to_income": "Debt-to-Income",
    "credit_to_income_ratio": "Credit/Income Ratio", "employment_months": "Employment Length",
    "age_years": "Applicant Age", "AMT_CREDIT": "Credit Amount",
    "AMT_INCOME_TOTAL": "Annual Income", "CNT_CHILDREN": "Number of Children",
    "annuity_to_credit_ratio": "Annuity/Credit Ratio", "has_income_stability": "Income Stability",
    "NAME_INCOME_TYPE_Working": "Income: Working", "AMT_ANNUITY": "Monthly Payment",
    "DAYS_EMPLOYED": "Days Employed", "FLAG_OWN_CAR": "Owns Car", "FLAG_OWN_REALTY": "Owns Realty",
}


def render_shap_waterfall(shap_values: dict):
    """Interactive Plotly bar (renders client-side — no matplotlib/server freeze)."""
    if not shap_values:
        st.info("SHAP values not available.")
        return
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10][::-1]
    labels = [SHAP_DISPLAY.get(f, f.replace("_", " ").title()) for f, _ in items]
    values = [round(v, 4) for _, v in items]
    colors = ["#dc3545" if v > 0 else "#28a745" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Feature Impact on Risk Score",
        xaxis_title="SHAP value — impact on log-odds of default (model margin)",
        height=440, margin=dict(l=10, r=10, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0, line_width=1, line_color="#888")
    st.plotly_chart(fig, use_container_width=True)


def render_decision_badge(decision: Optional[str]):
    css = {"APPROVE": "decision-approve", "DECLINE": "decision-decline",
           "REFER": "decision-refer"}.get(decision, "decision-refer")
    icon = {"APPROVE": "✅", "DECLINE": "❌", "REFER": "⚠️"}.get(decision, "⏳")
    st.markdown(f'<div class="{css}">{icon} &nbsp; {decision or "AWAITING REVIEW"}</div>',
                unsafe_allow_html=True)


# Risk-tier thresholds (must match src/agents/risk_scoring_agent.py).
TIER_THRESHOLDS = {"LOW": 0.30, "MEDIUM": 0.55, "HIGH": 0.75}
DECLINE_THRESHOLD = 0.75


def _format_feature_value(feat: str, v) -> str:
    """Human-readable rendering of an engineered feature value."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if feat.startswith("EXT_SOURCE"):
        return f"{v:.2f} / 1.00"
    if feat in ("debt_to_income", "credit_to_income_ratio", "annuity_to_credit_ratio"):
        return f"{v:.2f}"
    if feat.startswith("AMT_"):
        return f"${v:,.0f}"
    if feat in ("employment_months",):
        return f"{v:.0f} months"
    if feat in ("age_years",):
        return f"{v:.0f} years"
    if feat in ("CNT_CHILDREN",):
        return f"{int(v)}"
    if feat.startswith("FLAG_") or feat.startswith("has_") or feat.startswith("NAME_"):
        return "Yes" if v >= 0.5 else "No"
    return f"{v:.2f}"


# Neutral, analyst-facing feature names (no "High/Limited" qualifier — the actual
# value + impact convey direction). The qualifier-laden FEATURE_DISPLAY_NAMES are
# reserved for the legal adverse-action notice phrasing.
NEUTRAL_FACTOR_NAMES = {
    "EXT_SOURCE_1": "External credit score (bureau 1)",
    "EXT_SOURCE_2": "External credit score (bureau 2)",
    "EXT_SOURCE_3": "External credit score (bureau 3)",
    "debt_to_income": "Debt-to-income ratio",
    "credit_to_income_ratio": "Credit-to-income ratio",
    "annuity_to_credit_ratio": "Annuity-to-credit ratio",
    "employment_months": "Employment history",
    "AMT_ANNUITY": "Monthly payment obligation",
    "AMT_CREDIT": "Credit amount requested",
    "AMT_INCOME_TOTAL": "Income level",
    "CNT_CHILDREN": "Number of dependents",
    "has_income_stability": "Income-type stability",
    "NAME_INCOME_TYPE_Working": "Employment income status",
}


def _neutral_name(feat: str) -> str:
    if feat in NEUTRAL_FACTOR_NAMES:
        return NEUTRAL_FACTOR_NAMES[feat]
    from src.ml.explainer import FEATURE_DISPLAY_NAMES
    name = FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())
    return _re.sub(r"^(High|Low|Limited)\s+", "", name).capitalize()


def build_adverse_factors(r: dict, top_n: int = 4):
    """Connect each adverse factor (the 'why') to its actual value and the SHAP
    impact that drove it (the 'what'). Returns a list of dicts; protected-basis
    features are never surfaced as reasons."""
    try:
        from src.ml.explainer import PROTECTED_FROM_ADVERSE_REASONS
        from src.ml.features import engineer_single_application
    except Exception:
        return []
    shap = r.get("shap_values") or {}
    raw = r.get("raw_application") or {}
    try:
        feats = engineer_single_application(raw)
    except Exception:
        feats = {}
    out = []
    for feat, val in sorted(shap.items(), key=lambda x: x[1], reverse=True):
        if val <= 0 or feat in PROTECTED_FROM_ADVERSE_REASONS:
            continue
        out.append({
            "feature": feat,
            "name": _neutral_name(feat),
            "value": _format_feature_value(feat, feats.get(feat)),
            "impact": float(val),
        })
        if len(out) >= top_n:
            break
    return out


import html as _html
import re as _re

_SRC_RE = _re.compile(r"^\s*\[Source:\s*([^,\]]+?)(?:,\s*Page:\s*(\d+))?\s*\]\s*", _re.IGNORECASE)


def _prettify_source(filename: str) -> str:
    """Turn a raw corpus filename into a professional citation label."""
    stem = _re.sub(r"\.(txt|pdf|md)$", "", filename.strip(), flags=_re.IGNORECASE)
    words = stem.replace("-", " ").replace("_", " ").split()
    acronyms = {"cfpb": "CFPB", "ecoa": "ECOA", "fcra": "FCRA", "reg": "Reg.", "b": "B"}
    pretty = " ".join(acronyms.get(w.lower(), w.capitalize()) for w in words)
    return pretty or filename


def _split_excerpt(ex: str):
    """Parse '[Source: file, Page: n] body' into (citation_html, body_html).

    Returns the FULL body (no truncation) with a clean citation. Page numbers are
    only shown for paginated (PDF) sources, where they are meaningful."""
    m = _SRC_RE.match(ex)
    if m:
        fname, page = m.group(1), m.group(2)
        body = ex[m.end():].strip()
        label = _prettify_source(fname)
        if page is not None and fname.lower().endswith(".pdf"):
            label += f" · p.{page}"
    else:
        label, body = "CFPB Fair-Lending Corpus", ex.strip()
    body = _html.escape(body)
    # Collapse hard-wrapped lines into readable paragraphs.
    body = _re.sub(r"\n{2,}", "</p><p>", body)
    body = body.replace("\n", " ")
    return _html.escape(label), f"<p>{body}</p>"


def _recommend(factor: dict):
    """Map a critical adverse factor to empowering, actionable guidance — the
    'financial advisor' layer. Returns (title, advice) or None for factors that
    are not constructively actionable. Advice references the observed value but
    never promises approval (compliance-safe phrasing)."""
    feat = factor.get("feature", "")
    val = factor.get("value", "—")
    if feat.startswith("EXT_SOURCE"):
        return ("Strengthen your credit-bureau profile",
                f"This external credit score ({val}) is the strongest driver of the decision. "
                "Consistent on-time payments, lower credit-card utilization, and a longer credit "
                "history are the most effective ways to raise it over time.")
    if feat == "debt_to_income":
        return ("Lower your debt-to-income ratio",
                f"Your debt-to-income is currently {val}. Lenders generally look for 0.36 or below — "
                "paying down existing balances or increasing documented income moves you toward that range.")
    if feat == "credit_to_income_ratio":
        return ("Right-size the loan to your income",
                f"The requested amount is large relative to income (ratio {val}). A smaller principal, "
                "or documenting additional income, would better align the request with typical approvals.")
    if feat == "annuity_to_credit_ratio":
        return ("Ease the repayment intensity",
                f"The scheduled payment is high relative to the loan size (ratio {val}). A longer "
                "repayment term or a smaller principal would reduce this.")
    if feat == "AMT_ANNUITY":
        return ("Reduce the monthly payment",
                f"The monthly payment of {val} weighs on affordability. Extending the term or borrowing "
                "slightly less would lower it.")
    if feat == "AMT_CREDIT":
        return ("Consider a more modest amount",
                f"The requested credit of {val} increases exposure. Requesting a smaller amount would "
                "improve the affordability picture.")
    if feat == "AMT_INCOME_TOTAL":
        return ("Document additional income",
                f"Higher or better-documented income (currently {val}) would strengthen demonstrated "
                "capacity to repay.")
    if feat == "employment_months":
        return ("Build employment tenure",
                f"A longer continuous employment record ({val}) signals stability — remaining in role helps.")
    if feat in ("has_income_stability", "NAME_INCOME_TYPE_Working", "NAME_INCOME_TYPE_Commercial_associate"):
        return ("Stabilize your income source",
                "A steady, well-documented primary income source strengthens the overall profile.")
    return None


def _threshold_context(decision: Optional[str], prob: Optional[float]) -> str:
    """Plain-English statement tying the probability to the threshold it crossed."""
    if prob is None:
        return "No probability was produced for this application."
    pct = f"{prob:.1%}"
    if decision == "DECLINE":
        return f"Default probability <strong>{pct}</strong> is at or above the <strong>{DECLINE_THRESHOLD:.0%} auto-decline</strong> threshold."
    if decision == "REFER":
        return (f"Default probability <strong>{pct}</strong> falls in the <strong>{TIER_THRESHOLDS['LOW']:.0%}–{TIER_THRESHOLDS['MEDIUM']:.0%} "
                f"review band</strong> — routed to a human underwriter.")
    if decision == "APPROVE":
        return f"Default probability <strong>{pct}</strong> is below the <strong>{TIER_THRESHOLDS['LOW']:.0%} auto-approve</strong> threshold."
    return f"Default probability <strong>{pct}</strong>."


def render_system_health():
    """Render the live monitoring expander. Called into a sidebar placeholder
    AFTER the decision pipeline runs, so a fresh submit updates it in real time
    (the sidebar block itself executes before the submit handler)."""
    with st.expander("🩺 System Health & Monitoring", expanded=True):
        try:
            from src import store
            from src.drift import drift_report
            summ = store.summary()
            drift = drift_report()
            fl = summ.get("fair_lending", {})
            st.caption("Live model-risk & fair-lending monitoring")
            total = summ.get("total", 0)
            session_n = st.session_state.get("session_decisions", 0)
            seeded = max(total - session_n, 0)
            st.metric("Decisions this session", session_n,
                      help="Applications you have scored since opening the app.")
            st.caption(f"On record: **{total:,}** total "
                       f"({seeded:,} seeded baseline + {session_n:,} this session). "
                       "Monitoring below aggregates all of them.")
            ds = drift.get("overall_status", "no-reference")
            dico = {"stable": "🟢", "moderate": "🟡", "significant": "🔴"}.get(ds, "⚪")
            st.markdown(f"**Model drift (PSI):** {dico} {ds} · max {drift.get('max_psi', 0)}")
            ratio = fl.get("adverse_impact_ratio")
            if ratio is not None:
                flico = "🔴" if fl.get("flag") else "🟢"
                st.markdown(f"**Fair-lending (4/5 rule):** {flico} ratio {ratio} "
                            f"{'— below 0.80, investigate' if fl.get('flag') else '— within tolerance'}")
            else:
                st.markdown("**Fair-lending (4/5 rule):** ⚪ accumulating data")
            last = st.session_state.get("last_result", {})
            if last.get("processing_time_ms"):
                st.markdown(f"**Last decision latency:** {last['processing_time_ms']:.0f} ms")
            st.caption("PSI <0.10 stable · 0.10–0.25 investigate · >0.25 retrain. "
                       "Sex is excluded from the model and used only for this disparate-impact test.")
        except Exception as exc:
            st.caption(f"Monitoring unavailable: {exc}")


with st.sidebar:
    st.title("🏦 CredAgent")
    st.caption("Agentic Credit Decisioning System")
    st.markdown("---")
    st.markdown("**Decision Policy**")
    st.markdown("""
| Risk band | Default prob. | Action |
|---|---|---|
| 🟢 LOW | < 30% | Auto-approve |
| 🟡 MEDIUM | 30–55% | Human review (HITL) |
| 🔴 HIGH / DECLINE | ≥ 55% | Auto-decline + notice |
""")
    st.caption("HIGH (55–75%) and DECLINE (≥75%) are reported as separate risk "
               "tiers for monitoring, but both take the same action — auto-decline "
               "with an adverse-action notice.")
    st.markdown("---")
    st.markdown("**Quick Test Cases** *(real applicants)*")
    if st.button("📗 Low Risk"):
        st.session_state["prefill"] = "low"
    if st.button("📙 Medium Risk"):
        st.session_state["prefill"] = "medium"
    if st.button("📕 High Risk"):
        st.session_state["prefill"] = "high"
    st.markdown("---")
    # Placeholder rendered into AFTER the submit handler (end of script) so a new
    # decision updates these metrics in the same run — no extra click needed.
    sys_health_slot = st.empty()
    st.caption("LangGraph · XGBoost · SHAP · ChromaDB · Groq")

preset = PRESETS.get(st.session_state.get("prefill"), {})

st.title("Applicant Credit Assessment")
st.caption("Real-time agentic risk decisioning for employer-sponsored installment lending.")

with st.form("app_form"):
    c1, c2 = st.columns(2)
    with c1:
        applicant_id = st.text_input("Applicant ID", value=preset.get("applicant_id", ""), placeholder="APP-001")
        amt_income = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0, value=float(preset.get("amt_income_total", 150000.0)))
        amt_credit = st.number_input("Requested Credit ($)", min_value=0.0, step=500.0, value=float(preset.get("amt_credit", 250000.0)))
        amt_annuity = st.number_input("Monthly Payment ($)", min_value=0.0, step=10.0, value=float(preset.get("amt_annuity", 20000.0)))
        cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=int(preset.get("cnt_children", 0)))
    with c2:
        days_birth = st.number_input("Days Since Birth (negative)", value=int(preset.get("days_birth", -12000)))
        days_employed = st.number_input("Days Employed (negative)", value=int(preset.get("days_employed", -2000)))
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, float(preset.get("ext_source_1", 0.60)), 0.01)
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, float(preset.get("ext_source_2", 0.58)), 0.01)
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, float(preset.get("ext_source_3", 0.55)), 0.01)

    c3, c4 = st.columns(2)
    with c3:
        code_gender = st.selectbox("Gender", ["M", "F", "X"], index=["M", "F", "X"].index(preset.get("code_gender", "F")))
        st.caption("⚖️ Collected for fair-lending monitoring only — **excluded from the model** (sex is a prohibited basis under ECOA).")
        flag_own_car = st.checkbox("Owns Car", value=bool(preset.get("flag_own_car", 0)))
        flag_own_realty = st.checkbox("Owns Realty", value=bool(preset.get("flag_own_realty", 0)))
    with c4:
        income_types = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"]
        edu_types = ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"]
        name_income_type = st.selectbox("Income Type", income_types, index=income_types.index(preset.get("name_income_type", "Working")))
        name_education_type = st.selectbox("Education Level", edu_types, index=edu_types.index(preset.get("name_education_type", "Secondary / secondary special")))
        st.caption("⚖️ Education is **excluded from the model** — a documented proxy for race/national origin under CFPB disparate-impact guidance.")

    submitted = st.form_submit_button("🚀 Submit for Decisioning", type="primary", use_container_width=True)

if submitted:
    if not applicant_id.strip():
        st.error("Applicant ID is required.")
    else:
        payload = {
            "applicant_id": applicant_id.strip(),
            "amt_credit": amt_credit, "amt_income_total": amt_income, "amt_annuity": amt_annuity,
            "days_birth": int(days_birth), "days_employed": int(days_employed),
            "ext_source_1": ext_source_1, "ext_source_2": ext_source_2, "ext_source_3": ext_source_3,
            "code_gender": code_gender, "flag_own_car": int(flag_own_car),
            "flag_own_realty": int(flag_own_realty), "cnt_children": int(cnt_children),
            "name_income_type": name_income_type, "name_education_type": name_education_type,
        }
        with st.spinner("Running agentic decisioning pipeline..."):
            st.session_state["last_result"] = run_pipeline(payload)
        st.session_state.pop("prefill", None)

if "last_result" in st.session_state:
    r = st.session_state["last_result"]
    st.markdown("---")
    st.subheader("Decision Result")
    render_decision_badge(r.get("final_decision"))

    prob = r.get("risk_probability"); limit = r.get("credit_limit"); ms = r.get("processing_time_ms")
    decision = r.get("final_decision")

    # ── Primary hierarchy: the decision + the threshold it crossed, then the
    #    plain-English rationale. Raw metrics are demoted to "supporting data".
    st.markdown(f"<div class='threshold-line'>{_threshold_context(decision, prob)}</div>",
                unsafe_allow_html=True)
    if r.get("decision_reasoning"):
        st.info(r["decision_reasoning"])

    t1, t2, t3, t4, t5 = st.tabs(["📋 Decision", "📊 SHAP", "⚖️ Compliance", "📄 Adverse Notice", "🔍 Audit"])
    with t1:
        factors = build_adverse_factors(r)
        approve = decision == "APPROVE"
        if factors and not approve:
            st.markdown("##### 🚩 Critical adverse factors")
            st.caption("The specific reasons that drove this decision — value observed and its "
                       "impact on the model's risk score. These are the disclosable adverse-action reasons.")
            max_impact = max(f["impact"] for f in factors) or 1.0
            for i, f in enumerate(factors, 1):
                width = max(6, round(100 * f["impact"] / max_impact))
                st.markdown(
                    f"<div class='factor-card'>"
                    f"<div class='factor-head'><span class='factor-name'>{i}. {f['name']}</span>"
                    f"<span class='factor-val'>{f['value']}</span></div>"
                    f"<div class='factor-bar'><div class='factor-fill' style='width:{width}%'></div></div>"
                    f"<div class='factor-impact'>risk impact +{f['impact']:.3f} (log-odds)</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        elif approve:
            st.success("✅ No adverse factors outweighed the applicant's repayment indicators.")
            pos = build_adverse_factors({**r, "shap_values": {k: -v for k, v in (r.get("shap_values") or {}).items()}})
            if pos:
                st.markdown("##### Strongest approving factors")
                for i, f in enumerate(pos, 1):
                    st.markdown(f"**{i}. {f['name']}** — {f['value']}  ·  lowers risk by {f['impact']:.3f}")
        else:
            st.caption("No single dominant adverse factor was identified.")

        # ── Recommendations: turn the adverse factors into financial-advisor
        #    guidance (declines/refers) or strengths to maintain (approvals).
        st.markdown("<div class='rec-header'>💡 Recommendations</div>", unsafe_allow_html=True)
        if approve:
            st.markdown(
                "<div class='rec-card rec-positive'><div class='rec-title'>✅ Approved — strong profile</div>"
                "<div class='rec-body'>This application meets the approval criteria"
                + (f", with a recommended limit of <strong>${limit:,.0f}</strong>." if limit else ".")
                + " To preserve or grow this limit, keep payments on time and income documentation current.</div></div>",
                unsafe_allow_html=True)
        else:
            recs = [x for x in (_recommend(f) for f in factors) if x][:3]
            if recs:
                st.caption("Good-faith steps that would strengthen a future application. "
                           "Illustrative guidance — not a guarantee of approval.")
                for title, advice in recs:
                    st.markdown(
                        f"<div class='rec-card'><div class='rec-title'>💡 {title}</div>"
                        f"<div class='rec-body'>{advice}</div></div>",
                        unsafe_allow_html=True)
            else:
                st.caption("No constructively actionable factors were isolated for this decision.")

        # Subtle, de-emphasised audit footer — traceability without clutter.
        prob_txt = f"{prob*100:.1f}%" if prob is not None else "n/a"
        st.markdown(
            f"<div class='audit-footer'>Audit trail · model {r.get('model_version', 'n/a')} · "
            f"risk tier {r.get('risk_tier', '—')} @ {prob_txt} · "
            f"processed in {ms:.0f} ms</div>" if ms is not None else
            f"<div class='audit-footer'>Audit trail · model {r.get('model_version', 'n/a')} · "
            f"risk tier {r.get('risk_tier', '—')} @ {prob_txt}</div>",
            unsafe_allow_html=True)
    with t2:
        render_shap_waterfall(r.get("shap_values") or {})
        st.caption("Values are SHAP contributions to the model's log-odds (margin); a higher "
                   "log-odds means a higher default probability. Red = increases default risk · "
                   "Green = decreases default risk.")
    with t3:
        flags = r.get("compliance_flags", [])
        if flags:
            st.warning(f"⚠️ {len(flags)} compliance flag(s)")
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ No compliance issues detected.")
        excerpts = r.get("retrieved_policy_excerpts", [])
        if excerpts:
            st.markdown("##### 📚 CFPB policy evidence retrieved")
            st.caption("The fair-lending passages the PolicyComplianceAgent retrieved (RAG) and "
                       "grounded its check in — part of the audit trail.")
            for ex in excerpts:
                src, body = _split_excerpt(ex)
                st.markdown(
                    f"<div class='policy-card'>"
                    f"<div class='policy-src'>{src}</div>"
                    f"<div class='policy-body'>{body}</div></div>",
                    unsafe_allow_html=True,
                )
    with t4:
        notice = r.get("adverse_action_notice")
        if notice:
            st.markdown(f'<div class="adverse-notice">{notice}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Download Notice", data=notice,
                               file_name=f"adverse_{r.get('applicant_id', 'applicant')}.txt", mime="text/plain")
        elif r.get("final_decision") == "APPROVE":
            st.success("No adverse action notice required.")
        else:
            st.info("Adverse notice appears here for declined applications.")
    with t5:
        for entry in r.get("audit_trail", []):
            st.code(entry, language=None)

    if r.get("requires_human_review"):
        st.markdown("---")
        st.subheader("👤 Human Review Required")
        st.warning("Borderline risk score or compliance flag detected. Manual review needed.")
        with st.form("hr_form"):
            hr_dec = st.radio("Your Decision", ["APPROVE", "DECLINE"], horizontal=True)
            hr_notes = st.text_area("Reviewer Notes")
            if st.form_submit_button("Submit Review", type="primary"):
                with st.spinner("Re-running pipeline with human decision..."):
                    st.session_state["last_result"] = run_pipeline(
                        r.get("raw_application", {}), human_decision=hr_dec, human_notes=hr_notes)
                st.rerun()

# Fill the sidebar System Health placeholder LAST, so it reflects any decision
# just submitted this run (real-time, no extra click).
with sys_health_slot.container():
    render_system_health()
