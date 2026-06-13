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

st.markdown("""
<style>
 .decision-approve{background:#d4edda;color:#155724;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #28a745;}
 .decision-decline{background:#f8d7da;color:#721c24;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #dc3545;}
 .decision-refer{background:#fff3cd;color:#856404;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #ffc107;}
 .adverse-notice{background:#f8f9fa;padding:16px;border-radius:8px;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;border:1px solid #dee2e6;}
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


with st.sidebar:
    st.title("🏦 CredAgent")
    st.caption("Agentic Credit Decisioning System")
    st.markdown("---")
    st.markdown("**Decision Tiers**")
    st.markdown("""
| Tier | Probability | Action |
|---|---|---|
| 🟢 LOW | < 30% | Auto-Approve |
| 🟡 MEDIUM | 30–55% | Human Review |
| 🟠 HIGH | 55–75% | Auto-Decline |
| 🔴 DECLINE | > 75% | Auto-Decline |
""")
    st.markdown("---")
    st.markdown("**Quick Test Cases** *(real applicants)*")
    if st.button("📗 Low Risk"):
        st.session_state["prefill"] = "low"
    if st.button("📙 Medium Risk"):
        st.session_state["prefill"] = "medium"
    if st.button("📕 High Risk"):
        st.session_state["prefill"] = "high"
    st.markdown("---")
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
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    prob = r.get("risk_probability"); limit = r.get("credit_limit"); ms = r.get("processing_time_ms")
    col1.metric("Default Probability", f"{prob:.1%}" if prob is not None else "N/A")
    col2.metric("Risk Tier", r.get("risk_tier", "N/A"))
    col3.metric("Credit Limit", f"${limit:,.0f}" if limit else "—")
    col4.metric("Processing Time", f"{ms:.0f} ms" if ms else "N/A")
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📋 Decision", "📊 SHAP", "⚖️ Compliance", "📄 Adverse Notice", "🔍 Audit"])
    with t1:
        if r.get("decision_reasoning"):
            st.info(r["decision_reasoning"])
        for i, f in enumerate(r.get("top_risk_factors") or [], 1):
            st.markdown(f"**{i}.** {f}")
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
            with st.expander("CFPB Policy Excerpts Retrieved"):
                for ex in excerpts:
                    st.text(ex[:500] + "..." if len(ex) > 500 else ex)
                    st.markdown("---")
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
