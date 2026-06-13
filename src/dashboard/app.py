"""Local Streamlit dashboard — talks to the FastAPI backend over HTTP.

Run the API first:   uvicorn src.api.main:app --reload
Then the dashboard:  streamlit run src/dashboard/app.py

For the zero-backend cloud demo, use ``streamlit_app.py`` at the repo root.
"""
import os
import sys
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.demo_presets import PRESETS

API_BASE = os.getenv("CREDAGENT_API", "http://localhost:8000/api/v1")
st.set_page_config(page_title="CredAgent — Credit Decisioning", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
 .decision-approve{background:#d4edda;color:#155724;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #28a745;}
 .decision-decline{background:#f8d7da;color:#721c24;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #dc3545;}
 .decision-refer{background:#fff3cd;color:#856404;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #ffc107;}
 .decision-pending{background:#e2e3e5;color:#383d41;padding:16px 24px;border-radius:8px;font-size:1.4rem;font-weight:700;border-left:6px solid #6c757d;}
 .adverse-notice{background:#f8f9fa;padding:16px;border-radius:8px;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;border:1px solid #dee2e6;}
</style>""", unsafe_allow_html=True)


def check_api_health() -> dict:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).json()
    except Exception:
        return {"status": "error", "model_loaded": False, "vectorstore_ready": False}


def submit_application(payload: dict) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}/decisions", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def submit_human_review(applicant_id: str, decision: str, notes: str) -> Optional[dict]:
    try:
        r = requests.post(f"{API_BASE}/decisions/{applicant_id}/human-review",
                          json={"human_decision": decision, "human_notes": notes}, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Human review error: {e}")
    return None


def render_shap_waterfall(shap_values: dict):
    if not shap_values:
        st.info("SHAP values not available.")
        return
    items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    features, values = [i[0] for i in items], [i[1] for i in items]
    display = {
        "EXT_SOURCE_1": "Ext Credit Score 1", "EXT_SOURCE_2": "Ext Credit Score 2",
        "EXT_SOURCE_3": "Ext Credit Score 3", "debt_to_income": "Debt-to-Income",
        "credit_to_income_ratio": "Credit/Income Ratio", "annuity_to_credit_ratio": "Annuity/Credit Ratio",
        "employment_months": "Employment Length", "age_years": "Applicant Age",
        "AMT_CREDIT": "Credit Amount", "AMT_INCOME_TOTAL": "Annual Income",
        "CNT_CHILDREN": "Number of Children", "has_income_stability": "Income Stability",
    }
    labels = [display.get(f, f.replace("_", " ").title()) for f in features]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#dc3545" if v > 0 else "#28a745" for v in values]
    y = np.arange(len(features))
    bars = ax.barh(y, values, color=colors, height=0.6, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value — impact on log-odds of default (model margin)", fontsize=10)
    ax.set_title("Feature Impact on Risk Score", fontsize=12, fontweight="bold", pad=12)
    for bar, val in zip(bars, values):
        xpos = bar.get_width() + (0.002 if val >= 0 else -0.002)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color="#dc3545", label="Increases default risk"),
                       mpatches.Patch(color="#28a745", label="Decreases default risk")],
              fontsize=9, loc="lower right")
    ax.invert_yaxis(); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)


def render_decision_badge(decision: Optional[str]):
    css = {"APPROVE": "decision-approve", "DECLINE": "decision-decline",
           "REFER": "decision-refer", None: "decision-pending"}.get(decision, "decision-pending")
    icon = {"APPROVE": "✅", "DECLINE": "❌", "REFER": "⚠️", None: "⏳"}.get(decision, "⏳")
    st.markdown(f'<div class="{css}">{icon} &nbsp; {decision or "AWAITING REVIEW"}</div>',
                unsafe_allow_html=True)


def render_result(result: dict):
    st.markdown("---")
    st.subheader("Decision Result")
    render_decision_badge(result.get("final_decision"))
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    prob, limit, ms = result.get("risk_probability"), result.get("credit_limit"), result.get("processing_time_ms")
    c1.metric("Default Probability", f"{prob:.1%}" if prob is not None else "N/A")
    c2.metric("Risk Tier", result.get("risk_tier", "N/A"))
    c3.metric("Credit Limit", f"${limit:,.0f}" if limit else "—")
    c4.metric("Processing Time", f"{ms:.0f} ms" if ms else "N/A")
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3, t4, t5 = st.tabs(["📋 Decision", "📊 SHAP Analysis", "⚖️ Compliance", "📄 Adverse Notice", "🔍 Audit Trail"])
    with t1:
        if result.get("decision_reasoning"):
            st.markdown("**Decision Reasoning**"); st.info(result["decision_reasoning"])
        for i, f in enumerate(result.get("top_risk_factors") or [], 1):
            st.markdown(f"**{i}.** {f}")
    with t2:
        render_shap_waterfall(result.get("shap_values") or {})
        st.caption("SHAP contributions to the model's log-odds (margin); higher log-odds = "
                   "higher default probability. Red increases risk · Green decreases it.")
    with t3:
        flags = result.get("compliance_flags", [])
        if flags:
            st.warning(f"⚠️ {len(flags)} compliance flag(s) detected")
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ No compliance issues detected.")
        excerpts = result.get("retrieved_policy_excerpts", [])
        if excerpts:
            with st.expander("View Retrieved CFPB Policy Excerpts"):
                for i, ex in enumerate(excerpts, 1):
                    st.markdown(f"**Excerpt {i}:**")
                    st.text(ex[:600] + "..." if len(ex) > 600 else ex)
                    st.markdown("---")
    with t4:
        notice = result.get("adverse_action_notice")
        if notice:
            st.markdown(f'<div class="adverse-notice">{notice}</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Download Notice", data=notice,
                               file_name=f"adverse_action_{result.get('applicant_id', 'applicant')}.txt", mime="text/plain")
        elif result.get("final_decision") == "APPROVE":
            st.success("No adverse action notice required — application approved.")
        else:
            st.info("Adverse action notice will appear here for declined applications.")
    with t5:
        for entry in result.get("audit_trail", []):
            st.code(entry, language=None)

    if result.get("requires_human_review") and result.get("final_decision") != "APPROVE":
        st.markdown("---")
        st.subheader("👤 Human Review Required")
        st.warning("This application requires manual review (borderline score or compliance flag).")
        with st.form("human_review_form"):
            hr_decision = st.radio("Your Decision", ["APPROVE", "DECLINE"], horizontal=True)
            hr_notes = st.text_area("Reviewer Notes", placeholder="Justification for the override decision...")
            if st.form_submit_button("Submit Review", type="primary"):
                with st.spinner("Submitting human review..."):
                    hr = submit_human_review(result.get("applicant_id", ""), hr_decision, hr_notes)
                if hr:
                    st.session_state["last_result"] = hr
                    st.rerun()


with st.sidebar:
    st.title("🏦 CredAgent")
    st.caption("Agentic Credit Decisioning System")
    st.markdown("---")
    health = check_api_health()
    status = health.get("status", "error")
    st.markdown("**System Status**")
    st.markdown(
        f"{'🟢' if status == 'ok' else '🔴'} API: `{status}`\n\n"
        f"{'🟢' if health.get('model_loaded') else '🔴'} ML Model\n\n"
        f"{'🟢' if health.get('vectorstore_ready') else '🔴'} Vector Store"
    )
    st.markdown("---")
    st.markdown("**Decision Tiers**")
    st.markdown("""
| Tier | Probability | Action |
|------|------------|--------|
| 🟢 LOW | < 30% | Auto-Approve |
| 🟡 MEDIUM | 30–55% | Human Review |
| 🟠 HIGH | 55–75% | Auto-Decline |
| 🔴 DECLINE | > 75% | Auto-Decline |
""")
    st.markdown("---")
    st.markdown("**Quick Load** *(real applicants)*")
    if st.button("📗 Low Risk Applicant"):
        st.session_state["prefill"] = "low"
    if st.button("📙 Medium Risk Applicant"):
        st.session_state["prefill"] = "medium"
    if st.button("📕 High Risk Applicant"):
        st.session_state["prefill"] = "high"

preset = PRESETS.get(st.session_state.get("prefill"), {})

st.title("Applicant Credit Assessment")
st.caption("Submit a loan application for real-time agentic risk decisioning.")

with st.form("application_form"):
    st.subheader("Applicant Information")
    col1, col2 = st.columns(2)
    with col1:
        applicant_id = st.text_input("Applicant ID", value=preset.get("applicant_id", ""), placeholder="e.g. APP-20260101-001")
        amt_income = st.number_input("Annual Income ($)", min_value=0.0, step=1000.0, value=float(preset.get("amt_income_total", 150000.0)))
        amt_credit = st.number_input("Requested Credit ($)", min_value=0.0, step=500.0, value=float(preset.get("amt_credit", 250000.0)))
        amt_annuity = st.number_input("Monthly Payment ($)", min_value=0.0, step=10.0, value=float(preset.get("amt_annuity", 20000.0)))
        cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=int(preset.get("cnt_children", 0)))
    with col2:
        days_birth = st.number_input("Days Since Birth (negative integer)", value=int(preset.get("days_birth", -12000)), help="-12000 ≈ 32.8 years old")
        days_employed = st.number_input("Days Employed (negative = employed)", value=int(preset.get("days_employed", -2000)), help="-2000 ≈ employed ~5.5 years")
        ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, float(preset.get("ext_source_1", 0.60)), 0.01)
        ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, float(preset.get("ext_source_2", 0.58)), 0.01)
        ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, float(preset.get("ext_source_3", 0.55)), 0.01)

    st.subheader("Demographics & Profile")
    col3, col4 = st.columns(2)
    with col3:
        code_gender = st.selectbox("Gender", ["M", "F", "X"], index=["M", "F", "X"].index(preset.get("code_gender", "F")))
        flag_own_car = st.checkbox("Owns Car", value=bool(preset.get("flag_own_car", 0)))
        flag_own_realty = st.checkbox("Owns Realty", value=bool(preset.get("flag_own_realty", 0)))
    with col4:
        income_types = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"]
        edu_types = ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"]
        name_income_type = st.selectbox("Income Type", income_types, index=income_types.index(preset.get("name_income_type", "Working")))
        name_education_type = st.selectbox("Education Level", edu_types, index=edu_types.index(preset.get("name_education_type", "Secondary / secondary special")))

    submitted = st.form_submit_button("🚀 Submit for Decisioning", type="primary", use_container_width=True)

if submitted:
    if not applicant_id.strip():
        st.error("Applicant ID is required.")
    elif health.get("status") != "ok":
        st.error("API is not reachable. Make sure `uvicorn src.api.main:app --reload` is running.")
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
            result = submit_application(payload)
        if result:
            st.session_state["last_result"] = result
            st.session_state.pop("prefill", None)

if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])
