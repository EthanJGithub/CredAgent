"""PolicyComplianceAgent — RAG over CFPB docs + LLM fair-lending review."""
import json
import logging
import re
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import CreditDecisionState
from src.llm import get_llm, evidence_record
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

COMPLIANCE_SYSTEM_PROMPT = """You are a fair lending compliance officer reviewing a credit decision
for violations of ECOA (Equal Credit Opportunity Act), FCRA, and CFPB guidance.

A prohibited basis is: race, color, religion, national origin, sex/gender, marital status, age, or
receipt of public assistance. You must flag a decision ONLY IF one of the stated adverse factors is
a prohibited basis or an obvious close proxy for one.

The following factors are PERMISSIBLE, business-justified, and must NOT be flagged when used:
- Credit-bureau / external credit scores
- Debt-to-income ratio, credit-to-income ratio, annuity-to-credit ratio
- Length of employment, income level, requested credit amount
- Number of dependents, income type (employment status), vehicle/property ownership

You MUST raise a compliance_flag if any adverse factor is a prohibited basis OR a known proxy:
- Education level / school attended — a documented proxy for race and national origin that the
  CFPB scrutinizes for disparate impact. Flag it (e.g. "education-proxy") if it appears.
- ZIP code / neighborhood — proxy for race (redlining). Flag if present.

If the adverse factors are limited to the permissible factors above, the decision is COMPLIANT and you
MUST return an empty compliance_flags list. Do not invent concerns beyond prohibited bases and the proxies listed.

Respond with ONLY this JSON and nothing else:
{"compliance_flags": [], "analysis": "<one sentence>", "clean": true}

Put a short string in compliance_flags ONLY for a genuine prohibited-basis problem; otherwise keep it empty.
"""


def _build_compliance_query(top_risk_factors: List[str], risk_tier: str) -> str:
    factors_str = "\n".join(f"- {f}" for f in (top_risk_factors or []))
    return (
        f"Credit decision: {risk_tier}\n"
        f"Top adverse factors used:\n{factors_str}\n\n"
        f"What CFPB regulations apply to this decision? "
        f"Are there any fair lending compliance concerns with these factors?"
    )


def run(state: CreditDecisionState) -> dict:
    applicant_id = state["applicant_id"]
    top_risk_factors = state.get("top_risk_factors", [])
    risk_tier = state.get("risk_tier", "UNKNOWN")

    try:
        query = _build_compliance_query(top_risk_factors, risk_tier)
        try:
            excerpts = retrieve(query, n_results=5)
        except Exception as exc:
            logger.warning("[PolicyComplianceAgent] retrieval unavailable: %s", exc)
            excerpts = []
        excerpts_text = "\n\n---\n\n".join(excerpts) if excerpts else "(no excerpts retrieved)"

        llm = get_llm(temperature=0)
        user_message = (
            f"Credit decision details:\n{query}\n\n"
            f"Relevant CFPB regulatory excerpts:\n{excerpts_text}\n\n"
            f"Analyze this decision for fair lending compliance issues."
        )
        response = llm.invoke([
            SystemMessage(content=COMPLIANCE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        evidence = evidence_record("PolicyComplianceAgent", llm, COMPLIANCE_SYSTEM_PROMPT, user_message)

        content = response.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                compliance_flags = json.loads(match.group()).get("compliance_flags", [])
            except json.JSONDecodeError:
                compliance_flags = []
                logger.warning("[PolicyComplianceAgent] JSON parse failed for %s", applicant_id)
        else:
            compliance_flags = []

        logger.info("[PolicyComplianceAgent] %s: %d flags", applicant_id, len(compliance_flags))

        return {
            "compliance_flags": compliance_flags,
            "retrieved_policy_excerpts": excerpts[:3],
            "policy_check_complete": True,
            "llm_calls": state.get("llm_calls", []) + [evidence],
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] PolicyComplianceAgent: "
                f"{len(compliance_flags)} flag(s), {len(excerpts)} excerpt(s) "
                f"[llm={getattr(llm, 'provider', 'unknown')}]"
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[PolicyComplianceAgent] Error for %s: %s", applicant_id, exc)
        return {
            "compliance_flags": [],
            "retrieved_policy_excerpts": [],
            "policy_check_complete": False,
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] PolicyComplianceAgent ERROR: {exc}"
            ],
        }
