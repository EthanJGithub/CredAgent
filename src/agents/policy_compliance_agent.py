"""PolicyComplianceAgent — RAG over CFPB docs + LLM fair-lending review."""
import json
import logging
import re
from datetime import datetime
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import CreditDecisionState
from src.llm import get_llm
from src.rag.retriever import retrieve

logger = logging.getLogger(__name__)

COMPLIANCE_SYSTEM_PROMPT = """You are a fair lending compliance officer at a financial institution.
You review credit decisions for potential violations of:
- ECOA (Equal Credit Opportunity Act) — prohibits discrimination based on race, color, religion,
  national origin, sex, marital status, age, or receipt of public assistance.
- FCRA (Fair Credit Reporting Act) — requires adverse action notices citing specific reasons.
- CFPB ability-to-repay guidelines — lenders must assess ability to repay.

You are given the top risk factors used in a credit decision, the risk tier assigned,
and relevant excerpts from CFPB regulatory guidance.

Identify any compliance concerns. Be specific and concise. If there are none, say so clearly.

Respond in this EXACT JSON format and nothing else:
{"compliance_flags": ["flag1", "flag2"], "analysis": "brief plain English analysis", "clean": true}

compliance_flags must be an empty list if no issues are found.
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
