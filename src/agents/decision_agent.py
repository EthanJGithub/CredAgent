"""DecisionAgent — final APPROVE / DECLINE / REFER + LLM reasoning + HITL routing."""
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import CreditDecisionState
from src.llm import get_llm

logger = logging.getLogger(__name__)

DECISION_SYSTEM_PROMPT = """You are a credit analyst at an employer-sponsored installment lending company.
Based on the risk assessment and compliance check provided, generate a concise decision explanation.

Rules:
- APPROVE (LOW risk, no compliance flags): 2-3 sentences explaining why the applicant qualifies.
- DECLINE (HIGH/DECLINE risk): 2-3 sentences explaining the decline, referencing the top adverse
  factors. Do NOT mention protected characteristics.
- REFER (compliance flags present): 1-2 sentences explaining why manual review is needed.

Be professional, factual, and avoid discriminatory language. Reference specific financial metrics.
Respond with ONLY the explanation text, no preamble.
"""

CREDIT_LIMITS = {"LOW": 2500.0, "MEDIUM": 1000.0, "HIGH": None, "DECLINE": None}


def run(state: CreditDecisionState) -> dict:
    applicant_id = state["applicant_id"]
    risk_tier = state.get("risk_tier", "DECLINE")
    compliance_flags = state.get("compliance_flags", [])
    human_decision = state.get("human_decision")
    top_risk_factors = state.get("top_risk_factors", [])
    risk_probability = state.get("risk_probability") or 1.0
    derived = state.get("derived_features", {}) or {}

    # Compliance flags → REFER for human review (unless a human already ruled).
    if compliance_flags and human_decision is None:
        return {
            "final_decision": "REFER",
            "credit_limit": None,
            "decision_reasoning": (
                "This application has been referred for manual review due to "
                f"{len(compliance_flags)} compliance flag(s) identified during policy review."
            ),
            "decision_confidence": 0.5,
            "requires_human_review": True,
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] DecisionAgent: REFER — "
                f"{len(compliance_flags)} compliance flag(s)"
            ],
        }

    # MEDIUM tier with no prior human decision → pause for human review.
    if risk_tier == "MEDIUM" and human_decision is None:
        return {
            "final_decision": None,
            "credit_limit": None,
            "decision_reasoning": None,
            "decision_confidence": None,
            "requires_human_review": True,
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] DecisionAgent: MEDIUM tier — "
                f"flagged for human review (prob={risk_probability:.3f})"
            ],
        }

    if human_decision in ("APPROVE", "DECLINE"):
        final_decision = human_decision
        confidence = 0.99
    elif risk_tier == "LOW":
        final_decision = "APPROVE"
        confidence = 1.0 - risk_probability
    elif risk_tier in ("HIGH", "DECLINE"):
        final_decision = "DECLINE"
        confidence = risk_probability
    else:
        final_decision = "DECLINE"
        confidence = 0.70

    credit_limit = CREDIT_LIMITS.get(risk_tier) if final_decision == "APPROVE" else None
    if human_decision == "APPROVE" and credit_limit is None:
        credit_limit = 1000.0  # conservative limit for a human-overridden approval

    try:
        llm = get_llm(temperature=0.3)
        context = (
            f"Decision: {final_decision}\n"
            f"Risk tier: {risk_tier}\n"
            f"Default probability: {risk_probability:.1%}\n"
            f"Debt-to-income ratio: {derived.get('debt_to_income', 'N/A')}\n"
            f"Credit-to-income ratio: {derived.get('credit_to_income_ratio', 'N/A')}\n"
            f"Top adverse factors: {', '.join(top_risk_factors[:3]) if top_risk_factors else 'None'}\n"
            f"Human review override: {human_decision or 'None'}\n"
            f"Compliance flags: {len(compliance_flags)}"
        )
        response = llm.invoke([
            SystemMessage(content=DECISION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        reasoning = response.content.strip()
    except Exception as exc:
        logger.warning("[DecisionAgent] Reasoning generation failed: %s", exc)
        reasoning = (
            f"Application {final_decision.lower()}d based on {risk_tier.lower()} "
            f"risk assessment (default probability: {risk_probability:.1%})."
        )

    logger.info(
        "[DecisionAgent] %s: %s, limit=%s, confidence=%.2f",
        applicant_id, final_decision, credit_limit, confidence,
    )

    return {
        "final_decision": final_decision,
        "credit_limit": credit_limit,
        "decision_reasoning": reasoning,
        "decision_confidence": round(confidence, 3),
        "requires_human_review": False,
        "audit_trail": state.get("audit_trail", []) + [
            f"[{datetime.now().isoformat()}] DecisionAgent: {final_decision} "
            f"(tier={risk_tier}, prob={risk_probability:.3f}, limit=${credit_limit})"
        ],
    }
