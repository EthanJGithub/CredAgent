"""AuditAgent — adverse-action notice generation + audit trail + packaging."""
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import CreditDecisionState
from src.llm import get_llm

logger = logging.getLogger(__name__)

ADVERSE_ACTION_SYSTEM_PROMPT = """You are generating a CFPB-compliant adverse action notice
for a declined credit application, as required by ECOA (Regulation B, 12 CFR 1002.9) and FCRA.

The notice MUST include:
1. A statement that the application was denied.
2. The name of the creditor (use: CredAgent Lending Platform).
3. A statement of the specific principal reason(s) for the action (use the adverse factors provided).
4. The ECOA notice: "The Federal Equal Credit Opportunity Act prohibits creditors from
   discriminating against credit applicants on the basis of race, color, religion, national
   origin, sex, marital status, age, or because all or part of the applicant's income derives
   from any public assistance program."
5. Contact information for questions: compliance@credagent.io

Format as a professional letter. Be concise (under 220 words). Do not mention model scores or probabilities.

CRITICAL FORMATTING RULES:
- Do NOT include any placeholder fields in brackets such as [Applicant Name], [Address],
  [Credit Product], [Date], [City, State ZIP]. Never emit square brackets at all.
- Do NOT invent the applicant's name, address, or any personal details you were not given.
- Address the recipient only as "Dear Applicant,". Use only the information explicitly provided.
- Start the letter with the date and the applicant ID provided, then the salutation.
"""

# Strips any leftover bracketed placeholders like "[Applicant Name]" the LLM may emit.
_PLACEHOLDER_RE = re.compile(r"\[[^\]]{0,60}\]")


def _strip_placeholders(text: str) -> str:
    cleaned = _PLACEHOLDER_RE.sub("", text)
    # Tidy whitespace left behind by removed placeholders.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def _fallback_notice(applicant_id: str, factors) -> str:
    factors_str = "; ".join(factors[:3]) if factors else "insufficient creditworthiness"
    return (
        "NOTICE OF ADVERSE ACTION\n\n"
        f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Re: Credit Application {applicant_id}\n\n"
        "Dear Applicant,\n\n"
        "After careful review, we are unable to approve your application for credit at this "
        f"time. The specific principal reason(s) for this decision were: {factors_str}.\n\n"
        "The Federal Equal Credit Opportunity Act prohibits creditors from discriminating "
        "against credit applicants on the basis of race, color, religion, national origin, sex, "
        "marital status, age, or because all or part of the applicant's income derives from any "
        "public assistance program.\n\n"
        "If you have questions regarding this notice, contact us at compliance@credagent.io.\n\n"
        "CredAgent Lending Platform"
    )


def run(state: CreditDecisionState) -> dict:
    applicant_id = state["applicant_id"]
    final_decision = state.get("final_decision")
    top_risk_factors = state.get("top_risk_factors", []) or []
    request_timestamp = state.get("request_timestamp", "")

    try:
        start_dt = datetime.fromisoformat(request_timestamp)
        processing_ms = (datetime.now() - start_dt).total_seconds() * 1000
    except Exception:
        processing_ms = 0.0

    adverse_action_notice = None
    if final_decision == "DECLINE":
        try:
            llm = get_llm(temperature=0)
            factors_str = "\n".join(f"- {f}" for f in top_risk_factors[:4])
            response = llm.invoke([
                SystemMessage(content=ADVERSE_ACTION_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Applicant ID: {applicant_id}\n"
                    f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
                    f"Adverse factors:\n{factors_str}"
                )),
            ])
            adverse_action_notice = _strip_placeholders(response.content.strip())
            # If the model still produced something hollow, use the clean template.
            if len(adverse_action_notice) < 120:
                adverse_action_notice = _fallback_notice(applicant_id, top_risk_factors)
        except Exception as exc:
            logger.warning("[AuditAgent] Adverse action generation failed: %s", exc)
            adverse_action_notice = _fallback_notice(applicant_id, top_risk_factors)

    final_audit_entry = (
        f"[{datetime.now().isoformat()}] AuditAgent: decision={final_decision}, "
        f"processing_time={processing_ms:.0f}ms, "
        f"adverse_notice={'generated' if adverse_action_notice else 'N/A'}"
    )
    logger.info("[AuditAgent] %s: %s — packaged in %.0fms", applicant_id, final_decision, processing_ms)

    return {
        "adverse_action_notice": adverse_action_notice,
        "processing_time_ms": round(processing_ms, 1),
        "final_response_packaged": True,
        "audit_trail": state.get("audit_trail", []) + [final_audit_entry],
    }
