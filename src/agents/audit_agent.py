"""AuditAgent — adverse-action notice generation + audit trail + packaging."""
import logging
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.state import CreditDecisionState
from src.llm import get_llm

logger = logging.getLogger(__name__)

# Concrete demo values so the letter is complete, not a skeleton template.
CREDITOR_NAME = "CredAgent Lending Platform"
CREDITOR_ADDRESS = "123 Finance Avenue, Suite 400, New York, NY 10017"
CREDITOR_EMAIL = "compliance@credagent.io"
CREDIT_PRODUCT = "Employer-Sponsored Installment Loan"

ADVERSE_ACTION_SYSTEM_PROMPT = """You are generating a CFPB-compliant adverse action notice
for a declined credit application, as required by ECOA (Regulation B, 12 CFR 1002.9) and FCRA.

Produce a COMPLETE, professional business letter using ONLY the concrete values supplied in the
user message (creditor name and address, product, date, application reference, and the specific
reasons). The letter MUST include:
1. The creditor letterhead (name + address) and the date.
2. A statement that the application for the named product was denied.
3. The specific principal reason(s) for the action — use exactly the reasons provided.
4. The verbatim ECOA notice: "The Federal Equal Credit Opportunity Act prohibits creditors from
   discriminating against credit applicants on the basis of race, color, religion, national
   origin, sex, marital status, age, or because all or part of the applicant's income derives
   from any public assistance program."
5. The contact email for questions, then a sign-off with the creditor name.

ABSOLUTE RULES:
- NEVER output square brackets or placeholder tokens (no [Applicant Name], [Address], [Date], etc.).
- Do NOT invent an applicant's personal name; address the recipient as "Dear Applicant,".
- Use only the values provided. Be concise (under 230 words). Do not mention model scores or probabilities.
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
    reasons = "\n".join(f"  • {f}" for f in (factors[:4] or ["Insufficient creditworthiness"]))
    date = datetime.now().strftime("%B %d, %Y")
    return (
        f"{CREDITOR_NAME}\n{CREDITOR_ADDRESS}\n\n"
        f"{date}\n\n"
        f"Re: Application {applicant_id} — {CREDIT_PRODUCT}\n\n"
        "Dear Applicant,\n\n"
        f"After careful review, we are unable to approve your application for the "
        f"{CREDIT_PRODUCT} at this time. Under the Equal Credit Opportunity Act, you are "
        "entitled to a statement of the specific principal reason(s) for this decision:\n\n"
        f"{reasons}\n\n"
        "The Federal Equal Credit Opportunity Act prohibits creditors from discriminating "
        "against credit applicants on the basis of race, color, religion, national origin, sex, "
        "marital status, age, or because all or part of the applicant's income derives from any "
        "public assistance program.\n\n"
        f"If you have questions regarding this notice, please contact us at {CREDITOR_EMAIL}.\n\n"
        f"Sincerely,\n{CREDITOR_NAME}"
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
            factors_str = "\n".join(f"- {f}" for f in top_risk_factors[:4]) or "- Insufficient creditworthiness"
            response = llm.invoke([
                SystemMessage(content=ADVERSE_ACTION_SYSTEM_PROMPT),
                HumanMessage(content=(
                    "Use exactly these values; do not add placeholders.\n"
                    f"Creditor name: {CREDITOR_NAME}\n"
                    f"Creditor address: {CREDITOR_ADDRESS}\n"
                    f"Creditor contact email: {CREDITOR_EMAIL}\n"
                    f"Credit product: {CREDIT_PRODUCT}\n"
                    f"Date: {datetime.now().strftime('%B %d, %Y')}\n"
                    f"Application reference: {applicant_id}\n"
                    f"Specific principal reasons for denial:\n{factors_str}"
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
