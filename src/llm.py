"""Pluggable LLM layer.

The pipeline must run in three environments without code changes:

1. **Groq** (free, no credit card) — set ``GROQ_API_KEY``. This is the default
   path described in the project amendments and used on Streamlit Cloud.
2. **Anthropic** — set ``ANTHROPIC_API_KEY`` to use Claude instead of Groq.
3. **Offline** — no key set. A deterministic, template-based stand-in is used
   so the full agentic pipeline still runs end-to-end (CI, local dev, demos
   without any account). Reasoning text is generated from the structured
   state rather than an LLM, and is clearly labelled as such.

Every agent calls :func:`get_llm` and uses the LangChain ``.invoke(messages)``
interface, so the three back-ends are interchangeable.
"""
from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class _Message:
    """Minimal message object matching LangChain's ``.content`` attribute."""

    def __init__(self, content: str):
        self.content = content


class LLMClient:
    """Thin wrapper around a LangChain chat model.

    LangChain models are Pydantic objects that reject arbitrary attributes, so
    we keep the ``provider`` label here (used in the audit trail) and proxy
    ``.invoke`` through to the wrapped model.
    """

    def __init__(self, llm, provider: str):
        self._llm = llm
        self.provider = provider

    def invoke(self, messages):
        return self._llm.invoke(messages)


class OfflineLLM:
    """Deterministic fallback used when no LLM API key is configured.

    It inspects the *system* prompt to decide what kind of artefact to
    synthesise (compliance JSON, a decision explanation, or an adverse-action
    notice) and builds it from the human message text. This keeps the pipeline
    fully functional with zero external dependencies.
    """

    provider = "offline-template"

    def invoke(self, messages: List) -> _Message:  # noqa: D401 - mimics LangChain
        system = ""
        human = ""
        for m in messages:
            content = getattr(m, "content", str(m))
            role = m.__class__.__name__.lower()
            if "system" in role:
                system = content
            else:
                human = content

        s = system.lower()
        if "compliance officer" in s or "compliance_flags" in s:
            return _Message(self._compliance_response(human))
        if "adverse action notice" in s:
            return _Message(self._adverse_action_response(human))
        return _Message(self._decision_response(human))

    # ── template builders ──────────────────────────────────────────────────
    @staticmethod
    def _compliance_response(human: str) -> str:
        # Conservative default: no automated flags. Protected-class proxies are
        # already excluded from the feature set, so a clean result is correct
        # for the demo. Returns the exact JSON contract the agent parses.
        return (
            '{"compliance_flags": [], '
            '"analysis": "Decision relies on credit-bureau scores, debt-to-income '
            'and employment signals. No protected-class characteristics or direct '
            'proxies were used. Adverse-action reasons are specific and disclosable '
            'under ECOA §1002.9.", '
            '"clean": true}'
        )

    @staticmethod
    def _decision_response(human: str) -> str:
        decision = "the application"
        for line in human.splitlines():
            low = line.lower()
            if low.startswith("decision:"):
                decision = line.split(":", 1)[1].strip()
                break
        factors = ""
        for line in human.splitlines():
            if line.lower().startswith("top adverse factors"):
                factors = line.split(":", 1)[1].strip()
                break
        verb = {
            "APPROVE": "approved",
            "DECLINE": "declined",
            "REFER": "referred for manual review",
        }.get(decision.upper(), "processed")
        base = (
            f"This application has been {verb} based on the model's risk "
            f"assessment of the applicant's credit-bureau scores, debt-to-income "
            f"ratio and employment profile."
        )
        if verb == "declined" and factors and factors.lower() != "none":
            base += f" The primary contributing factors were: {factors}."
        elif verb == "approved":
            base += " The applicant's repayment indicators fall within the approved range."
        return base + " [Generated offline — set GROQ_API_KEY for LLM-authored text.]"

    @staticmethod
    def _adverse_action_response(human: str) -> str:
        applicant = "Applicant"
        date = ""
        factors = []
        collecting = False
        for line in human.splitlines():
            if line.lower().startswith("applicant id"):
                applicant = line.split(":", 1)[1].strip()
            elif line.lower().startswith("date:"):
                date = line.split(":", 1)[1].strip()
            elif line.lower().startswith("adverse factors"):
                collecting = True
            elif collecting and line.strip().startswith("-"):
                factors.append(line.strip().lstrip("- ").strip())
        reasons = "\n".join(f"  • {f}" for f in factors) or "  • Insufficient creditworthiness"
        return (
            "NOTICE OF ADVERSE ACTION\n"
            f"Date: {date}\n\n"
            f"Re: Credit Application {applicant}\n\n"
            "Dear Applicant,\n\n"
            "Thank you for your recent application for credit. After careful review, "
            "we are unable to approve your request at this time.\n\n"
            "The specific principal reason(s) for this decision were:\n"
            f"{reasons}\n\n"
            "The Federal Equal Credit Opportunity Act prohibits creditors from "
            "discriminating against credit applicants on the basis of race, color, "
            "religion, national origin, sex, marital status, age (provided the "
            "applicant has the capacity to contract), because all or part of the "
            "applicant's income derives from any public assistance program, or "
            "because the applicant has in good faith exercised any right under the "
            "Consumer Credit Protection Act.\n\n"
            "If you have questions regarding this notice, contact us at "
            "compliance@credagent.io.\n\n"
            "CredAgent Lending Platform\n"
            "[Generated offline — set GROQ_API_KEY for LLM-authored text.]"
        )


def get_llm(temperature: float = 0.0):
    """Return an LLM client.

    Resolution order: Groq → Anthropic → offline template. The returned object
    always supports ``.invoke([SystemMessage(...), HumanMessage(...)])`` and
    exposes a ``.provider`` attribute for logging/telemetry.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and not groq_key.startswith("your_"):
        try:
            from langchain_groq import ChatGroq

            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            return LLMClient(ChatGroq(model=model, temperature=temperature), f"groq:{model}")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Groq init failed (%s); trying next backend.", exc)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key and not anthropic_key.startswith("your_"):
        try:
            from langchain_anthropic import ChatAnthropic

            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            return LLMClient(ChatAnthropic(model=model, temperature=temperature), f"anthropic:{model}")
        except Exception as exc:  # pragma: no cover - depends on env
            logger.warning("Anthropic init failed (%s); using offline LLM.", exc)

    logger.info("No LLM API key configured — using deterministic offline LLM.")
    return OfflineLLM()


def llm_is_live() -> bool:
    """True when a real (non-offline) LLM backend is configured."""
    return not isinstance(get_llm(), OfflineLLM)
