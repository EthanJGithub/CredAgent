"""LangGraph state schema — the single source of truth for what flows
between all five agents in the credit-decisioning pipeline.
"""
from typing import TypedDict, Optional, List, Dict, Any


class CreditDecisionState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    applicant_id: str
    raw_application: Dict[str, Any]
    request_timestamp: str

    # ── IngestionAgent ─────────────────────────────────────────────────────
    cleaned_features: Optional[Dict[str, Any]]
    derived_features: Optional[Dict[str, Any]]
    ingestion_errors: List[str]
    ingestion_complete: bool

    # ── RiskScoringAgent ───────────────────────────────────────────────────
    risk_probability: Optional[float]
    risk_tier: Optional[str]
    shap_values: Optional[Dict[str, float]]
    top_risk_factors: Optional[List[str]]
    model_version: Optional[str]

    # ── PolicyComplianceAgent ──────────────────────────────────────────────
    compliance_flags: List[str]
    retrieved_policy_excerpts: List[str]
    policy_check_complete: bool

    # ── Human-in-the-loop ──────────────────────────────────────────────────
    requires_human_review: bool
    human_decision: Optional[str]
    human_notes: Optional[str]

    # ── DecisionAgent ──────────────────────────────────────────────────────
    final_decision: Optional[str]
    credit_limit: Optional[float]
    decision_reasoning: Optional[str]
    decision_confidence: Optional[float]

    # ── AuditAgent ─────────────────────────────────────────────────────────
    adverse_action_notice: Optional[str]
    audit_trail: List[str]
    processing_time_ms: Optional[float]
    final_response_packaged: bool

    # ── AI Evidence Hub (governance / traceability) ────────────────────────
    # One record per LLM invocation: agent, provider, model, system+user prompt.
    llm_calls: List[Dict[str, Any]]
