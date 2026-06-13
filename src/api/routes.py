import json
import os
import time
from datetime import datetime

import csv
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src import store
from src.api.schemas import (
    ApplicationRequest, DecisionResponse, HumanReviewRequest,
    HealthResponse, ModelInfoResponse,
)
from src.graph.state import CreditDecisionState
from src.graph.workflow import app_graph, memory  # noqa: F401  (memory kept for parity)
from src.rag.retriever import is_ready as vectorstore_is_ready

router = APIRouter(prefix="/api/v1")

MODEL_PATH = "models/xgboost_risk.pkl"
METADATA_PATH = "models/model_metadata.json"

# In-memory store for portfolio demo purposes:
#   applicant_id -> {"state": <result dict>, "thread_id": str}
decisions_store: dict = {}


def _empty_state(applicant_id: str, raw: dict) -> CreditDecisionState:
    return {
        "applicant_id": applicant_id,
        "raw_application": raw,
        "request_timestamp": datetime.now().isoformat(),
        "cleaned_features": None,
        "derived_features": None,
        "ingestion_errors": [],
        "ingestion_complete": False,
        "risk_probability": None,
        "risk_tier": None,
        "shap_values": None,
        "top_risk_factors": None,
        "model_version": None,
        "compliance_flags": [],
        "retrieved_policy_excerpts": [],
        "policy_check_complete": False,
        "requires_human_review": False,
        "human_decision": None,
        "human_notes": None,
        "final_decision": None,
        "credit_limit": None,
        "decision_reasoning": None,
        "decision_confidence": None,
        "adverse_action_notice": None,
        "audit_trail": [],
        "processing_time_ms": None,
        "final_response_packaged": False,
    }


def _build_response(result: dict) -> DecisionResponse:
    return DecisionResponse(
        applicant_id=result["applicant_id"],
        final_decision=result.get("final_decision"),
        credit_limit=result.get("credit_limit"),
        risk_probability=result.get("risk_probability"),
        risk_tier=result.get("risk_tier"),
        top_risk_factors=result.get("top_risk_factors"),
        shap_values=result.get("shap_values"),
        decision_reasoning=result.get("decision_reasoning"),
        decision_confidence=result.get("decision_confidence"),
        compliance_flags=result.get("compliance_flags", []),
        retrieved_policy_excerpts=result.get("retrieved_policy_excerpts", []),
        adverse_action_notice=result.get("adverse_action_notice"),
        processing_time_ms=result.get("processing_time_ms"),
        requires_human_review=result.get("requires_human_review", False),
        audit_trail=result.get("audit_trail", []),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=os.path.exists(MODEL_PATH),
        vectorstore_ready=vectorstore_is_ready(),
    )


@router.post("/decisions", response_model=DecisionResponse)
async def submit_decision(application: ApplicationRequest):
    start = time.time()
    thread_id = application.applicant_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = _empty_state(application.applicant_id, application.model_dump())
    result = app_graph.invoke(initial_state, config=config)
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    decisions_store[application.applicant_id] = {"state": result, "thread_id": thread_id}
    store.log_decision(result, source="live")
    return _build_response(result)


@router.get("/decisions/{applicant_id}", response_model=DecisionResponse)
async def get_decision(applicant_id: str):
    if applicant_id not in decisions_store:
        raise HTTPException(status_code=404, detail="Decision not found")
    return _build_response(decisions_store[applicant_id]["state"])


@router.post("/decisions/{applicant_id}/human-review", response_model=DecisionResponse)
async def submit_human_review(applicant_id: str, review: HumanReviewRequest):
    if applicant_id not in decisions_store:
        raise HTTPException(status_code=404, detail="Decision not found")

    stored = decisions_store[applicant_id]
    if not stored["state"].get("requires_human_review"):
        raise HTTPException(status_code=400, detail="This decision does not require human review")

    thread_id = stored["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    start = time.time()
    update = {
        "human_decision": review.human_decision,
        "human_notes": review.human_notes,
        "requires_human_review": False,
    }
    result = app_graph.invoke(update, config=config)
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    decisions_store[applicant_id] = {"state": result, "thread_id": thread_id}
    store.log_decision(result, source="live")
    return _build_response(result)


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m src.ml.train")
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return ModelInfoResponse(
        model_version=meta["model_version"],
        features=meta["features"],
        training_auc=meta["training_auc"],
        decision_thresholds=meta["decision_thresholds"],
    )


# ── Monitoring / portfolio analytics ─────────────────────────────────────────
@router.get("/monitoring/summary")
async def monitoring_summary():
    """Aggregate portfolio metrics + fair-lending disparate-impact analysis."""
    return store.summary()


@router.get("/monitoring/decisions")
async def monitoring_decisions(limit: int = 50):
    """Most recent decisions for the audit table."""
    return {"decisions": store.fetch_recent(limit=limit)}


@router.get("/monitoring/export.csv")
async def monitoring_export():
    """Download the full decision log as CSV (audit / regulatory export)."""
    rows = store.fetch_all()
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=credagent_decisions.csv"},
    )
