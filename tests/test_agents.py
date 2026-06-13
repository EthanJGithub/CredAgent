"""Agent / pipeline tests exercising the full LangGraph graph end-to-end."""
from src.api.routes import _empty_state
from src.graph.workflow import app_graph
from src.demo_presets import PRESETS


def _run(preset_key):
    p = PRESETS[preset_key]
    cfg = {"configurable": {"thread_id": f"test-{preset_key}"}}
    state = _empty_state(p["applicant_id"], p)
    return app_graph.invoke(state, config=cfg)


def test_ingestion_derives_features():
    from src.agents import ingestion_agent
    out = ingestion_agent.run(_empty_state("x", PRESETS["low"]))
    assert out["ingestion_complete"] is True
    assert "debt_to_income" in out["derived_features"]


def test_low_risk_full_pipeline_approves():
    r = _run("low")
    assert r["risk_tier"] == "LOW"
    assert r["final_decision"] == "APPROVE"
    assert r["credit_limit"] == 2500.0
    assert r["shap_values"]  # non-empty SHAP dict
    assert r["final_response_packaged"] is True


def test_medium_risk_pauses_for_human_review():
    r = _run("medium")
    assert r["risk_tier"] == "MEDIUM"
    assert r["requires_human_review"] is True
    assert r["final_decision"] is None


def test_medium_risk_resumes_with_human_approval():
    cfg = {"configurable": {"thread_id": "test-hitl"}}
    p = PRESETS["medium"]
    first = app_graph.invoke(_empty_state(p["applicant_id"], p), config=cfg)
    assert first["requires_human_review"] is True
    resumed = app_graph.invoke(
        {"human_decision": "APPROVE", "human_notes": "Verified employment.",
         "requires_human_review": False},
        config=cfg,
    )
    assert resumed["final_decision"] == "APPROVE"
    assert resumed["credit_limit"] == 1000.0


def test_high_risk_declines_with_adverse_notice():
    r = _run("high")
    assert r["risk_tier"] in ("HIGH", "DECLINE")
    assert r["final_decision"] == "DECLINE"
    assert r["adverse_action_notice"]
    assert "Equal Credit Opportunity Act" in r["adverse_action_notice"]
