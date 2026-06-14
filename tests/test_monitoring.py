"""Tests for the persistent decision store + monitoring endpoints."""


def test_decision_is_logged_and_surfaced(client, sample_applications):
    app = sample_applications[0]  # low-risk -> APPROVE
    client.post("/api/v1/decisions", json=app)

    recent = client.get("/api/v1/monitoring/decisions?limit=10")
    assert recent.status_code == 200
    ids = [d["applicant_id"] for d in recent.json()["decisions"]]
    assert app["applicant_id"] in ids


def test_monitoring_summary_shape(client, sample_applications):
    for app in sample_applications:
        client.post("/api/v1/decisions", json=app)

    summary = client.get("/api/v1/monitoring/summary").json()
    assert summary["total"] >= 2
    assert set(summary["decision_counts"]) == {"APPROVE", "DECLINE", "REFER"}
    assert 0.0 <= summary["approval_rate"] <= 1.0
    fl = summary["fair_lending"]
    assert "adverse_impact_ratio" in fl and "by_group" in fl
    assert fl["rule"].startswith("four-fifths")


def test_export_csv(client, sample_applications):
    client.post("/api/v1/decisions", json=sample_applications[0])
    r = client.get("/api/v1/monitoring/export.csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert "applicant_id" in r.text


def test_evidence_hub_captures_governance_bundle(client, sample_applications):
    app = sample_applications[1]  # high-risk -> DECLINE (exercises all 3 LLM agents)
    client.post("/api/v1/decisions", json=app)
    ev = client.get(f"/api/v1/evidence/{app['applicant_id']}")
    assert ev.status_code == 200
    bundle = ev.json()
    assert bundle["model_version"] == "xgb-v1.0"
    assert bundle["feature_inputs"] is not None
    assert bundle["shap_values"]                       # SHAP attribution recorded
    assert len(bundle["llm_calls"]) >= 1               # exact prompts captured
    call = bundle["llm_calls"][0]
    assert "system_prompt" in call and "user_prompt" in call and "provider" in call


def test_evidence_404_for_unknown(client):
    assert client.get("/api/v1/evidence/nope").status_code == 404


def test_system_health_endpoint(client, sample_applications):
    for app in sample_applications:
        client.post("/api/v1/decisions", json=app)
    s = client.get("/api/v1/monitoring/system").json()
    assert "metrics" in s and "drift" in s
    m = s["metrics"]
    assert m["requests_total"] >= 1
    assert set(m["latency_ms"]) == {"p50", "p95", "p99", "max"}
    assert "overall_status" in s["drift"]


def test_drift_endpoint_shape(client, sample_applications):
    for app in sample_applications:
        client.post("/api/v1/decisions", json=app)
    d = client.get("/api/v1/monitoring/drift").json()
    assert d["reference_available"] is True
    assert {"feature", "psi", "status"} <= set(d["features"][0])
    assert d["overall_status"] in ("stable", "moderate", "significant")
