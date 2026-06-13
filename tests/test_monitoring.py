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
