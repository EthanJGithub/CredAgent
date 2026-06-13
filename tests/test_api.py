def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True  # model trained in this repo


def test_model_info(client):
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    meta = response.json()
    assert meta["model_version"] == "xgb-v1.0"
    assert meta["training_auc"] >= 0.74
    assert len(meta["features"]) == 19
    # Sex (prohibited basis) and education (race/national-origin proxy) must
    # never be model features.
    assert not any("GENDER" in f.upper() for f in meta["features"])
    assert not any("EDUCATION" in f.upper() for f in meta["features"])


def test_submit_low_risk_approves(client, sample_applications):
    app = sample_applications[0]
    response = client.post("/api/v1/decisions", json=app)
    assert response.status_code == 200
    data = response.json()
    assert data["applicant_id"] == app["applicant_id"]
    assert data["risk_tier"] == "LOW"
    assert data["final_decision"] == "APPROVE"
    assert data["credit_limit"] == 2500.0
    assert data["risk_probability"] < 0.30
    assert data["adverse_action_notice"] is None
    assert len(data["audit_trail"]) >= 5


def test_submit_high_risk_declines_with_notice(client, sample_applications):
    app = sample_applications[1]
    response = client.post("/api/v1/decisions", json=app)
    assert response.status_code == 200
    data = response.json()
    assert data["final_decision"] == "DECLINE"
    assert data["risk_probability"] >= 0.75
    notice = data["adverse_action_notice"]
    assert notice is not None
    assert "Equal Credit Opportunity Act" in notice
    # No unfilled placeholders like [Applicant Name] / [Address].
    assert "[" not in notice and "]" not in notice
    # Education must never be cited as an adverse reason.
    assert "education" not in " ".join(data.get("top_risk_factors") or []).lower()


def test_get_decision_after_submit(client, sample_applications):
    app = sample_applications[0]
    client.post("/api/v1/decisions", json=app)
    response = client.get(f"/api/v1/decisions/{app['applicant_id']}")
    assert response.status_code == 200
    assert response.json()["applicant_id"] == app["applicant_id"]


def test_get_unknown_decision_404(client):
    assert client.get("/api/v1/decisions/does-not-exist").status_code == 404


MEDIUM_APPLICANT = {
    "applicant_id": "api-medium", "amt_credit": 495000.0, "amt_income_total": 180000.0,
    "amt_annuity": 24750.0, "days_birth": -11623, "days_employed": -1809,
    "ext_source_1": 0.641, "ext_source_2": 0.482, "ext_source_3": 0.161,
    "code_gender": "F", "flag_own_car": 1, "flag_own_realty": 1, "cnt_children": 0,
    "name_income_type": "Working", "name_education_type": "Incomplete higher",
}


def test_medium_risk_human_review_flow(client):
    r = client.post("/api/v1/decisions", json=MEDIUM_APPLICANT).json()
    assert r["risk_tier"] == "MEDIUM"
    assert r["requires_human_review"] is True
    assert r["final_decision"] is None

    resumed = client.post(
        f"/api/v1/decisions/{MEDIUM_APPLICANT['applicant_id']}/human-review",
        json={"human_decision": "APPROVE", "human_notes": "Verified employment."},
    )
    assert resumed.status_code == 200
    data = resumed.json()
    assert data["final_decision"] == "APPROVE"
    assert data["credit_limit"] == 1000.0
    assert data["requires_human_review"] is False


def test_human_review_on_non_referred_rejected(client, sample_applications):
    app = sample_applications[0]  # low-risk, auto-approved (no review needed)
    client.post("/api/v1/decisions", json=app)
    r = client.post(f"/api/v1/decisions/{app['applicant_id']}/human-review",
                    json={"human_decision": "DECLINE"})
    assert r.status_code == 400


def test_invalid_application_rejected(client):
    bad = {"applicant_id": "bad", "amt_credit": -5, "amt_income_total": 1000,
           "amt_annuity": 10, "days_birth": -1000, "days_employed": -10,
           "code_gender": "Z", "name_income_type": "Working",
           "name_education_type": "Higher education"}
    assert client.post("/api/v1/decisions", json=bad).status_code == 422
