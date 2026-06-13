"""RiskScoringAgent — XGBoost inference + SHAP explanation + risk tiering."""
import json
import logging
from datetime import datetime

import joblib
import pandas as pd

from src.graph.state import CreditDecisionState
from src.ml.explainer import compute_shap_values
from src.ml.features import engineer_single_application, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_model = None
_explainer = None
_metadata = None


def _load_artifacts():
    global _model, _explainer, _metadata
    if _model is None:
        logger.info("Loading XGBoost model + SHAP explainer...")
        _model = joblib.load("models/xgboost_risk.pkl")
        _explainer = joblib.load("models/shap_explainer.pkl")
        with open("models/model_metadata.json") as f:
            _metadata = json.load(f)
        logger.info("Model loaded. Version: %s", _metadata["model_version"])


def _assign_risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    if probability < 0.55:
        return "MEDIUM"
    if probability < 0.75:
        return "HIGH"
    return "DECLINE"


def run(state: CreditDecisionState) -> dict:
    _load_artifacts()
    applicant_id = state["applicant_id"]
    raw = state.get("cleaned_features") or state.get("raw_application", {})

    try:
        features = engineer_single_application(raw)
        feature_df = pd.DataFrame([features])[FEATURE_COLUMNS]

        prob = float(_model.predict_proba(feature_df)[0, 1])
        tier = _assign_risk_tier(prob)
        shap_dict, top_factors = compute_shap_values(_explainer, feature_df)

        logger.info("[RiskScoringAgent] %s: prob=%.3f, tier=%s", applicant_id, prob, tier)

        return {
            "risk_probability": prob,
            "risk_tier": tier,
            "shap_values": shap_dict,
            "top_risk_factors": top_factors or ["No single dominant adverse factor"],
            "model_version": _metadata["model_version"],
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] RiskScoringAgent: "
                f"prob={prob:.3f}, tier={tier}, model={_metadata['model_version']}"
            ],
        }
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[RiskScoringAgent] Error for %s: %s", applicant_id, exc)
        return {
            "risk_probability": None,
            "risk_tier": "DECLINE",
            "shap_values": {},
            "top_risk_factors": ["Scoring error — defaulting to DECLINE"],
            "model_version": _metadata["model_version"] if _metadata else "unknown",
            "audit_trail": state.get("audit_trail", []) + [
                f"[{datetime.now().isoformat()}] RiskScoringAgent ERROR: {exc}"
            ],
        }
