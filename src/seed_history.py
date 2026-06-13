"""Seed the decision store with historical decisions.

Scores the committed ``data/sample_applicants.csv`` (600 real Home Credit
applicants, including gender) through the trained model and records the
resulting decisions. This gives the monitoring dashboard realistic volume and a
genuine gender distribution for disparate-impact analysis on a fresh deploy,
where the full training set is not present.

Scoring here is model-only (no LLM, no RAG) so seeding is fast (~1-2s) and
deterministic. The tier -> decision rule mirrors DecisionAgent:
    LOW -> APPROVE,  MEDIUM -> REFER (human review),  HIGH/DECLINE -> DECLINE.
"""
from __future__ import annotations

import logging
import os

import joblib
import pandas as pd

from src import store
from src.ml.features import engineer_features, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

SAMPLE_PATH = "data/sample_applicants.csv"
MODEL_PATH = "models/xgboost_risk.pkl"

_CREDIT_LIMITS = {"LOW": 2500.0, "MEDIUM": 1000.0}


def _tier(p: float) -> str:
    if p < 0.30:
        return "LOW"
    if p < 0.55:
        return "MEDIUM"
    if p < 0.75:
        return "HIGH"
    return "DECLINE"


def seed() -> int:
    if not (os.path.exists(SAMPLE_PATH) and os.path.exists(MODEL_PATH)):
        logger.warning("Seed skipped: missing %s or %s", SAMPLE_PATH, MODEL_PATH)
        return 0

    df = pd.read_csv(SAMPLE_PATH)
    model = joblib.load(MODEL_PATH)

    # Map the lowercase intake columns to the RAW names engineer_features wants.
    raw = pd.DataFrame({
        "AMT_CREDIT": df.amt_credit, "AMT_INCOME_TOTAL": df.amt_income_total,
        "AMT_ANNUITY": df.amt_annuity, "DAYS_BIRTH": df.days_birth,
        "DAYS_EMPLOYED": df.days_employed, "EXT_SOURCE_1": df.ext_source_1,
        "EXT_SOURCE_2": df.ext_source_2, "EXT_SOURCE_3": df.ext_source_3,
        "CODE_GENDER": df.code_gender, "FLAG_OWN_CAR": df.flag_own_car,
        "FLAG_OWN_REALTY": df.flag_own_realty, "CNT_CHILDREN": df.cnt_children,
        "NAME_INCOME_TYPE": df.name_income_type,
        "NAME_EDUCATION_TYPE": df.name_education_type,
    })
    X = engineer_features(raw)[FEATURE_COLUMNS]
    probs = model.predict_proba(X)[:, 1]

    store.init_db()
    n = 0
    for i, p in enumerate(probs):
        p = float(p)
        tier = _tier(p)
        if tier == "LOW":
            decision, limit, hitl = "APPROVE", _CREDIT_LIMITS["LOW"], False
        elif tier == "MEDIUM":
            decision, limit, hitl = "REFER", None, True
        else:
            decision, limit, hitl = "DECLINE", None, False
        row = df.iloc[i]
        store.log_decision(
            {
                "applicant_id": row.applicant_id,
                "raw_application": {
                    "code_gender": row.code_gender,
                    "amt_income_total": float(row.amt_income_total),
                    "amt_credit": float(row.amt_credit),
                    "amt_annuity": float(row.amt_annuity),
                    "name_income_type": row.name_income_type,
                },
                "risk_probability": round(p, 4),
                "risk_tier": tier,
                "final_decision": decision,
                "credit_limit": limit,
                "requires_human_review": hitl,
                "compliance_flags": [],
            },
            source="historical",
        )
        n += 1
    logger.info("Seeded %d historical decisions.", n)
    return n


def ensure_seeded() -> int:
    """Seed only if the store is empty (idempotent on restarts)."""
    if store.count() > 0:
        return 0
    return seed()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Seeded rows: {seed()} | store total: {store.count()}")
