"""IngestionAgent — validate, normalise and enrich the raw application."""
import logging
from datetime import datetime

from src.graph.state import CreditDecisionState

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "amt_credit", "amt_income_total", "amt_annuity",
    "days_birth", "days_employed", "code_gender",
    "name_income_type", "name_education_type",
]

VALID_INCOME_TYPES = [
    "Working", "Commercial associate", "Pensioner", "State servant",
    "Unemployed", "Student", "Businessman", "Maternity leave",
]

VALID_EDUCATION_TYPES = [
    "Higher education", "Secondary / secondary special", "Incomplete higher",
    "Lower secondary", "Academic degree",
]


def run(state: CreditDecisionState) -> dict:
    applicant_id = state["applicant_id"]
    raw = state.get("raw_application", {})
    errors = []

    normalized = {k.lower(): v for k, v in raw.items()}

    for field in REQUIRED_FIELDS:
        if field not in normalized or normalized[field] is None:
            errors.append(f"Missing required field: {field}")

    income_type = normalized.get("name_income_type", "")
    if income_type not in VALID_INCOME_TYPES:
        errors.append(f"Invalid name_income_type: '{income_type}'.")

    edu_type = normalized.get("name_education_type", "")
    if edu_type not in VALID_EDUCATION_TYPES:
        errors.append(f"Invalid name_education_type: '{edu_type}'.")

    cleaned = {
        "AMT_CREDIT": float(normalized.get("amt_credit", 0) or 0),
        "AMT_INCOME_TOTAL": float(normalized.get("amt_income_total", 0) or 0),
        "AMT_ANNUITY": float(normalized.get("amt_annuity", 0) or 0),
        "DAYS_BIRTH": int(normalized.get("days_birth", -10000) or -10000),
        "DAYS_EMPLOYED": int(normalized.get("days_employed", 0) or 0),
        "EXT_SOURCE_1": normalized.get("ext_source_1"),
        "EXT_SOURCE_2": normalized.get("ext_source_2"),
        "EXT_SOURCE_3": normalized.get("ext_source_3"),
        "CODE_GENDER": str(normalized.get("code_gender", "X")).upper(),
        "FLAG_OWN_CAR": int(normalized.get("flag_own_car", 0) or 0),
        "FLAG_OWN_REALTY": int(normalized.get("flag_own_realty", 0) or 0),
        "CNT_CHILDREN": int(normalized.get("cnt_children", 0) or 0),
        "NAME_INCOME_TYPE": normalized.get("name_income_type", "Working"),
        "NAME_EDUCATION_TYPE": normalized.get(
            "name_education_type", "Secondary / secondary special"
        ),
    }

    inc = cleaned["AMT_INCOME_TOTAL"]
    derived = {
        "debt_to_income": round(cleaned["AMT_ANNUITY"] / (inc + 1), 4),
        "credit_to_income_ratio": round(cleaned["AMT_CREDIT"] / (inc + 1), 4),
        "age_years": round(abs(cleaned["DAYS_BIRTH"]) / 365, 1),
        "employment_months": round(min(abs(cleaned["DAYS_EMPLOYED"]) / 30, 480), 1),
        "annuity_to_credit_ratio": round(
            cleaned["AMT_ANNUITY"] / (cleaned["AMT_CREDIT"] + 1), 4
        ),
    }

    blocking = [e for e in errors if e.startswith("Missing required")]
    logger.info(
        "[IngestionAgent] %s: age=%syr, dti=%.3f, errors=%d",
        applicant_id, derived["age_years"], derived["debt_to_income"], len(errors),
    )

    return {
        "cleaned_features": cleaned,
        "derived_features": derived,
        "ingestion_errors": errors,
        "ingestion_complete": len(blocking) == 0,
        "audit_trail": state.get("audit_trail", []) + [
            f"[{datetime.now().isoformat()}] IngestionAgent: "
            f"age={derived['age_years']}yr, dti={derived['debt_to_income']:.3f}, "
            f"validation_errors={len(errors)}"
        ],
    }
