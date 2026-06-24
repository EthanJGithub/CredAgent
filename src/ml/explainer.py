"""SHAP explainability wrapper.

Turns a trained TreeExplainer + a single applicant's feature vector into a
``{feature: shap_value}`` dict and a list of plain-English adverse factors
suitable for an ECOA §1002.9 adverse-action notice.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap

from src.ml.features import FEATURE_COLUMNS, AUX_FEATURES

logger = logging.getLogger(__name__)

# Features that must NOT be stated as adverse-action reasons to an applicant.
# Age is a protected basis under ECOA; even where age is permissibly used in an
# empirically derived scoring system, it cannot be cited as a denial reason.
# Sex and education-related variables are excluded entirely from the model, but
# we list them here as defense-in-depth so they can never surface as a reason.
# We keep any such values in the SHAP chart (transparency/monitoring) but never
# surface them as the "specific principal reasons" in the notice/compliance check.
#
# AUX_FEATURES (relational credit-history aggregations) are also excluded from
# CITED reasons at single-application inference: the demo imputes them to medians
# (no access to a stranger's bureau history), so it would be misleading to cite
# "your installment history" as a reason. They remain in the model and in the
# evaluated AUC, where they are real.
PROTECTED_FROM_ADVERSE_REASONS = {
    "age_years", "DAYS_BIRTH", "employed_to_age",
    "CODE_GENDER_F", "CODE_GENDER_M",
    "NAME_EDUCATION_TYPE_Higher_education",
} | set(AUX_FEATURES)

FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_1": "External credit score (bureau 1)",
    "EXT_SOURCE_2": "External credit score (bureau 2)",
    "EXT_SOURCE_3": "External credit score (bureau 3)",
    "debt_to_income": "High debt-to-income ratio",
    "credit_to_income_ratio": "High credit-to-income ratio",
    "annuity_to_credit_ratio": "High annuity-to-credit ratio",
    "employment_months": "Limited employment history",
    "age_years": "Applicant age factor",
    "AMT_CREDIT": "Credit amount requested",
    "AMT_INCOME_TOTAL": "Income level",
    "AMT_ANNUITY": "Monthly payment obligation",
    "CNT_CHILDREN": "Number of dependents",
    "has_income_stability": "Income type stability",
    "NAME_INCOME_TYPE_Working": "Employment income status",
    "NAME_INCOME_TYPE_Commercial_associate": "Self-employment income status",
    "NAME_EDUCATION_TYPE_Higher_education": "Education level",
    "FLAG_OWN_CAR": "Vehicle ownership",
    "FLAG_OWN_REALTY": "Property ownership",
    "DAYS_EMPLOYED": "Length of current employment",
    "DAYS_BIRTH": "Age-related factor",
    "CODE_GENDER_F": "Demographic indicator",
    "CODE_GENDER_M": "Demographic indicator",
    # Expanded application-level features
    "EXT_SOURCE_mean": "Average external credit score",
    "EXT_SOURCE_min": "Lowest external credit score",
    "EXT_SOURCE_max": "Highest external credit score",
    "EXT_SOURCE_std": "External credit score spread",
    "EXT_SOURCE_prod": "Combined external credit score",
    "AMT_GOODS_PRICE": "Price of financed goods",
    "credit_to_goods": "Credit-to-goods-price ratio",
    "income_per_person": "Income per household member",
    "payment_rate": "Loan payment rate",
    "DAYS_REGISTRATION": "Time since registration",
    "DAYS_ID_PUBLISH": "Time since ID issued",
    "DAYS_LAST_PHONE_CHANGE": "Time since last phone change",
    "OWN_CAR_AGE": "Age of owned vehicle",
    "ext_mean_x_age": "Credit score relative to age",
    "CNT_FAM_MEMBERS": "Household size",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in social circle (30d)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults in social circle (60d)",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observations in social circle (30d)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Recent credit-bureau enquiries (quarter)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit-bureau enquiries (year)",
    "FLAG_DOCUMENT_3": "Document provided",
    "FLAG_EMP_PHONE": "Employer phone on file",
    "FLAG_PHONE": "Phone on file",
    "NAME_INCOME_TYPE_Pensioner": "Pension income status",
}


# The external-score family are all variants of one underlying signal; collapse
# them to a single disclosable reason so an adverse-action notice doesn't list
# "average / lowest / highest external credit score" as four separate reasons.
_EXT_FAMILY = {
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_mean", "EXT_SOURCE_min", "EXT_SOURCE_max", "EXT_SOURCE_std", "EXT_SOURCE_prod",
    "ext_mean_x_age",
}


def _reason_name(feat: str) -> str:
    if feat in _EXT_FAMILY:
        return "External credit score"
    return FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())


def compute_shap_values(
    explainer: shap.TreeExplainer,
    feature_vector: pd.DataFrame,
) -> Tuple[Dict[str, float], List[str]]:
    """Return (shap_dict, top_adverse_factors) for a single applicant row."""
    shap_vals = explainer.shap_values(feature_vector)

    # Binary XGBoost may return a list of two arrays; class 1 = default.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_array = np.asarray(shap_vals)
    if shap_array.ndim == 2:
        shap_array = shap_array[0]

    shap_dict = {
        feature: float(val) for feature, val in zip(FEATURE_COLUMNS, shap_array)
    }

    # Positive SHAP value = pushes prediction toward default (adverse).
    # Skip protected-basis features — they cannot be cited as denial reasons.
    sorted_features = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    top_factors: List[str] = []
    for feat, val in sorted_features:
        if val <= 0 or feat in PROTECTED_FROM_ADVERSE_REASONS:
            continue
        name = _reason_name(feat)
        if name not in top_factors:  # collapses the external-score family to one reason
            top_factors.append(name)
        if len(top_factors) >= 5:
            break

    return shap_dict, top_factors
