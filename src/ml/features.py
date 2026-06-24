"""Feature engineering for the credit-risk model.

This module is the single source of truth for the feature contract. The exact
``FEATURE_COLUMNS`` order is what the trained XGBoost model expects at inference
time — reordering silently produces wrong predictions.

The model is trained on the **full Home Credit relational dataset**: the
application table PLUS aggregations of the applicant's credit-bureau history,
prior applications, and installment-payment behaviour. This is how real
underwriting works (a lender pulls bureau data), and it is what lifts AUC well
above an application-only model.

Two feature groups:
  * **APPLICANT_FEATURES** — derivable from a single application form
    (amounts, external scores, employment, age). The interactive demo collects
    these and they drive the live prediction.
  * **AUX_FEATURES** — aggregations of the applicant's relational history
    (bureau / previous applications / installments / POS / credit card). At
    training/eval time these are real. In the interactive demo, which has no
    access to a stranger's credit history, they are imputed to the training
    medians (see ``models/feature_medians.json``) — documented, not hidden.

Fair-lending exclusions from the SCORING model (kept out of every group):
  * Sex (CODE_GENDER): a prohibited basis under ECOA (15 U.S.C. 1691 / Reg B).
  * Education (NAME_EDUCATION_TYPE): a documented proxy for race / national
    origin that the CFPB scrutinises in disparate-impact analysis.
  * Geography (REGION_*/ all location features): excluded to avoid redlining /
    location-based disparate-impact proxies.
Sex is still collected at intake and retained for *post-decision* fair-lending
monitoring — the standard separation between scoring and disparate-impact testing.
"""
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Raw application columns read from the Home-Credit dataset (incl. the join key).
RAW_COLUMNS = [
    "SK_ID_CURR", "TARGET",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "OWN_CAR_AGE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE", "FLAG_PHONE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
]

# Application-level features (computable from one application form).
APPLICANT_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_mean", "EXT_SOURCE_min", "EXT_SOURCE_max", "EXT_SOURCE_std", "EXT_SOURCE_prod",
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "debt_to_income", "credit_to_income_ratio", "annuity_to_credit_ratio",
    "credit_to_goods", "income_per_person", "payment_rate",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "employed_to_age", "employment_months", "age_years",
    "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "OWN_CAR_AGE",
    "ext_mean_x_age",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE", "FLAG_PHONE",
    "has_income_stability",
    "NAME_INCOME_TYPE_Working", "NAME_INCOME_TYPE_Commercial_associate", "NAME_INCOME_TYPE_Pensioner",
]

# Auxiliary (relational-history) features — imputed to medians at single-app
# inference time, real at training/eval time.
AUX_FEATURES = [
    # bureau (credit-bureau history)
    "BU_CNT", "BU_ACTIVE", "BU_DAYS_CREDIT_mean", "BU_DAYS_CREDIT_min", "BU_ENDDATE_max",
    "BU_OVERDUE_mean", "BU_OVERDUE_max", "BU_AMT_SUM", "BU_AMT_DEBT_sum", "BU_AMT_OVERDUE_sum",
    "BU_PROLONG_sum", "BU_BB_DPD_mean", "BU_DEBT_RATIO", "BU_ACTIVE_RATIO",
    # previous Home Credit applications
    "PREV_CNT", "PREV_APPROVED", "PREV_REFUSED", "PREV_AMT_APP_mean", "PREV_AMT_CREDIT_mean",
    "PREV_CNT_PAYMENT_mean", "PREV_DOWN_mean", "PREV_DAYS_DECISION_max",
    "PREV_REFUSED_RATIO", "PREV_APPROVED_RATIO",
    # installment-payment behaviour (days past due, payment shortfall)
    "INS_CNT", "INS_DPD_mean", "INS_DPD_max", "INS_DBD_mean",
    "INS_PAYRATIO_mean", "INS_PAYRATIO_min", "INS_SHORT_mean", "INS_LATE_mean",
    # POS cash + credit card delinquency
    "POS_DPD_mean", "POS_DPD_max", "POS_DPDDEF_mean", "CC_DPD_mean", "CC_UTIL_mean",
]

# Final feature list — ORDER MATTERS for inference.
FEATURE_COLUMNS = APPLICANT_FEATURES + AUX_FEATURES

# Clip bounds drawn from the Home Credit training distribution.
CLIP_BOUNDS = {
    "AMT_CREDIT": (45000, 4050000),
    "AMT_INCOME_TOTAL": (26100, 1575000),
    "AMT_ANNUITY": (1615, 258025),
    "DAYS_BIRTH": (-25229, -7489),
    "DAYS_EMPLOYED": (-17912, 0),
}

# Medians used to impute the external bureau scores when absent.
EXT_SOURCE_MEDIANS = {"EXT_SOURCE_1": 0.502, "EXT_SOURCE_2": 0.514, "EXT_SOURCE_3": 0.511}

_MEDIANS_PATH = os.getenv("CREDAGENT_MEDIANS", "models/feature_medians.json")
_medians_cache: Optional[Dict[str, float]] = None


def load_medians() -> Dict[str, float]:
    """Per-feature training medians, used to impute auxiliary (and any missing
    application) features at single-application inference time."""
    global _medians_cache
    if _medians_cache is None:
        if os.path.exists(_MEDIANS_PATH):
            with open(_MEDIANS_PATH) as f:
                _medians_cache = json.load(f)
        else:
            _medians_cache = {}
    return _medians_cache


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the APPLICANT_FEATURES from a raw application dataframe.

    Robust to missing columns (fills numerics with NaN so XGBoost can handle
    them, or with sensible derivations). Used both in training (full data) and
    inference (single application)."""
    df = df.copy()
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # DAYS_EMPLOYED sentinel 365243 means "unemployed" → 0.
    df["DAYS_EMPLOYED"] = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce").replace(365243, 0)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].clip(*CLIP_BOUNDS["DAYS_EMPLOYED"])
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col != "DAYS_EMPLOYED":
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)

    for col, median in EXT_SOURCE_MEDIANS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    out = pd.DataFrame(index=df.index)
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for c in ext:
        out[c] = df[c]
    out["EXT_SOURCE_mean"] = out[ext].mean(axis=1)
    out["EXT_SOURCE_min"] = out[ext].min(axis=1)
    out["EXT_SOURCE_max"] = out[ext].max(axis=1)
    out["EXT_SOURCE_std"] = out[ext].std(axis=1).fillna(0)
    out["EXT_SOURCE_prod"] = out["EXT_SOURCE_1"] * out["EXT_SOURCE_2"] * out["EXT_SOURCE_3"]

    inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") + 1
    credit = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
    annuity = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
    goods = pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce").fillna(credit)
    fam = pd.to_numeric(df["CNT_FAM_MEMBERS"], errors="coerce")
    out["AMT_CREDIT"] = credit
    out["AMT_INCOME_TOTAL"] = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce")
    out["AMT_ANNUITY"] = annuity
    out["AMT_GOODS_PRICE"] = goods
    out["debt_to_income"] = annuity / inc
    out["credit_to_income_ratio"] = credit / inc
    out["annuity_to_credit_ratio"] = annuity / (credit + 1)
    out["credit_to_goods"] = credit / (goods + 1)
    out["income_per_person"] = out["AMT_INCOME_TOTAL"] / (fam.fillna(1) + 1)
    out["payment_rate"] = annuity / (credit + 1)

    db = pd.to_numeric(df["DAYS_BIRTH"], errors="coerce")
    de = df["DAYS_EMPLOYED"]
    out["DAYS_BIRTH"] = db
    out["DAYS_EMPLOYED"] = de
    out["employed_to_age"] = de / db
    out["employment_months"] = (de.abs() / 30).clip(0, 480)
    out["age_years"] = (db.abs() / 365).clip(18, 70)
    out["DAYS_REGISTRATION"] = pd.to_numeric(df["DAYS_REGISTRATION"], errors="coerce")
    out["DAYS_ID_PUBLISH"] = pd.to_numeric(df["DAYS_ID_PUBLISH"], errors="coerce")
    out["DAYS_LAST_PHONE_CHANGE"] = pd.to_numeric(df["DAYS_LAST_PHONE_CHANGE"], errors="coerce")
    out["OWN_CAR_AGE"] = pd.to_numeric(df["OWN_CAR_AGE"], errors="coerce")
    out["ext_mean_x_age"] = out["EXT_SOURCE_mean"] * (db / -365)

    out["CNT_CHILDREN"] = pd.to_numeric(df["CNT_CHILDREN"], errors="coerce").fillna(0)
    out["CNT_FAM_MEMBERS"] = fam
    for c in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        s = df[c]
        out[c] = (s == "Y").astype(int) if s.dtype == object else pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    for c in ["DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE",
              "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
              "FLAG_DOCUMENT_3", "FLAG_EMP_PHONE", "FLAG_PHONE"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")

    inc_type = df["NAME_INCOME_TYPE"]
    out["has_income_stability"] = inc_type.isin(["Working", "Commercial associate"]).astype(int)
    out["NAME_INCOME_TYPE_Working"] = (inc_type == "Working").astype(int)
    out["NAME_INCOME_TYPE_Commercial_associate"] = (inc_type == "Commercial associate").astype(int)
    out["NAME_INCOME_TYPE_Pensioner"] = (inc_type == "Pensioner").astype(int)
    # CODE_GENDER / NAME_EDUCATION_TYPE / geography intentionally NOT encoded.
    return out[APPLICANT_FEATURES]


def engineer_features(df: pd.DataFrame, aux: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Full training matrix: application features joined to auxiliary aggregations.

    ``aux`` is a frame indexed by SK_ID_CURR (from ``aggregations.build_auxiliary_features``).
    When omitted, auxiliary features are filled with medians/NaN (single-table fallback)."""
    app = engineer_application_features(df)
    if aux is not None and "SK_ID_CURR" in df.columns:
        merged = df[["SK_ID_CURR"]].join(aux, on="SK_ID_CURR")
        for col in AUX_FEATURES:
            app[col] = merged[col].values if col in merged.columns else np.nan
    else:
        for col in AUX_FEATURES:
            app[col] = np.nan
    return app[FEATURE_COLUMNS]


def engineer_single_application(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Engineer the full feature vector for ONE application (inference).

    Application-level features come from the form; auxiliary relational-history
    features are imputed to the training medians (the demo cannot pull a
    stranger's credit history). Returns a dict keyed by ``FEATURE_COLUMNS``."""
    normalized = {k.upper(): v for k, v in raw.items()}
    df = pd.DataFrame([normalized])
    app = engineer_application_features(df).iloc[0].to_dict()

    medians = load_medians()
    row: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        if col in app and pd.notna(app[col]):
            row[col] = app[col]
        else:
            row[col] = medians.get(col, 0.0)
    return row
