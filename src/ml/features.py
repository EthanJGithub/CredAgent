"""Feature engineering for the credit-risk model.

This module is the single source of truth for the feature contract. The exact
``FEATURE_COLUMNS`` order is what the trained XGBoost model expects at
inference time — reordering silently produces wrong predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

# Columns read from the raw Home-Credit-style dataset.
RAW_COLUMNS = [
    "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
]

# Final feature list after engineering — ORDER MATTERS for inference.
FEATURE_COLUMNS = [
    "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "CODE_GENDER_F", "CODE_GENDER_M",
    "debt_to_income", "employment_months", "age_years",
    "credit_to_income_ratio", "annuity_to_credit_ratio",
    "has_income_stability",
    "NAME_INCOME_TYPE_Working", "NAME_INCOME_TYPE_Commercial_associate",
    "NAME_EDUCATION_TYPE_Higher_education",
]

# Clip bounds drawn from the Home Credit training distribution.
CLIP_BOUNDS = {
    "AMT_CREDIT": (45000, 4050000),
    "AMT_INCOME_TOTAL": (26100, 1575000),
    "AMT_ANNUITY": (1615, 258025),
    "DAYS_BIRTH": (-25229, -7489),
    "DAYS_EMPLOYED": (-17912, 0),
}

# Medians used to impute the external bureau scores.
EXT_SOURCE_MEDIANS = {"EXT_SOURCE_1": 0.502, "EXT_SOURCE_2": 0.514, "EXT_SOURCE_3": 0.511}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw dataframe (subset of ``RAW_COLUMNS``) into the model matrix."""
    df = df.copy()

    # DAYS_EMPLOYED sentinel 365243 means "unemployed" → treat as 0.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].clip(*CLIP_BOUNDS["DAYS_EMPLOYED"])

    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns and col != "DAYS_EMPLOYED":
            df[col] = df[col].clip(lo, hi)

    for col, median in EXT_SOURCE_MEDIANS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    inc = df["AMT_INCOME_TOTAL"] + 1
    df["debt_to_income"] = df["AMT_ANNUITY"] / inc
    df["employment_months"] = (df["DAYS_EMPLOYED"].abs() / 30).clip(0, 480)
    df["age_years"] = (df["DAYS_BIRTH"].abs() / 365).clip(18, 70)
    df["credit_to_income_ratio"] = df["AMT_CREDIT"] / inc
    df["annuity_to_credit_ratio"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)

    df["has_income_stability"] = df["NAME_INCOME_TYPE"].isin(
        ["Working", "Commercial associate"]
    ).astype(int)

    df["CODE_GENDER_F"] = (df["CODE_GENDER"] == "F").astype(int)
    df["CODE_GENDER_M"] = (df["CODE_GENDER"] == "M").astype(int)

    df["NAME_INCOME_TYPE_Working"] = (df["NAME_INCOME_TYPE"] == "Working").astype(int)
    df["NAME_INCOME_TYPE_Commercial_associate"] = (
        df["NAME_INCOME_TYPE"] == "Commercial associate"
    ).astype(int)
    df["NAME_EDUCATION_TYPE_Higher_education"] = (
        df["NAME_EDUCATION_TYPE"] == "Higher education"
    ).astype(int)

    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if df[col].dtype == object:
            df[col] = (df[col] == "Y").astype(int)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["CNT_CHILDREN"] = pd.to_numeric(df["CNT_CHILDREN"], errors="coerce").fillna(0).astype(int)

    return df[FEATURE_COLUMNS]


def engineer_single_application(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Engineer features for one application dict (used at inference time)."""
    normalized = {k.upper(): v for k, v in raw.items()}
    df = pd.DataFrame([normalized])

    for col in RAW_COLUMNS:
        if col == "TARGET":
            continue
        if col not in df.columns:
            df[col] = None

    engineered = engineer_features(df)
    return engineered.iloc[0].to_dict()
