import pandas as pd

from src.ml.features import engineer_single_application, FEATURE_COLUMNS


def test_feature_engineering_produces_full_vector():
    raw = {
        "amt_credit": 239850.0, "amt_income_total": 157500.0, "amt_annuity": 23494.5,
        "days_birth": -12967, "days_employed": -1996,
        "ext_source_1": 0.838, "ext_source_2": 0.356, "ext_source_3": 0.608,
        "code_gender": "F", "flag_own_car": 0, "flag_own_realty": 1, "cnt_children": 0,
        "name_income_type": "Working", "name_education_type": "Secondary / secondary special",
    }
    feats = engineer_single_application(raw)
    assert set(feats.keys()) == set(FEATURE_COLUMNS)
    assert feats["CODE_GENDER_F"] == 1 and feats["CODE_GENDER_M"] == 0
    assert feats["has_income_stability"] == 1
    assert feats["debt_to_income"] > 0


def test_missing_ext_sources_imputed():
    raw = {
        "amt_credit": 100000.0, "amt_income_total": 50000.0, "amt_annuity": 5000.0,
        "days_birth": -12000, "days_employed": -1000,
        "code_gender": "M", "flag_own_car": 0, "flag_own_realty": 0, "cnt_children": 1,
        "name_income_type": "Working", "name_education_type": "Higher education",
    }
    feats = engineer_single_application(raw)
    # imputed with training medians, never NaN
    assert pd.notna(feats["EXT_SOURCE_1"])
    assert pd.notna(feats["EXT_SOURCE_3"])


def test_unemployed_sentinel_handled():
    raw = {
        "amt_credit": 100000.0, "amt_income_total": 50000.0, "amt_annuity": 5000.0,
        "days_birth": -12000, "days_employed": 365243,  # unemployed sentinel
        "code_gender": "M", "flag_own_car": 0, "flag_own_realty": 0, "cnt_children": 0,
        "name_income_type": "Unemployed", "name_education_type": "Lower secondary",
    }
    feats = engineer_single_application(raw)
    assert feats["employment_months"] == 0
    assert feats["has_income_stability"] == 0
