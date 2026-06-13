"""Generate a synthetic, Home-Credit-shaped training dataset.

The real project trains on Kaggle's *Home Credit Default Risk*
``application_train.csv`` (307K rows, requires a Kaggle account). To keep the
pipeline reproducible end-to-end with **no external download**, this script
emits a CSV with the same schema (``RAW_COLUMNS``) and realistic, *learnable*
signal: default probability is a logistic function of the bureau scores,
debt-to-income, employment length and income type, plus noise. A model trained
on it reaches a validation AUC around 0.74–0.78 — comparable to the real data
with these features.

Run:
    python -m src.ml.make_synthetic_data --rows 60000

The output is written to ``data/raw/application_train.csv``. Dropping the real
Kaggle file at that path instead is fully supported — ``train.py`` does not
care which one it reads.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OUT_PATH = "data/raw/application_train.csv"

INCOME_TYPES = [
    "Working", "Commercial associate", "Pensioner", "State servant",
    "Unemployed", "Student", "Businessman", "Maternity leave",
]
INCOME_TYPE_P = [0.52, 0.23, 0.16, 0.06, 0.012, 0.008, 0.005, 0.005]

EDU_TYPES = [
    "Secondary / secondary special", "Higher education", "Incomplete higher",
    "Lower secondary", "Academic degree",
]
EDU_TYPE_P = [0.71, 0.24, 0.034, 0.014, 0.002]

# Income types that carry extra default risk in the latent model.
RISKY_INCOME = {"Unemployed": 1.4, "Student": 0.6, "Maternity leave": 0.5}


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Raw demographic / financial draws ───────────────────────────────────
    age_years = rng.normal(43, 11, rows).clip(21, 69)
    days_birth = (-age_years * 365).astype(int)

    income = rng.lognormal(mean=np.log(165000), sigma=0.45, size=rows).clip(26100, 1_575_000)
    credit = (income * rng.lognormal(mean=np.log(3.2), sigma=0.5, size=rows)).clip(45000, 4_050_000)
    annuity = (credit / rng.uniform(18, 60, rows)).clip(1615, 258025)

    income_type = rng.choice(INCOME_TYPES, size=rows, p=INCOME_TYPE_P)
    edu_type = rng.choice(EDU_TYPES, size=rows, p=EDU_TYPE_P)

    employed_years = np.where(
        income_type == "Pensioner",
        0.0,
        rng.exponential(6.0, rows).clip(0, 48),
    )
    days_employed = np.where(
        income_type == "Unemployed",
        365243,                      # Home Credit's "unemployed" sentinel
        (-employed_years * 365).astype(int),
    )

    gender = rng.choice(["M", "F"], size=rows, p=[0.34, 0.66])
    own_car = rng.binomial(1, 0.34, rows)
    own_realty = rng.binomial(1, 0.69, rows)
    children = rng.poisson(0.42, rows).clip(0, 12)

    # External bureau scores: correlated with (lower) latent risk.
    base_quality = rng.normal(0, 1, rows)
    def _score(noise_sd):
        return (0.5 + 0.16 * base_quality + rng.normal(0, noise_sd, rows)).clip(0.01, 0.99)
    ext1 = _score(0.18)
    ext2 = _score(0.16)
    ext3 = _score(0.17)
    # Mimic real-world missingness in EXT_SOURCE_1 / _3.
    ext1[rng.random(rows) < 0.40] = np.nan
    ext3[rng.random(rows) < 0.18] = np.nan

    # ── Latent default logit ────────────────────────────────────────────────
    dti = annuity / (income + 1)
    cti = credit / (income + 1)
    emp_months = np.minimum(np.abs(np.where(days_employed > 0, 0, days_employed)) / 30, 480)

    ext_mean = np.nanmean(np.vstack([ext1, ext2, ext3]), axis=0)
    income_risk = np.array([RISKY_INCOME.get(t, 0.0) for t in income_type])

    logit = (
        -2.35                                  # base rate ~ 8%
        - 3.1 * (ext_mean - 0.5)               # strong protective effect of bureau scores
        + 6.0 * (dti - 0.025)                  # debt-to-income
        + 0.18 * (cti - 3.0)                   # credit-to-income
        - 0.010 * emp_months                   # tenure protects
        - 0.011 * (age_years - 43)             # older = slightly safer
        + 0.08 * children
        + income_risk
        - 0.12 * own_realty
        + rng.normal(0, 0.45, rows)            # irreducible noise
    )
    p_default = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, p_default)

    df = pd.DataFrame({
        "TARGET": target,
        "AMT_CREDIT": credit.round(0),
        "AMT_INCOME_TOTAL": income.round(0),
        "AMT_ANNUITY": annuity.round(0),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": ext1.round(4),
        "EXT_SOURCE_2": ext2.round(4),
        "EXT_SOURCE_3": ext3.round(4),
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": np.where(own_car == 1, "Y", "N"),
        "FLAG_OWN_REALTY": np.where(own_realty == 1, "Y", "N"),
        "CNT_CHILDREN": children,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": edu_type,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Home-Credit-shaped data.")
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=OUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    logger.info(
        "Wrote %s rows to %s | default rate: %.3f",
        f"{len(df):,}", args.out, df["TARGET"].mean(),
    )


if __name__ == "__main__":
    main()
