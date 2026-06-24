"""Auxiliary relational-history aggregations for the credit-risk model.

The Home Credit dataset is relational: besides the application table, each
applicant (``SK_ID_CURR``) has a credit-bureau history, prior Home Credit
applications, and installment / POS / credit-card payment records. Summarising
these into per-applicant aggregations is what lifts model AUC well above an
application-only model — and mirrors how a real lender underwrites by pulling
bureau data.

``build_auxiliary_features(raw_dir)`` returns a DataFrame indexed by
``SK_ID_CURR`` with the columns in ``features.AUX_FEATURES``. It reads the raw
CSVs that ``get_data.ensure_auxiliary`` places in ``data/raw`` (gitignored — too
large to commit; only the trained model + medians ship).
"""
from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

AUX_FILES = [
    "bureau.csv", "bureau_balance.csv", "previous_application.csv",
    "installments_payments.csv", "POS_CASH_balance.csv", "credit_card_balance.csv",
]


def auxiliary_available(raw_dir: str = "data/raw") -> bool:
    """True when every auxiliary CSV needed for aggregation is present."""
    return all(os.path.exists(os.path.join(raw_dir, f)) for f in AUX_FILES)


def _bureau(raw_dir: str) -> pd.DataFrame:
    bureau = pd.read_csv(os.path.join(raw_dir, "bureau.csv"))
    bb = pd.read_csv(os.path.join(raw_dir, "bureau_balance.csv"))
    bb["dpd"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(BB_DPD=("dpd", "mean")).reset_index()
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    g = bureau.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "BU_CNT": g.size(),
        "BU_ACTIVE": g["CREDIT_ACTIVE"].apply(lambda s: (s == "Active").sum()),
        "BU_DAYS_CREDIT_mean": g["DAYS_CREDIT"].mean(),
        "BU_DAYS_CREDIT_min": g["DAYS_CREDIT"].min(),
        "BU_ENDDATE_max": g["DAYS_CREDIT_ENDDATE"].max(),
        "BU_OVERDUE_mean": g["CREDIT_DAY_OVERDUE"].mean(),
        "BU_OVERDUE_max": g["CREDIT_DAY_OVERDUE"].max(),
        "BU_AMT_SUM": g["AMT_CREDIT_SUM"].sum(),
        "BU_AMT_DEBT_sum": g["AMT_CREDIT_SUM_DEBT"].sum(),
        "BU_AMT_OVERDUE_sum": g["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BU_PROLONG_sum": g["CNT_CREDIT_PROLONG"].sum(),
        "BU_BB_DPD_mean": g["BB_DPD"].mean(),
    })
    out["BU_DEBT_RATIO"] = out["BU_AMT_DEBT_sum"] / (out["BU_AMT_SUM"] + 1)
    out["BU_ACTIVE_RATIO"] = out["BU_ACTIVE"] / (out["BU_CNT"] + 1)
    return out


def _previous(raw_dir: str) -> pd.DataFrame:
    prev = pd.read_csv(os.path.join(raw_dir, "previous_application.csv"))
    g = prev.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "PREV_CNT": g.size(),
        "PREV_APPROVED": g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Approved").sum()),
        "PREV_REFUSED": g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Refused").sum()),
        "PREV_AMT_APP_mean": g["AMT_APPLICATION"].mean(),
        "PREV_AMT_CREDIT_mean": g["AMT_CREDIT"].mean(),
        "PREV_CNT_PAYMENT_mean": g["CNT_PAYMENT"].mean(),
        "PREV_DOWN_mean": g["AMT_DOWN_PAYMENT"].mean(),
        "PREV_DAYS_DECISION_max": g["DAYS_DECISION"].max(),
    })
    out["PREV_REFUSED_RATIO"] = out["PREV_REFUSED"] / (out["PREV_CNT"] + 1)
    out["PREV_APPROVED_RATIO"] = out["PREV_APPROVED"] / (out["PREV_CNT"] + 1)
    return out


def _installments(raw_dir: str) -> pd.DataFrame:
    ins = pd.read_csv(os.path.join(raw_dir, "installments_payments.csv"))
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    ins["PAY_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins["SHORT"] = (ins["AMT_PAYMENT"] < ins["AMT_INSTALMENT"]).astype(int)
    ins["LATE"] = (ins["DPD"] > 0).astype(int)
    g = ins.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "INS_CNT": g.size(),
        "INS_DPD_mean": g["DPD"].mean(),
        "INS_DPD_max": g["DPD"].max(),
        "INS_DBD_mean": g["DBD"].mean(),
        "INS_PAYRATIO_mean": g["PAY_RATIO"].mean(),
        "INS_PAYRATIO_min": g["PAY_RATIO"].min(),
        "INS_SHORT_mean": g["SHORT"].mean(),
        "INS_LATE_mean": g["LATE"].mean(),
    })


def _pos_and_cc(raw_dir: str) -> pd.DataFrame:
    pos = pd.read_csv(os.path.join(raw_dir, "POS_CASH_balance.csv"),
                      usecols=["SK_ID_CURR", "SK_DPD", "SK_DPD_DEF"])
    pg = pos.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "POS_DPD_mean": pg["SK_DPD"].mean(),
        "POS_DPD_max": pg["SK_DPD"].max(),
        "POS_DPDDEF_mean": pg["SK_DPD_DEF"].mean(),
    })
    cc = pd.read_csv(os.path.join(raw_dir, "credit_card_balance.csv"),
                     usecols=["SK_ID_CURR", "SK_DPD", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"])
    cc["UTIL"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cg = cc.groupby("SK_ID_CURR")
    out = out.join(pd.DataFrame({"CC_DPD_mean": cg["SK_DPD"].mean(), "CC_UTIL_mean": cg["UTIL"].mean()}), how="outer")
    return out


def build_auxiliary_features(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Per-applicant auxiliary feature table indexed by SK_ID_CURR."""
    logger.info("Aggregating bureau ...")
    parts = [_bureau(raw_dir)]
    logger.info("Aggregating previous_application ...")
    parts.append(_previous(raw_dir))
    logger.info("Aggregating installments_payments ...")
    parts.append(_installments(raw_dir))
    logger.info("Aggregating POS + credit_card ...")
    parts.append(_pos_and_cc(raw_dir))
    aux = parts[0]
    for p in parts[1:]:
        aux = aux.join(p, how="outer")
    aux.index.name = "SK_ID_CURR"
    logger.info("Auxiliary features: %d applicants x %d columns", len(aux), aux.shape[1])
    return aux
