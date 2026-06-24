"""Train the XGBoost credit-risk model on the full Home Credit relational data.

Run once:
    python -m src.ml.train

Pipeline:
  1. Fetch the real application table (``get_data.ensure_dataset``) and the
     auxiliary relational tables (``get_data.ensure_auxiliary``).
  2. Aggregate bureau / previous-application / installment / POS / credit-card
     history per applicant (``aggregations.build_auxiliary_features``).
  3. Engineer the feature matrix (``features.engineer_features``) and train
     XGBoost with early stopping; report held-out ROC-AUC.
  4. Persist the model, SHAP explainer, per-feature medians (for single-app
     inference imputation), metadata, and the PSI drift reference.

If the auxiliary tables cannot be obtained, training proceeds application-only
and records that fact in the metadata (no silent claim of the higher AUC).

Outputs:
    models/xgboost_risk.pkl
    models/shap_explainer.pkl
    models/feature_medians.json
    models/model_metadata.json
"""
import json
import logging
import os

import joblib
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.ml.aggregations import auxiliary_available, build_auxiliary_features
from src.ml.features import engineer_features, RAW_COLUMNS, FEATURE_COLUMNS
from src.ml.get_data import ensure_dataset, ensure_auxiliary, read_source

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "models"
MODEL_VERSION = "xgb-v2.0"


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    path, n_rows, is_real = ensure_dataset()
    src_info = read_source()
    logger.info("Data source: %s (real=%s, rows=%s)", src_info["source"], is_real, f"{n_rows:,}")
    if not is_real:
        logger.warning("TRAINING ON SYNTHETIC FALLBACK DATA — metrics are illustrative only.")

    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS).dropna(subset=["TARGET"])
    logger.info("Loaded %s rows. Default rate: %.3f", f"{len(df):,}", df["TARGET"].mean())

    # Auxiliary relational history (the AUC lever). Application-only if unavailable.
    ensure_auxiliary()
    use_aux = auxiliary_available()
    aux = build_auxiliary_features() if use_aux else None
    if use_aux:
        logger.info("Auxiliary relational features: ON (%d applicants)", len(aux))
    else:
        logger.warning("Auxiliary tables unavailable — training APPLICATION-ONLY.")

    y = df["TARGET"].astype(int)
    X = engineer_features(df.drop(columns=["TARGET"]), aux=aux)

    # Per-feature medians: impute auxiliary (and any missing app) features at
    # single-application inference time. Saved for the serving path.
    medians = X.median(numeric_only=True).round(6).to_dict()
    with open(os.path.join(MODEL_DIR, "feature_medians.json"), "w") as f:
        json.dump(medians, f, indent=2)
    logger.info("Saved feature medians (%d features).", len(medians))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %s | Val: %s", f"{len(X_train):,}", f"{len(X_val):,}")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=3000, max_depth=6, learning_rate=0.02,
        subsample=0.85, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=2.0,
        min_child_weight=30, scale_pos_weight=scale_pos_weight,
        eval_metric="auc", early_stopping_rounds=150,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    logger.info("Training XGBoost on %d features ...", len(FEATURE_COLUMNS))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    logger.info("Held-out ROC-AUC: %.4f", auc)

    try:
        from src.drift import build_reference
        build_reference({
            "risk_probability": val_probs.tolist(),
            "amt_income_total": X_val["AMT_INCOME_TOTAL"].tolist(),
            "amt_credit": X_val["AMT_CREDIT"].tolist(),
            "amt_annuity": X_val["AMT_ANNUITY"].tolist(),
        })
        logger.info("Drift reference saved.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not build drift reference: %s", exc)

    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_risk.pkl"))
    logger.info("Building SHAP explainer ...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))

    best_iter = getattr(model, "best_iteration", None)
    metadata = {
        "model_version": MODEL_VERSION,
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "training_auc": round(float(auc), 4),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "data_source": src_info["source"] + (" + relational(bureau/prev/installments/pos/cc)" if use_aux else ""),
        "uses_relational_features": bool(use_aux),
        "trained_on_real_data": bool(is_real),
        "decision_thresholds": {
            "LOW_max": 0.30, "MEDIUM_max": 0.55, "HIGH_max": 0.75, "DECLINE_min": 0.75,
        },
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved. Held-out AUC: %.4f. Training complete.", auc)


if __name__ == "__main__":
    train()
