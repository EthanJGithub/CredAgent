"""Train the XGBoost credit-risk model.

Run once:
    python -m src.ml.train

Reads ``data/raw/application_train.csv`` (either the real Kaggle file or the
synthetic one from ``src.ml.make_synthetic_data``). If the file is missing, a
synthetic dataset is generated automatically so the command never fails.

Outputs:
    models/xgboost_risk.pkl
    models/shap_explainer.pkl
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

from src.ml.features import engineer_features, RAW_COLUMNS, FEATURE_COLUMNS
from src.ml.get_data import ensure_dataset, DATA_PATH, read_source

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "models"


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Always prefer the real Home Credit dataset; synthetic is a last resort.
    path, n_rows, is_real = ensure_dataset()
    src_info = read_source()
    logger.info("Data source: %s (real=%s, rows=%s)", src_info["source"], is_real, f"{n_rows:,}")
    if not is_real:
        logger.warning("TRAINING ON SYNTHETIC FALLBACK DATA — metrics are illustrative only.")

    # Only read the columns we need (works for the full 122-column file too).
    df = pd.read_csv(path, usecols=lambda c: c in RAW_COLUMNS).dropna(subset=["TARGET"])
    logger.info("Loaded %s rows. Default rate: %.3f", f"{len(df):,}", df["TARGET"].mean())

    y = df["TARGET"].astype(int)
    X = engineer_features(df.drop(columns=["TARGET"]))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %s | Val: %s", f"{len(X_train):,}", f"{len(X_val):,}")

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    logger.info("scale_pos_weight: %.2f", scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    logger.info("Validation ROC-AUC: %.4f", auc)

    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_risk.pkl"))
    logger.info("Model saved.")

    logger.info("Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
    logger.info("SHAP explainer saved.")

    best_iter = getattr(model, "best_iteration", None)
    metadata = {
        "model_version": "xgb-v1.0",
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "training_auc": round(float(auc), 4),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "data_source": src_info["source"],
        "trained_on_real_data": bool(is_real),
        "decision_thresholds": {
            "LOW_max": 0.30, "MEDIUM_max": 0.55, "HIGH_max": 0.75, "DECLINE_min": 0.75,
        },
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved. AUC: %.4f. Training complete.", auc)


if __name__ == "__main__":
    train()
