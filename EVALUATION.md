**Evaluation protocol — September 2026**

Training/validation/final-test are stratified 60/20/20 splits (seeds 42, 43). Only validation selects early stopping. This is a within-dataset estimate, not prospective customer validation; the underlying public dataset has been explored previously. No claim of a never-before-seen benchmark dataset is made.

Uncalibrated class-weighted model scores must not be interpreted as validated real-world probabilities. Temporal/customer-disjoint evaluation and independent deployment validation remain future work.

```json
{
  "model_version": "xgb-v3.0",
  "n_features": 81,
  "training_auc": 0.7782,
  "validation_auc": 0.7782,
  "test_auc_full_history": 0.7803,
  "test_auc_imputed_history": 0.7546,
  "test_brier_full_history": 0.16310206055641174,
  "test_brier_imputed_history": 0.13202330470085144,
  "probability_calibrated": false,
  "score_interpretation": "uncalibrated model risk score; not a calibrated default probability",
  "evaluation_protocol": "stratified 60/20/20; medians fit on training; early stopping on validation; test reserved for final full-history and imputed-history evaluation",
  "n_test": 61503,
  "n_train": 184506,
  "n_val": 61502,
  "data_source": "huggingface:jlh/home-credit + relational(bureau/prev/installments/pos/cc)",
  "uses_relational_features": true,
  "trained_on_real_data": true,
  "decision_thresholds": {
    "LOW_max": 0.3,
    "MEDIUM_max": 0.55,
    "HIGH_max": 0.75,
    "DECLINE_min": 0.75
  },
  "best_iteration": 1130,
  "test_auc_form_inputs": 0.7459,
  "test_brier_form_inputs": 0.11579412966966629
}
```
