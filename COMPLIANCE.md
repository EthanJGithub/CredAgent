# CredAgent ? governance boundaries

CredAgent is an engineering prototype, not a certified lending product. Generated adverse-action notices are drafts for qualified review. SHAP and retrieval do not establish legal compliance, fairness or fitness for a lender.

Implemented: XGBoost scoring and SHAP attributions; recorded model versions, retrieved excerpts and audit sequence; selected human-review routing; SQLite decision persistence and distribution monitoring. Sex, education and geography fields are excluded from scoring.

Age-derived and other potentially sensitive/proxy features remain. Excluding selected columns does not establish the absence of discrimination. The four-fifths ratio is a diagnostic, not a complete fair-lending assessment. Scores are uncalibrated; thresholds are demonstration policy choices. Full-history, history-imputed and actual form-input evaluation are separated in [EVALUATION.md](EVALUATION.md).

Draft reasons must be checked against the actual decision and available inputs. Retrieval relevance needs independent domain evaluation. SQLite persistence is not cryptographic immutability. An operational deployment requires appropriate access controls, retention, durable review state and independent model-risk, privacy, security and legal review. This repository does not claim those reviews occurred. Use example inputs in the public demonstration.
