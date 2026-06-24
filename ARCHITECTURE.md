# Architecture — CredAgent

> How CredAgent is built and **why** each component was chosen. This document is
> written for an engineering reviewer evaluating production-readiness, not just
> "does it run." Compliance obligations are documented separately in
> [COMPLIANCE.md](COMPLIANCE.md).

---

## 1. System at a glance

CredAgent is a **five-agent [LangGraph](https://langchain-ai.github.io/langgraph/)
pipeline** that takes a loan application and returns an APPROVE / DECLINE / REFER
decision, a SHAP-attributed risk explanation, a CFPB fair-lending review, and —
for declines — a Regulation B-compliant adverse-action notice. Every decision is
persisted with its full evidence bundle so it is reproducible after the fact.

```
Application ─▶ Ingestion ─▶ RiskScoring ─▶ PolicyCompliance ─▶ Decision ─▶ Audit ─▶ Persisted decision + evidence
              (validate)   (XGBoost+SHAP)  (RAG over CFPB)     (APPROVE/    (adverse
                                                                DECLINE/     notice +
                                            MEDIUM ─▶ HITL ◀──  REFER)       audit trail)
```

| Layer | Component | Path |
|-------|-----------|------|
| Orchestration | LangGraph state machine | [src/graph/workflow.py](src/graph/workflow.py), [src/graph/state.py](src/graph/state.py) |
| Agents | Ingestion / RiskScoring / PolicyCompliance / Decision / Audit | [src/agents/](src/agents/) |
| Model | XGBoost risk model + SHAP explainer | [src/ml/](src/ml/), [models/](models/) |
| Policy retrieval | ChromaDB RAG over CFPB corpus | [src/rag/](src/rag/), [vectorstore/](vectorstore/) |
| LLM abstraction | Groq → Anthropic → offline template | [src/llm.py](src/llm.py) |
| Persistence | SQLite decision store + AI Evidence Hub | [src/store.py](src/store.py) |
| Monitoring | PSI drift + disparate-impact testing | [src/drift.py](src/drift.py), [src/store.py](src/store.py) |
| Serving | FastAPI API + Streamlit dashboard | [src/api/](src/api/), [streamlit_app.py](streamlit_app.py) |

---

## 2. Why these choices (the defensible decisions)

### Why LangGraph (not a plain function chain or a free-form agent)
Underwriting is a **regulated, auditable decision** — the sequence of steps and
the state at each step must be reproducible. LangGraph models the pipeline as an
explicit **state machine** ([CreditDecisionState](src/graph/state.py)) with
deterministic transitions, rather than an open-ended "agent decides what to do
next" loop. This buys three things a regulator cares about:

1. **Determinism** — the same application traverses the same nodes in the same
   order every time. There is no nondeterministic tool-selection step in the
   decision path.
2. **Inspectable state** — every agent reads and writes a typed state object, so
   the intermediate artifacts (features, SHAP values, retrieved policy text,
   LLM prompts) are all capturable for the audit trail.
3. **First-class human-in-the-loop** — LangGraph's interrupt mechanism lets the
   graph *pause* at a MEDIUM-risk decision and resume after a human verdict,
   without losing state. HITL is a graph primitive here, not a bolt-on.

### Why XGBoost (not a deep model or an LLM-as-scorer)
The risk model is a gradient-boosted tree because, for tabular credit data, it is
**the proven, well-understood tool**: strong performance on the real Home Credit
dataset (held-out ROC-AUC **0.7815**, 5-fold CV **0.7818 ± 0.0030**,
[models/model_metadata.json](models/model_metadata.json)), fast inference, and —
critically — **exact SHAP attribution**. An LLM is never used to produce the risk
score. The LLM's role is confined to *narrating* and *policy-checking* decisions
the deterministic model has already made. This is the core senior-signal stance:
**the LLM is only as powerful as the guardrails around it.**

### Why relational aggregations (the AUC lever)
An application-table-only model plateaus near **0.76** — the external bureau scores
dominate and additional application fields add little (measured: 0.765 with 45
tuned application features). Real underwriting pulls the applicant's **credit
history**, so the model aggregates the Home Credit relational tables
([src/ml/aggregations.py](src/ml/aggregations.py)) — bureau credit lines, prior
applications, installment-payment days-past-due / shortfalls, POS & credit-card
delinquency — into 37 per-applicant features. That is what lifts AUC to **0.78**.
The aggregation is a training-time step over large raw tables (gitignored); only
the trained model and a `feature_medians.json` ship. At single-application
inference the relational features are imputed to medians and **excluded from cited
adverse-action reasons** ([src/ml/explainer.py](src/ml/explainer.py)) — documented,
not hidden (see [COMPLIANCE.md](COMPLIANCE.md)).

### Why exclude geography (a deliberate AUC trade-off)
Region/location features are predictive but are a recognised **redlining /
disparate-impact proxy**. We exclude them (alongside sex and education), which
costs a little AUC versus uncapped leaderboard solutions — a trade made on purpose
for fair-lending defensibility.

### Why SHAP (not feature importances or model-free explanations)
ECOA / Regulation B requires a creditor to disclose the **specific principal
reasons** for an adverse action. Generic global feature importances do not
satisfy this — the reason must be specific *to this applicant*. SHAP provides
per-prediction, locally-accurate attributions, which the AuditAgent converts
directly into the disclosable reasons on the adverse-action notice. The
explanation is therefore a **legal artifact derived from the model math**, not a
post-hoc LLM rationalization.

### Why RAG over a hardcoded rules engine (PolicyCompliance)
Fair-lending policy is text-heavy and evolving. The PolicyComplianceAgent
retrieves relevant CFPB / ECOA / FCRA passages from a **ChromaDB** vector store
([src/rag/](src/rag/)) and grounds the LLM's compliance check in that retrieved
text, rather than baking brittle if/else rules into code. The retrieved excerpts
are persisted into the evidence bundle, so a reviewer can see *which* policy text
informed the check.

### Why the Groq → Anthropic → offline LLM ladder
The pipeline must run identically in CI, in local dev with no API key, and on the
hosted demo. [src/llm.py](src/llm.py) resolves a backend in a fixed order and
exposes a uniform `.invoke()` interface. The **offline** backend is a
deterministic, clearly-labelled template engine — every offline artifact is
stamped `[Generated offline — set GROQ_API_KEY for LLM-authored text.]`. This
honors the **no-silent-fallback** principle: a degraded path is never presented
as the real thing.

### Why SQLite (not Postgres, yet)
The decision store ([src/store.py](src/store.py)) is SQLite in WAL mode. It is
durable, transactional, zero-ops, and seeds reproducibly for the demo. The schema
and access patterns (upsert by `applicant_id`, append-only evidence rows) are
written so the store is **portable to Postgres** without changing the agent code —
the store module is the only place that touches SQL.

---

## 3. The five agents

| Agent | Responsibility | Key outputs into state |
|-------|----------------|------------------------|
| **IngestionAgent** | Validate the raw application; engineer features (debt-to-income, employment length, credit ratios). | `cleaned_features`, `derived_features` |
| **RiskScoringAgent** | Score default probability with XGBoost; attribute with SHAP; map to a risk tier. | `risk_probability`, `risk_tier`, `shap_values`, `top_risk_factors` |
| **PolicyComplianceAgent** | Retrieve relevant CFPB text (RAG) and LLM-check the decision for disparate-impact / ECOA / FCRA concerns. | `retrieved_policy_excerpts`, `compliance_flags` |
| **DecisionAgent** | Issue APPROVE / DECLINE / REFER with plain-English reasoning; route MEDIUM tier to human review. | `final_decision`, `requires_human_review` |
| **AuditAgent** | For declines, generate a Reg-B adverse-action notice; assemble a timestamped audit trail. | `adverse_action_notice`, `audit_trail`, `processing_time_ms` |

### Decision tiers (real thresholds, [models/model_metadata.json](models/model_metadata.json))
| Default probability | Tier | Routing |
|---------------------|------|---------|
| < 0.30 | LOW | APPROVE |
| 0.30 – 0.55 | MEDIUM | **REFER → human-in-the-loop** |
| 0.55 – 0.75 | HIGH | typically DECLINE |
| ≥ 0.75 | DECLINE | DECLINE + adverse-action notice |

---

## 4. Observability & monitoring

- **Latency** — the AuditAgent computes end-to-end `processing_time_ms` from the
  request timestamp and logs it per decision ([src/agents/audit_agent.py](src/agents/audit_agent.py)).
- **Model drift (PSI)** — [src/drift.py](src/drift.py) computes the Population
  Stability Index for the risk score and key inputs against a reference window
  captured at train time ([models/drift_reference.json](models/drift_reference.json)),
  with the conventional stable / moderate / significant bands (0.10 / 0.25).
- **Fair-lending monitoring** — kept *separate* from the scoring model. The store
  runs a four-fifths-rule disparate-impact test across gender groups
  ([src/store.py](src/store.py) `disparate_impact`). See [COMPLIANCE.md](COMPLIANCE.md).
- **AI Evidence Hub** — every decision persists model version, exact LLM prompts,
  feature inputs, SHAP attribution, and retrieved policy text, so any decision is
  fully reconstructable.

---

## 5. Data flow & persistence

1. Application enters the graph; IngestionAgent validates and engineers features.
2. RiskScoring → PolicyCompliance → Decision run in sequence over shared state.
3. MEDIUM-tier decisions interrupt for human review and resume in place (the
   store upserts by `applicant_id`).
4. AuditAgent finalizes; `store.log_decision()` writes the decision row **and**
   the evidence bundle in one call.
5. Monitoring views (drift, disparate impact, portfolio summary) read from the
   store on demand.

---

## 6. Known limitations / roadmap

These are stated plainly rather than hidden — see [COMPLIANCE.md §6](COMPLIANCE.md)
for the governance roadmap.

- **Access control** is designed-for but not enforced: the API has no authn/authz
  layer yet (single-tenant demo). The store schema and endpoints are shaped to
  add role-based access without restructuring.
- **Persistence** is SQLite; a multi-writer production deployment would move to
  Postgres (the store module isolates this).
- **Drift alerting** is computed on-demand and surfaced in the dashboard; it is
  not yet wired to an external alerting channel (PagerDuty/Slack).
- Disparate-impact testing currently covers the gender axis captured in the demo
  data; additional protected bases would extend the same separated-monitoring
  pattern.
