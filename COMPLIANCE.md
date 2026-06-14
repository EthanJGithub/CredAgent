# Compliance & Governance — CredAgent

> How CredAgent handles the obligations a fintech compliance team actually worries
> about: fair lending, adverse-action disclosure, PII minimization, auditability,
> and access control. Every claim below points at the code that implements it.
> Where something is **designed-for but not yet enforced**, it says so — honesty
> about the gap is part of the governance posture, not a marketing omission.

Regulatory frame: **ECOA / Regulation B (12 CFR 1002)**, **FCRA**, and CFPB
fair-lending guidance, applied to employer-sponsored installment lending.

---

## 1. Fair lending — prohibited-basis exclusion

**Protected characteristics are never model features.** The risk model trains on
19 credit/income/employment features ([models/model_metadata.json](models/model_metadata.json));
none is a protected basis. Gender is captured **only** in the decision store — and
explicitly excluded from scoring — so the compliance layer can test for disparate
impact. This is the exact separation a real compliance function maintains:

> *"Gender is intentionally NOT a model feature (it is a prohibited basis under
> ECOA). It is captured here only so the monitoring layer can run disparate-impact
> analysis."* — [src/store.py](src/store.py)

**Why keep the scoring model and fairness monitoring separate?** A model that can
*see* a protected attribute can learn to proxy it. Keeping fairness testing in a
separate, post-decision monitoring layer means the model cannot use the attribute,
while the institution can still prove it is not producing a disparate outcome.

### Disparate-impact testing (four-fifths rule)
[src/store.py](src/store.py) `disparate_impact()` computes the **adverse-impact
ratio** = (lowest group approval rate) / (highest group approval rate) across
gender groups. A ratio **below 0.80** raises a flag, per the conventional
four-fifths (80%) rule used by the EEOC/CFPB. This runs continuously over the
accumulated decisions and surfaces in the monitoring dashboard.

---

## 2. Adverse-action notices (ECOA §1002.9)

When a decision is DECLINE, the **AuditAgent** generates a complete, Regulation
B-compliant adverse-action notice ([src/agents/audit_agent.py](src/agents/audit_agent.py)).
The notice is engineered to meet the legal bar, not to look plausible:

- **Specific principal reasons** come directly from the model's **SHAP**
  attribution (`top_risk_factors`), satisfying the requirement to disclose the
  *specific* reasons for the action — not a generic template.
- The **verbatim ECOA anti-discrimination notice** is mandated in the system
  prompt and present in the deterministic fallback, so it is never paraphrased
  away.
- **No placeholders ship.** A regex strips any `[Applicant Name]`-style tokens
  the LLM might emit (`_strip_placeholders`), and a length check substitutes a
  clean template if the model produced something hollow. A half-rendered legal
  letter is treated as a defect.
- Scores and probabilities are **deliberately excluded** from the letter — the
  applicant receives disclosable reasons, not raw model internals.

---

## 3. Model explainability

Explainability is a **legal requirement** here, not a nice-to-have, because the
adverse-action reasons must be specific and defensible.

- **SHAP per-prediction attribution** ([src/ml/explainer.py](src/ml/explainer.py))
  is computed for every scored application and persisted into the evidence bundle.
- The explanation is derived from the **model math**, then narrated by the LLM —
  the LLM never invents reasons. Offline, reasons are generated deterministically
  from the structured state and labelled as such.
- XGBoost was chosen partly *because* it yields exact, fast SHAP values (see
  [ARCHITECTURE.md §2](ARCHITECTURE.md)).

---

## 4. Data minimization & PII

- **No direct identifiers are used or required.** Applications are keyed by an
  `applicant_id` reference; the model features are financial/credit-bureau signals
  ([models/model_metadata.json](models/model_metadata.json)), not names, SSNs, or
  contact details.
- **The decision store persists only what is needed** for audit and fair-lending
  monitoring ([src/store.py](src/store.py)): the reference id, financial inputs,
  the (excluded-from-scoring) gender attribute for disparate-impact testing, the
  decision, and the evidence bundle. There is no free-text PII column.
- **Adverse-action notices do not invent personal data** — recipients are
  addressed as "Dear Applicant," and bracketed placeholders are stripped, so no
  fabricated PII is ever emitted.
- **Demo data is real but de-identified** (Home Credit, via Hugging Face),
  consistent with the project's real-data-over-synthetic principle.

> **Logging note:** application logs record decision metadata, timings, and model
> versions — not raw application payloads. The evidence bundle (which does store
> feature inputs) lives in the access-controllable store, separate from stdout
> logs. Hardening item: a structured PII-scrubbing log filter is on the roadmap (§6).

---

## 5. Auditability — the AI Evidence Hub

Every decision is **reproducible after the fact**. `store.log_decision()` writes,
in one transaction, both the decision row and a full evidence bundle
([src/store.py](src/store.py) `_log_evidence`) containing:

- `model_version` and the resolved LLM `provider:model`,
- the **exact** system and user prompts for every LLM call (`evidence_record` in
  [src/llm.py](src/llm.py)),
- the cleaned + derived **feature inputs**,
- the **SHAP** attribution and `top_risk_factors`,
- the **retrieved CFPB policy excerpts** that informed the compliance check,
- a timestamped `audit_trail` of each agent step with latency.

This means a reviewer (or regulator) can answer "*why was this decision made, on
what data, by which model, citing which policy text, at what time?*" for any
single application — the defining property of an auditable lending system.

---

## 6. Access control & governance roadmap

Stated honestly:

| Control | Status |
|---------|--------|
| Prohibited-basis exclusion from scoring | **Implemented** ([src/store.py](src/store.py), feature set) |
| Disparate-impact (four-fifths) monitoring | **Implemented** ([src/store.py](src/store.py)) |
| SHAP-based adverse-action reasons | **Implemented** ([src/agents/audit_agent.py](src/agents/audit_agent.py)) |
| Reproducible evidence bundle per decision | **Implemented** ([src/store.py](src/store.py), [src/llm.py](src/llm.py)) |
| PSI model-drift monitoring | **Implemented** ([src/drift.py](src/drift.py)) |
| No-silent-fallback LLM degradation | **Implemented** ([src/llm.py](src/llm.py)) |
| Role-based access control (RBAC) on the API | **Designed-for, not enforced** — single-tenant demo; the store schema and endpoint surface are shaped so an auth layer (e.g. reviewer vs. analyst vs. auditor roles gating HITL overrides and evidence reads) drops in without restructuring. |
| Immutable/append-only audit log (WORM) | **Roadmap** — evidence rows are currently upsert-by-id; a true append-only audit table is the next hardening step. |
| PII-scrubbing structured log filter | **Roadmap** |
| Drift/fairness alerting to an external channel | **Roadmap** — currently surfaced in-dashboard. |

---

## 7. Testing the governance behaviors

The compliance behaviors are covered by hermetic tests (no network/keys required):
[tests/test_monitoring.py](tests/test_monitoring.py) exercises drift and
disparate-impact; [tests/test_agents.py](tests/test_agents.py) exercises the
agent pipeline including the adverse-action path. The principle is **test the
contract, not just the happy path** — the fair-lending and disclosure behaviors
are verified, not assumed.
