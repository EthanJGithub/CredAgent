# CredAgent 🏦

> **Agentic Credit Decisioning & Risk Explainability System**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit)](YOUR_STREAMLIT_URL_HERE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://langchain-ai.github.io/langgraph/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![Tests](https://img.shields.io/badge/tests-15_passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A multi-agent AI pipeline for real-time loan underwriting in employer-sponsored
installment lending, modeled on the underwriting needs of platforms like
Purchasing Power / PROG Holdings. It scores an application with a gradient-boosted
risk model, explains every prediction with SHAP, checks the decision against CFPB
fair-lending regulations via RAG, routes borderline cases to a human reviewer, and
generates a legally-compliant adverse-action notice — in well under a second.

**[→ Try the live demo](YOUR_STREAMLIT_URL_HERE)**

---

## What It Does

In a single API call, CredAgent runs a five-agent [LangGraph](https://langchain-ai.github.io/langgraph/) pipeline:

1. **IngestionAgent** — validates and enriches the application (debt-to-income, employment length, credit ratios).
2. **RiskScoringAgent** — scores default risk with **XGBoost (validation ROC-AUC ≈ 0.76 on the real 307K-row dataset)** and attributes the prediction with **SHAP**.
3. **PolicyComplianceAgent** — retrieves relevant CFPB fair-lending text from a **ChromaDB** vector store and uses an LLM to flag disparate-impact / ECOA / FCRA concerns.
4. **DecisionAgent** — issues **APPROVE / DECLINE / REFER** with plain-English reasoning; **MEDIUM-risk** cases pause for **human-in-the-loop** review.
5. **AuditAgent** — generates a **CFPB-compliant adverse-action notice** (ECOA §1002.9) for declines and assembles a timestamped audit trail.

---

## Architecture

```mermaid
flowchart TD
    A([Loan Application]) --> B[LangGraph Pipeline]

    subgraph B[Multi-Agent Pipeline]
        direction TB
        D[IngestionAgent<br/>Validation + Feature Engineering]
        E[RiskScoringAgent<br/>XGBoost + SHAP]
        F[PolicyComplianceAgent<br/>RAG over CFPB Docs + LLM]
        G[DecisionAgent<br/>Approve / Decline / Refer + LLM]
        H[AuditAgent<br/>Adverse Notice + Audit Trail]
        D --> E --> F --> G
        G -->|MEDIUM tier| I{{Human-in-the-Loop<br/>Interrupt}}
        I --> G
        G --> H
    end

    B --> J([Decision · SHAP Chart · Adverse Notice · Audit Trail])
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph 0.2 (stateful graph + MemorySaver checkpointing for HITL) |
| LLM (reasoning + compliance) | Pluggable: **Groq / Llama 3** or **Anthropic Claude**, with a deterministic offline fallback |
| ML risk model | XGBoost 2.0 (gradient-boosted trees) |
| Explainability | SHAP TreeExplainer (per-prediction feature attribution) |
| Regulatory RAG | ChromaDB + ONNX `all-MiniLM-L6-v2` embeddings |
| API | FastAPI + Pydantic v2 |
| Dashboard | Streamlit (+ SHAP waterfall via matplotlib) |
| Dataset | **Home Credit Default Risk — real 307,511-row dataset** |
| Compliance corpus | ECOA / Regulation B §1002.9, FCRA §615, CFPB fair-lending & BNPL material |

---

## Model Performance

| Metric | Value |
|---|---|
| Algorithm | XGBoost (gradient-boosted trees, 500 estimators, early stopping) |
| Training data | **Real Home Credit dataset** — 246,008 train / 61,503 validation rows |
| Validation ROC-AUC | **0.7611** |
| Class handling | `scale_pos_weight ≈ 11.4` (8.1% default base rate) |
| Features | 22 engineered financial features |
| Explainability | SHAP — per-prediction feature attribution |
| End-to-end latency | ~0.4 s (offline reasoning) / ~1–2 s (with LLM) |

The trained model card is in [`models/model_metadata.json`](models/model_metadata.json), which records the data source and `trained_on_real_data` flag for full transparency.

---

## Quickstart (Local)

```bash
git clone https://github.com/YOUR_USERNAME/credagent.git
cd credagent
python -m venv .venv && . .venv/Scripts/activate    # Windows
pip install -r requirements.txt

# (Optional) add a free Groq key for LLM-authored text; runs fine without one.
cp .env.example .env

# 1. Fetch the REAL Home Credit dataset (auto-downloads ~40 MB, no Kaggle account)
python -m src.ml.get_data

# 2. Train the XGBoost + SHAP model (~1–2 min)
python -m src.ml.train

# 3. Build the CFPB regulatory vector store (~1 min)
python -m src.rag.ingest

# 4a. Run the API
uvicorn src.api.main:app --reload          # http://localhost:8000/docs

# 4b. Run the dashboard (separate terminal)
streamlit run src/dashboard/app.py         # http://localhost:8501

# Tests
pytest                                       # 15 passing
```

> The repo ships with the trained `models/` and `vectorstore/` committed, so the
> Streamlit Cloud demo (and `streamlit_app.py`) work without steps 1–3.

### Data sourcing

`python -m src.ml.get_data` prefers **real data** in this order: an existing real
CSV at `data/raw/application_train.csv` → the public HuggingFace mirror of the
genuine Home Credit dataset → the Kaggle API (if credentials are configured) →
a clearly-flagged synthetic fallback used only when fully offline. The model in
this repo was trained on the **real** dataset.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`  | `/api/v1/health` | Service + model + vectorstore status |
| `POST` | `/api/v1/decisions` | Submit an application for decisioning |
| `GET`  | `/api/v1/decisions/{id}` | Retrieve a prior decision |
| `POST` | `/api/v1/decisions/{id}/human-review` | Submit a reviewer's decision for a REFER case |
| `GET`  | `/api/v1/model/info` | Model version, features, training AUC, thresholds |

---

## Compliance Framework

The `PolicyComplianceAgent` and `AuditAgent` apply guidance retrieved at inference time:

- **ECOA / Regulation B (12 CFR 1002.9)** — adverse-action notice with *specific* principal reasons.
- **FCRA §615 (15 U.S.C. 1681m)** — consumer-report adverse-action duties.
- **CFPB fair-lending principles** — disparate treatment vs. disparate impact; model-risk expectations for algorithmic underwriting.
- **CFPB BNPL market report** — installment-lending regulatory context.

Protected-class attributes and their close proxies are excluded from the decision; adverse-action reasons are derived from SHAP attributions over permissible financial factors.

---

## Project Structure

```
credagent/
├── streamlit_app.py          # Cloud entry point (runs pipeline inline)
├── src/
│   ├── agents/               # 5 LangGraph agents
│   ├── graph/                # State schema + workflow (HITL checkpointing)
│   ├── ml/                   # get_data, features, train, explainer
│   ├── rag/                  # CFPB corpus, ChromaDB ingest + retriever
│   ├── api/                  # FastAPI app, routes, schemas
│   ├── dashboard/            # Streamlit (connects to FastAPI)
│   ├── llm.py                # Pluggable Groq/Anthropic/offline LLM
│   └── demo_presets.py       # Real applicants for the demo tiers
├── models/                   # Trained XGBoost + SHAP + metadata (committed)
├── vectorstore/              # ChromaDB CFPB embeddings (committed)
└── tests/                    # pytest suite (15 tests)
```

---

## Notable Engineering Decisions

- **Real data, automatically.** `get_data.py` pulls the genuine 307K-row Home Credit dataset from a public mirror — no Kaggle gate — and records provenance in the model card.
- **Runs with zero API keys.** The LLM layer degrades gracefully to deterministic, template-based reasoning so the full pipeline (and CI) never blocks on a credential or rate limit.
- **Lightweight embeddings.** ChromaDB's ONNX `all-MiniLM-L6-v2` avoids a ~2 GB PyTorch dependency, keeping installs and cloud deploys fast.
- **Stateful HITL.** LangGraph's `MemorySaver` checkpointer lets a MEDIUM-tier decision pause and resume on the same `thread_id` when a human ruling arrives.

---

## License

MIT — see [LICENSE](LICENSE).
