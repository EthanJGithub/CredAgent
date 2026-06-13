"""Persistent decision store (SQLite).

Every decision the pipeline produces is written here so the system behaves like
a real underwriting platform: decisions accumulate, can be retrieved and
exported for audit, and feed a fair-lending monitoring view.

Gender is intentionally NOT a model feature (it is a prohibited basis under
ECOA). It is captured here only so the monitoring layer can run **disparate-
impact analysis** — comparing approval rates across groups using the
four-fifths (80%) rule — which is exactly the separation a real compliance team
maintains between the scoring model and fair-lending testing.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.getenv("CREDAGENT_DB", "data/decisions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    applicant_id          TEXT PRIMARY KEY,
    ts                    TEXT,
    code_gender           TEXT,
    amt_income_total      REAL,
    amt_credit            REAL,
    amt_annuity           REAL,
    name_income_type      TEXT,
    risk_probability      REAL,
    risk_tier             TEXT,
    final_decision        TEXT,
    credit_limit          REAL,
    requires_human_review INTEGER,
    human_decision        TEXT,
    compliance_flags      TEXT,
    source                TEXT
);
CREATE TABLE IF NOT EXISTS evidence (
    applicant_id  TEXT PRIMARY KEY,
    ts            TEXT,
    evidence_json TEXT
);
"""


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def count() -> int:
    try:
        with _connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def log_decision(state: Dict[str, Any], source: str = "live") -> None:
    """Upsert one decision keyed by applicant_id (so HITL resumes update in place)."""
    init_db()
    raw = state.get("raw_application", {}) or {}
    row = {
        "applicant_id": state.get("applicant_id"),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "code_gender": str(raw.get("code_gender", "")).upper() or None,
        "amt_income_total": raw.get("amt_income_total"),
        "amt_credit": raw.get("amt_credit"),
        "amt_annuity": raw.get("amt_annuity"),
        "name_income_type": raw.get("name_income_type"),
        "risk_probability": state.get("risk_probability"),
        "risk_tier": state.get("risk_tier"),
        "final_decision": state.get("final_decision") or (
            "REFER" if state.get("requires_human_review") else None
        ),
        "credit_limit": state.get("credit_limit"),
        "requires_human_review": int(bool(state.get("requires_human_review"))),
        "human_decision": state.get("human_decision"),
        "compliance_flags": json.dumps(state.get("compliance_flags", [])),
        "source": source,
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "applicant_id")
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO decisions ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(applicant_id) DO UPDATE SET {updates}",
            row,
        )
    _log_evidence(state, row)


def _log_evidence(state: Dict[str, Any], row: Dict[str, Any]) -> None:
    """AI Evidence Hub: persist the full governance bundle for a decision —
    model version, exact LLM prompts, feature inputs, SHAP attribution, and the
    retrieved policy excerpts — so any decision is fully reproducible/auditable."""
    bundle = {
        "applicant_id": row["applicant_id"],
        "ts": row["ts"],
        "model_version": state.get("model_version"),
        "final_decision": row["final_decision"],
        "risk_tier": row["risk_tier"],
        "risk_probability": row["risk_probability"],
        "feature_inputs": state.get("cleaned_features"),
        "derived_features": state.get("derived_features"),
        "shap_values": state.get("shap_values"),
        "top_risk_factors": state.get("top_risk_factors"),
        "retrieved_policy_excerpts": state.get("retrieved_policy_excerpts", []),
        "llm_calls": state.get("llm_calls", []),
    }
    with _connect() as conn:
        conn.execute(
            "INSERT INTO evidence (applicant_id, ts, evidence_json) VALUES (?, ?, ?) "
            "ON CONFLICT(applicant_id) DO UPDATE SET ts=excluded.ts, evidence_json=excluded.evidence_json",
            (row["applicant_id"], row["ts"], json.dumps(bundle, default=str)),
        )


def fetch_evidence(applicant_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        r = conn.execute(
            "SELECT evidence_json FROM evidence WHERE applicant_id = ?", (applicant_id,)
        ).fetchone()
    return json.loads(r["evidence_json"]) if r else None


def fetch_recent(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_all() -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM decisions ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]


def summary() -> Dict[str, Any]:
    """Aggregate portfolio metrics for the monitoring dashboard."""
    rows = fetch_all()
    total = len(rows)
    out: Dict[str, Any] = {
        "total": total,
        "decision_counts": {"APPROVE": 0, "DECLINE": 0, "REFER": 0},
        "tier_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "DECLINE": 0},
        "approval_rate": 0.0,
        "avg_default_probability": 0.0,
        "fair_lending": disparate_impact(rows),
    }
    if not total:
        return out
    probs = []
    for r in rows:
        d = r.get("final_decision")
        if d in out["decision_counts"]:
            out["decision_counts"][d] += 1
        t = r.get("risk_tier")
        if t in out["tier_counts"]:
            out["tier_counts"][t] += 1
        if r.get("risk_probability") is not None:
            probs.append(r["risk_probability"])
    out["approval_rate"] = round(out["decision_counts"]["APPROVE"] / total, 4)
    out["avg_default_probability"] = round(sum(probs) / len(probs), 4) if probs else 0.0
    return out


def disparate_impact(rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Four-fifths (80%) rule across gender groups.

    For each group: approval rate = APPROVE / total. The adverse-impact ratio is
    the lowest group rate divided by the highest. A ratio below 0.80 is the
    conventional flag for potential disparate impact.
    """
    if rows is None:
        rows = fetch_all()
    groups: Dict[str, Dict[str, int]] = {}
    for r in rows:
        g = (r.get("code_gender") or "?").upper()
        if g not in ("M", "F"):
            continue
        groups.setdefault(g, {"total": 0, "approved": 0})
        groups[g]["total"] += 1
        if r.get("final_decision") == "APPROVE":
            groups[g]["approved"] += 1

    by_group = {
        g: {
            "total": v["total"],
            "approved": v["approved"],
            "approval_rate": round(v["approved"] / v["total"], 4) if v["total"] else 0.0,
        }
        for g, v in groups.items()
    }
    rates = [v["approval_rate"] for v in by_group.values() if v["total"] >= 1]
    if len(rates) >= 2 and max(rates) > 0:
        ratio = round(min(rates) / max(rates), 4)
        flag = ratio < 0.80
    else:
        ratio, flag = None, False
    return {
        "by_group": by_group,
        "adverse_impact_ratio": ratio,
        "flag": flag,
        "rule": "four-fifths (80%) rule",
    }
