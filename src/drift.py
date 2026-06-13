"""Model / data drift detection via Population Stability Index (PSI).

PSI compares the distribution of a feature (or the model's output score) in a
**reference** window (the training/validation data, captured at train time) to a
**current** window (recent scored decisions from the store). It is the standard
model-monitoring metric for detecting covariate / score drift that would signal
the model is operating on a population it was not trained for.

    PSI = Σ (cur% − ref%) · ln(cur% / ref%)

Conventional thresholds:
    PSI < 0.10  → stable
    0.10–0.25   → moderate shift (investigate)
    > 0.25      → significant shift (retrain / alert)
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np

from src import store

REFERENCE_PATH = os.getenv("CREDAGENT_DRIFT_REF", "models/drift_reference.json")

# Fields tracked for drift (must exist as columns in the decisions store).
TRACKED = ["risk_probability", "amt_income_total", "amt_credit", "amt_annuity"]
_EPS = 1e-6


def build_reference(values: Dict[str, Sequence[float]], n_bins: int = 10) -> dict:
    """Compute quantile-bin edges + reference proportions and persist them."""
    ref = {}
    for name, arr in values.items():
        a = np.asarray([v for v in arr if v is not None], dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([a.min(), a.max() + 1e-9])
        counts, _ = np.histogram(a, bins=edges)
        props = counts / max(counts.sum(), 1)
        ref[name] = {"edges": edges.tolist(), "ref_props": props.tolist()}
    os.makedirs(os.path.dirname(REFERENCE_PATH) or ".", exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(ref, f, indent=2)
    return ref


def _load_reference() -> dict:
    if not os.path.exists(REFERENCE_PATH):
        return {}
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def _psi(ref_props: List[float], edges: List[float], current: Sequence[float]) -> float:
    cur = np.asarray([v for v in current if v is not None], dtype=float)
    cur = cur[np.isfinite(cur)]
    if cur.size == 0:
        return 0.0
    counts, _ = np.histogram(cur, bins=np.asarray(edges))
    cur_props = counts / max(counts.sum(), 1)
    ref = np.asarray(ref_props) + _EPS
    cur_props = cur_props + _EPS
    return float(np.sum((cur_props - ref) * np.log(cur_props / ref)))


def _status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def drift_report() -> dict:
    """Compute PSI for each tracked field over the current stored decisions."""
    reference = _load_reference()
    rows = store.fetch_all()
    features = []
    worst = 0.0
    for name in TRACKED:
        if name not in reference:
            continue
        current = [r.get(name) for r in rows]
        psi = round(_psi(reference[name]["ref_props"], reference[name]["edges"], current), 4)
        worst = max(worst, psi)
        features.append({"feature": name, "psi": psi, "status": _status(psi)})
    return {
        "reference_available": bool(reference),
        "n_current": len(rows),
        "max_psi": round(worst, 4),
        "overall_status": _status(worst) if features else "no-reference",
        "features": features,
    }
