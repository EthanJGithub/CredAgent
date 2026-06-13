"""Acquire the **real** Home Credit Default Risk training data.

Priority chain (real data is always preferred — synthetic is a last resort):

1. **Existing real file** — if ``data/raw/application_train.csv`` already exists
   and looks like the genuine dataset (has ``TARGET`` and >100k rows), use it.
   This is where you would drop the official Kaggle ``application_train.csv``.
2. **HuggingFace mirror** — download the real 307,511-row dataset from the
   public ``jlh/home-credit`` parquet mirror (no Kaggle account required) and
   write it to the CSV path. This is the default, fully-automated real path.
3. **Kaggle API** — if HuggingFace is unreachable but Kaggle credentials are
   configured (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``),
   pull the official competition file.
4. **Synthetic fallback** — only if every real source fails (e.g. fully offline
   CI). Clearly flagged in ``dataset_source.json`` and the model metadata so it
   is never mistaken for real performance.

Run directly to fetch real data ahead of training:
    python -m src.ml.get_data
"""
from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from typing import Tuple

import pandas as pd

from src.ml.features import RAW_COLUMNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/raw/application_train.csv"
SOURCE_PATH = "data/raw/dataset_source.json"

# Public parquet mirror of the genuine Kaggle Home Credit application_train.
HF_PARQUET_URL = (
    "https://huggingface.co/datasets/jlh/home-credit/resolve/main/"
    "data/train-00000-of-00001-e68d01965482ae18.parquet"
)
KAGGLE_COMPETITION = "home-credit-default-risk"
MIN_REAL_ROWS = 100_000


def _write_source(source: str, n_rows: int, real: bool) -> None:
    with open(SOURCE_PATH, "w") as f:
        json.dump({"source": source, "n_rows": int(n_rows), "is_real": bool(real)}, f, indent=2)


def read_source() -> dict:
    if os.path.exists(SOURCE_PATH):
        with open(SOURCE_PATH) as f:
            return json.load(f)
    return {"source": "unknown", "n_rows": 0, "is_real": False}


def _looks_real(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        head = pd.read_csv(path, nrows=5)
        if "TARGET" not in head.columns:
            return False
        # cheap row count without loading everything
        with open(path, "rb") as f:
            n = sum(1 for _ in f) - 1
        return n >= MIN_REAL_ROWS
    except Exception:
        return False


def _project_and_save(df: pd.DataFrame, source: str) -> Tuple[str, int]:
    """Keep only the columns the model uses, coerce TARGET, and persist."""
    keep = [c for c in RAW_COLUMNS if c in df.columns]
    out = df[keep].copy()
    out["TARGET"] = pd.to_numeric(out["TARGET"], errors="coerce").fillna(0).astype(int)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    out.to_csv(DATA_PATH, index=False)
    _write_source(source, len(out), real=True)
    logger.info("Saved %s real rows from %s -> %s", f"{len(out):,}", source, DATA_PATH)
    return DATA_PATH, len(out)


def _download_huggingface() -> Tuple[str, int]:
    import requests

    logger.info("Downloading real Home Credit data from HuggingFace mirror...")
    resp = requests.get(HF_PARQUET_URL, timeout=180)
    resp.raise_for_status()
    df = pd.read_parquet(io.BytesIO(resp.content))
    logger.info("Downloaded parquet: %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return _project_and_save(df, source="huggingface:jlh/home-credit")


def _download_kaggle() -> Tuple[str, int]:
    have_file = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    have_env = os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    if not (have_file or have_env):
        raise RuntimeError("No Kaggle credentials configured.")

    logger.info("Downloading real data via Kaggle API...")
    os.makedirs("data/raw", exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download",
         "-c", KAGGLE_COMPETITION, "-f", "application_train.csv", "-p", "data/raw"],
        check=True,
    )
    # Kaggle delivers a .zip; unzip if needed.
    import glob
    import zipfile
    for z in glob.glob("data/raw/application_train.csv*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall("data/raw")
        os.remove(z)
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in RAW_COLUMNS)
    return _project_and_save(df, source="kaggle:home-credit-default-risk")


def ensure_dataset(allow_synthetic: bool = True) -> Tuple[str, int, bool]:
    """Guarantee a training CSV exists. Returns (path, n_rows, is_real)."""
    if _looks_real(DATA_PATH):
        src = read_source()
        n = src.get("n_rows") or sum(1 for _ in open(DATA_PATH, "rb")) - 1
        logger.info("Using existing real dataset (%s rows).", f"{n:,}")
        return DATA_PATH, n, True

    for fetch in (_download_huggingface, _download_kaggle):
        try:
            path, n = fetch()
            return path, n, True
        except Exception as exc:
            logger.warning("%s failed: %s", fetch.__name__, exc)

    if not allow_synthetic:
        raise RuntimeError("Could not obtain real data and synthetic fallback is disabled.")

    logger.warning("All real sources failed — generating SYNTHETIC fallback data.")
    from src.ml.make_synthetic_data import generate
    df = generate(rows=60000)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    _write_source("synthetic-fallback", len(df), real=False)
    return DATA_PATH, len(df), False


if __name__ == "__main__":
    path, n, is_real = ensure_dataset()
    print(f"\nDataset ready: {path}\nRows: {n:,}\nReal data: {is_real}")
