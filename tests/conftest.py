import os
import tempfile

# Force the deterministic offline LLM for tests: assertions must not depend on a
# live model's exact wording or on network/API quota. Set before importing the
# app (load_dotenv uses override=False, so it won't replace these).
os.environ["GROQ_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
# Isolate the decision store to a temp DB and skip startup seeding for fast,
# side-effect-free tests.
os.environ["CREDAGENT_DB"] = os.path.join(tempfile.gettempdir(), "credagent_test.db")
os.environ["CREDAGENT_AUTOSEED"] = "0"
# Start each test session from a clean store.
for _suffix in ("", "-wal", "-shm"):
    _p = os.environ["CREDAGENT_DB"] + _suffix
    if os.path.exists(_p):
        os.remove(_p)

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_applications():
    with open("tests/fixtures/sample_applications.json") as f:
        return json.load(f)
