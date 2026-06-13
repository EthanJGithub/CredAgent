import os

# Force the deterministic offline LLM for tests: assertions must not depend on a
# live model's exact wording or on network/API quota. Set before importing the
# app (load_dotenv uses override=False, so it won't replace these).
os.environ["GROQ_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

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
