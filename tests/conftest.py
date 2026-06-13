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
