import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from api.main import app
load_dotenv()

client = TestClient(app)

def load_sample(type = 'normal'):
    if type in ['normal', 'missing', 'wrong']:
        file_path = (
            Path(__file__).resolve().parent.parent
            / "tests"
            / f"{type}_sample.json"
        )
        with open(file_path, "r") as f:
            data = f.read()
    else:
        raise ValueError("Type must be 'normal', 'missing' or 'wrong'")

    return json.loads(data)

@pytest.fixture
def load_normal_sample():
    return load_sample('normal')

@pytest.fixture
def load_missing_sample():
    return load_sample('missing')

class TestApi:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "Alive !"}

    def test_predict_with_wrong_api_key(self):
        response = client.post("/predict", headers={"x-api-key": "foo"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Forbidden"}

    def test_predict(self, load_normal_sample):
        response = client.post("/predict", json=load_normal_sample, headers={"x-api-key": os.getenv('API_KEY')})

        response_keys_needed = {"probability", "prediction", "sk_id_curr"}

        assert response.status_code == 200
        assert response_keys_needed <= response.json().keys()
        assert response_keys_needed == response.json().keys()

    def test_predict_with_missing_fields(self, load_missing_sample):

        response = client.post("/predict", json=load_missing_sample, headers={"x-api-key": os.getenv('API_KEY')})

        assert response.status_code == 422
