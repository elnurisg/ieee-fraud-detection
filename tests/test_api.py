import os
import json

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from fastapi.testclient import TestClient
from app.backend.main import app  # Adjust if necessary

client = TestClient(app)

# Define the path to the request sample file relative to the project root.
# For example, if the file is at /app/request_samples/request_sample.json, then:
REQUEST_SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "request_samples")

def test_predict_endpoint_valid_input():
    sample_file = os.path.join(REQUEST_SAMPLES_PATH, "request_sample.json")
    # Check if the sample file exists
    assert os.path.isfile(sample_file), f"Sample request file not found at {sample_file}"
    with open(sample_file, "r") as f:
        payload = json.load(f)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    assert 0.0 <= data["prediction"] <= 1.0


def test_predict_endpoint_invalid_input():
    payload = {"invalid_key": "invalid_value"}
    response = client.post("/predict", json=payload)
    # FastAPI returns 422 for validation errors.
    assert response.status_code == 422, response.text
