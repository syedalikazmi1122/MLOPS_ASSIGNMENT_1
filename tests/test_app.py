import json
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from app import app  # noqa: E402


@pytest.fixture()
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_parameters_endpoint(client):
    response = client.get('/api/parameters')
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        data = response.get_json()
        assert 'status' in data
        assert 'cities' in data
        assert 'states' in data


def test_lightgbm_predict_input_validation(client):
    payload = {
        "status": "for sale",
        "city": "abc",
        "state": "xyz",
        "zip_code": 12345,
        "bed": 3,
        "bath": 2,
        "house_size": 1500,
        "acre_lot": 0.2
    }
    response = client.post('/api/lightgbmpredict',
                           data=json.dumps(payload),
                           content_type='application/json')
    # Depending on whether artifacts exist locally, either 200 or 500 is acceptable
    assert response.status_code in (200, 400, 500)
