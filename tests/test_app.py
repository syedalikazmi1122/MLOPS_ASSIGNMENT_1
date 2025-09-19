import json
import os
import sys
from unittest.mock import patch, MagicMock

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


@patch('app.joblib.load')
def test_lightgbm_predict_input_validation(mock_load, client):
    # Mock the model and encoders
    mock_model = MagicMock()
    mock_model.predict.return_value = [10.5]  # log price prediction
    mock_encoders = {
        'status': MagicMock(),
        'city': MagicMock(),
        'state': MagicMock()
    }
    mock_encoders['status'].classes_ = ['for sale', 'sold']
    mock_encoders['city'].classes_ = ['los angeles', 'new york']
    mock_encoders['state'].classes_ = ['california', 'new york']
    mock_encoders['status'].transform.return_value = [0]
    mock_encoders['city'].transform.return_value = [0]
    mock_encoders['state'].transform.return_value = [0]
    
    mock_load.side_effect = lambda path: mock_model if 'model' in path else mock_encoders
    
    payload = {
        "status": "for sale",
        "city": "los angeles",
        "state": "california",
        "zip_code": 12345,
        "bed": 3,
        "bath": 2,
        "house_size": 1500,
        "acre_lot": 0.2
    }
    response = client.post('/api/lightgbmpredict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'predicted_price' in data
    assert 'comparableProperties' in data
