import pytest
from httpx import AsyncClient
from app import main

@pytest.mark.asyncio
async def test_root_endpoint():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Flood Probability Prediction API!"}

@pytest.mark.asyncio
async def test_prediction_endpoint():
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        payload = {
            "precipitation_sequence": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        response = await client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "flood_probability" in data
    assert 0 <= data["flood_probability"] <= 1
