from fastapi.testclient import TestClient
from src.tabular_ml.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status" :"ok"}

def test_predict_valid_input():
    sample = {
  "LotArea": 4600,
  "OverallQual": 7,
  "OverallCond": 6,
  "YearBuilt": 1986,
  "GrLivArea": 1717,
  "FullBath": 2,
  "BedroomAbvGr": 2,
  "GarageCars": 2,
  "MSZoning": "RL",
  "Neighborhood": "CollgCr",
  "HouseStyle": "2Story",
  "SaleCondition": "Normal"
    }
    response = client.post("/predict", json= sample)
    assert response.status_code ==200 
    assert response.json()["predicted_price"] > 0 
# To fix NaN and infinity will run and break the API
# Need to make sure to set up handlers
def test_predict_invalid_input():
    sample = {
          "LotArea": "not a number",
          "OverallQual": 7,
          "OverallCond": 6,
          "YearBuilt": 1986,
          "GrLivArea": 1717,
          "FullBath": 2,
          "BedroomAbvGr": 2,
          "GarageCars": 2,
          "MSZoning": "RL",
          "Neighborhood": "CollgCr",
          "HouseStyle": "2Story",
          "SaleCondition": "Normal"
        }
    response = client.post("/predict", json=sample)
    assert response.status_code == 422