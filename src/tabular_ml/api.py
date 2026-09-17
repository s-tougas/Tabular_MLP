from src.tabular_ml.model import tabular_mlp
from src.tabular_ml.config import config
from src.tabular_ml.data import apply_transf
import torch
from pydantic import BaseModel
import joblib
from fastapi import FastAPI
import pandas as pd

PROD_DIR = config.artifacts_dir / "production"

scaler = joblib.load(PROD_DIR/ "scaler.pkl")
encoders = joblib.load(PROD_DIR / "encoders.pkl")
target_scaler = joblib.load(PROD_DIR / 'target_scaler.pkl')

c_categories = {col: len(encoders[col].classes_) for col in config.categorical_cols}
model = tabular_mlp(config, c_categories)
model.load_state_dict(torch.load(PROD_DIR / "model.pt"))
model.eval()

# TODO: pydantic's float type accepts NaN/Infinity/-Infinity as valid
# This leads to an error when calculating the dollar value
# There is a way to disable the ability to us NaN and Infinities so,
# that will likely be used in a fix.
class HouseFeatures(BaseModel):
    ...

class HouseFeatures(BaseModel):
    LotArea: float
    OverallQual: float
    OverallCond: float
    YearBuilt: float
    GrLivArea: float
    FullBath: float
    BedroomAbvGr: float
    GarageCars: float
    MSZoning: str
    Neighborhood: str
    HouseStyle: str
    SaleCondition: str

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(house: HouseFeatures):
    input_df = pd.DataFrame([house.dict()])
    processed_df = apply_transf(input_df, config, scaler, encoders)

    num_tensor = torch.tensor(processed_df[config.numeric_cols].values, dtype= torch.float32)
    cat_tensor = torch.tensor(processed_df[config.categorical_cols].values,dtype= torch.long)

    with torch.no_grad():
        scaled_pred = model(num_tensor, cat_tensor)
    prediction_figure = target_scaler.inverse_transform(scaled_pred.numpy())

    return {'predicted_price': float(prediction_figure[0][0])}



