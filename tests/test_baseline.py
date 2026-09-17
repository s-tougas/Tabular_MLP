import pandas as pd
import numpy as np 
import pytest

from src.tabular_ml.config import Config
from src.tabular_ml.baseline import train_xgboost

@pytest.fixture

def sample_config():
    return Config(
        numeric_cols=["LotArea", "OverallQual"],
        categorical_cols=["MSZoning"],
        target_col="SalePrice",
        xgb_n_estimators=10,
        xgb_random_state=42,
    )


@pytest.fixture
def train_vals_dfs(sample_config):
     rng = np.random.default_rng(42)
     n = 30

     def make_df():
          return pd.DataFrame({
            "LotArea": rng.integers(1000, 20000, n),
            "OverallQual": rng.integers(1, 10, n),
            "MSZoning": rng.integers(0, 4, n),  
            "SalePrice": rng.integers(100000, 400000, n),
        })
     return make_df(), make_df()

def test_xgboost_ret_Model_ret_rmse(sample_config,train_vals_dfs):
    train, val = train_vals_dfs
    model, rmse=train_xgboost(train,val,sample_config)
    sample_prediction = model.predict(train[["LotArea", "OverallQual","MSZoning"]])

    assert rmse > 0 
    assert len(sample_prediction) > 0
    
    