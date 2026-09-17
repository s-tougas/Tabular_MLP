import pandas as pd
import pytest
import torch
from src.tabular_ml.config import Config
from src.tabular_ml.model import tabular_mlp

@pytest.fixture
def sample_config():
    return Config(
        numeric_cols=["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea", "FullBath", "BedroomAbvGr", "GarageCars"],
        categorical_cols=["MSZoning", "Neighborhood", "HouseStyle", "SaleCondition"],
        target_col="SalePrice",
        hidden_dims=[128, 64],
        embedding_dim_cap=50,
        dropout=0.2
    )

@pytest.fixture
def sample_categories():
    return{
        "MSZoning": 5,
        "Neighborhood": 25,
        "HouseStyle": 8,
        "SaleCondition": 6,
    }

@pytest.fixture
def sample_batch():
    batch_size =4
    numeric_input = torch.randn(batch_size,8)
    categorical_input = torch.tensor([
        [3,5,5,4],
        [3,24,2,4],
        [3,5,5,0],
        [3,15,5,4]
    ])
    return numeric_input, categorical_input

def test_model_output_shape(sample_batch,sample_categories,sample_config):
    numeric_input, categorical_input = sample_batch
    test_model = tabular_mlp(sample_config,sample_categories)
    output = test_model(numeric_input,categorical_input)
    assert output.shape == (4,1)

