import pandas as pd
import pytest
from src.tabular_ml.config import Config
from src.tabular_ml.data import clean, split_data, fit_transform_data, apply_transf

@pytest.fixture

def sample_df():
    return pd.DataFrame({

        "LotArea": [7000, 9600, None, 11250, 1230],
        "OverallQual": [7,6,7,7,8],
        "OverallCond": [5,8,5,5,5],
        "YearBuilt": [2003, 1999, 2001, 1912, 1945],
        "GrLivArea": [1710, 1262, 1789, 1717, 2198],
        "FullBath": [2,2,2,1,2],
        "BedroomAbvGr": [3,3,3,3,4],
        "GarageCars": [2,2,2,3,3],
        "MSZoning": ["RL","RL", None, "RL", "RL"],
        "Neighborhood": ["CollgCr", "Veenker", "CollgCr", "Crawfor", "NoRidge"],
        "HouseStyle": ["2Story", "1Story", "2Story", "2Story", "2Story"],
        "SaleCondition": ["Normal","Normal", "Normal", "Abnormal", "Abnormal"],
        "SalePrice": [208500,181500, 234500, 132000,860000],
    })

@pytest.fixture()
def sample_config():
    return Config(
        numeric_cols=["LotArea", "OverallQual", "OverallCond", "YearBuilt", "GrLivArea","FullBath", "BedroomAbvGr", "GarageCars"],
        categorical_cols= ["MSZoning", "Neighborhood", "HouseStyle", "SaleCondition"],
        target_col= "SalePrice"
    )

def test_clean_fills_missing_values(sample_df,sample_config):
    test_clean = clean(sample_df,sample_config)
    assert test_clean.isna().sum().sum() == 0


def test_split_data_sum_correct(sample_df,sample_config):
    train, val, test = split_data(sample_df,sample_config)
    assert len(train) + len(val) + len(test) == len(sample_df)

def test_split_data_no_overlapping(sample_df,sample_config):
    train,val, test = split_data(sample_df,sample_config)
    train_idx = set(train.index)
    val_idx = set(val.index)
    test_idx = set(test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)

def test_fit_transform_data_scales_numeric_cols(sample_df,sample_config):
    clean_df = clean(sample_df,sample_config)
    ret, scalar, encoder= fit_transform_data(clean_df,sample_config)

    for col in sample_config.numeric_cols:
        assert ret[col].mean() == pytest.approx(0, abs= 1e-8)
        assert ret[col].std(ddof=0) == pytest.approx(1, abs = 1e-2)
#python -m pytest tests/test_data.py -v
def test_apply_transf_uses_train_fitted_encoders(sample_df,sample_config):
    clean_df = clean(sample_df,sample_config)
    train_part = clean_df
    other_part = clean_df.iloc[[0,1]]

    train_df,u_scalar,u_encoder = fit_transform_data(train_part,sample_config)
    other_df = apply_transf(other_part,sample_config, u_scalar, u_encoder)
    #Other_df has transformed correctly
    assert other_df.isna().sum().sum() == 0
    #Checks to see if categories have been encoded 
    assert other_df[sample_config.categorical_cols].dtypes.apply(lambda d: d == "int64").all()
    #Checks to see if scaling has occurred 
    assert other_df[sample_config.numeric_cols].abs().max().max()  < 100